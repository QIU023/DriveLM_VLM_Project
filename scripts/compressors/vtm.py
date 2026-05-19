"""VTM (Video Token Merging) cross-frame compressor.

References
----------
* Bolya, Fu, Dai, Zhang, Hoffman, Feichtenhofer. "Token Merging: Your ViT but
  Faster." ICLR 2023. Original *image* bipartite soft-matching token merger
  (a.k.a. ToMe). https://arxiv.org/abs/2210.09461
* Choi et al. "Video Token Merging for Long-Form Video Understanding."
  NeurIPS 2024 (VTM). Extends ToMe across the time axis -- flatten
  ``(T, N)`` and let the bipartite matcher merge spatio-temporally similar
  tokens regardless of which frame they came from.
* Touvron et al. "VATLM: Visual-Audio-Text Pre-Training with Unified Masked
  Prediction for Speech Representation Learning" (2024) -- similar idea of
  temporally-aware token reduction.

This module re-implements the VTM-style merge for the Qwen2.5-VL VLA
training stack. It is *not* a verbatim port of any one paper; the bipartite
split + cosine-similarity match + top-r-keep + mean/weighted-merge recipe is
taken directly from ToMe (image), and the time-axis flattening + flat output
budget is taken from VTM.

Signature: ``(B, T, N, D) -> (B, N', D)`` with ``N' = N`` (default), so LM
step time matches the temporal-mean baseline.

Implementation notes
--------------------
* Bipartite split is done with simple even/odd index slicing on the flattened
  ``T*N`` axis. ToMe uses the same trick (their "alternating" rule).
* Similarity is cosine, computed in fp32 for numerical stability with bf16
  visual encoders, then cast back.
* We optionally mix in a *learnable* embedding ``key_proj(x)`` so the model
  can learn a similarity metric distinct from raw feature cosine -- useful
  when the visual encoder is frozen.
* Everything is vectorised with ``torch.gather`` / boolean masking. No
  python loops over tokens (FSDP/CUDA-friendly).
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from . import CrossFrameCompressor, register


_VALID_MERGE_TYPES = ("mean", "weighted")


@register("vtm")
class VTMCompressor(CrossFrameCompressor):
    """Bipartite-merge cross-frame compressor (ToMe-on-time).

    Parameters
    ----------
    target_tokens : int
        Target output token count ``N'``. Default = ``N`` (set per-instance;
        callers typically pass the same N as their per-frame token count).
    merge_type : {"mean", "weighted"}
        ``"mean"``     -- merged tokens are equal-weight averaged (ToMe).
        ``"weighted"`` -- merged tokens are weighted by their cosine
        similarity to the destination (sharper merges for near-duplicates).
    dim : int, optional
        Feature dim ``D``. Required when ``use_learnable_key=True`` so we can
        allocate the projection.
    use_learnable_key : bool, default ``False``
        If True, similarity is computed in a learned key-space rather than
        on raw features. Adds a single ``nn.Linear(D, D, bias=False)``.
    """

    def __init__(
        self,
        target_tokens: int,
        merge_type: str = "mean",
        dim: Optional[int] = None,
        use_learnable_key: bool = False,
    ) -> None:
        super().__init__()
        if target_tokens < 1:
            raise ValueError(f"target_tokens must be >= 1, got {target_tokens}")
        if merge_type not in _VALID_MERGE_TYPES:
            raise ValueError(
                f"merge_type must be one of {_VALID_MERGE_TYPES}, got {merge_type!r}"
            )
        self.target_tokens = int(target_tokens)
        self.merge_type = merge_type
        self.use_learnable_key = use_learnable_key

        if use_learnable_key:
            if dim is None or dim < 1:
                raise ValueError("use_learnable_key=True requires dim >= 1")
            self.key_proj = nn.Linear(dim, dim, bias=False)
            # init close to identity so behaviour starts as raw-cosine ToMe.
            with torch.no_grad():
                self.key_proj.weight.copy_(torch.eye(dim))
        else:
            self.key_proj = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _bipartite_split(
        M: int, device: torch.device, a_size: Optional[int] = None
    ) -> tuple[Tensor, Tensor]:
        """Return (idx_A, idx_B) over a flat axis of length M.

        When ``a_size`` is ``None`` (the ToMe default for moderate compression
        ratios), A := even positions and B := odd positions, i.e. ``|A| ~ M/2``.

        When ``a_size`` is given (used for heavy compression where the target
        is much smaller than ``M/2``), A is the stride-k subset that produces
        ``|A| == a_size`` and B is the rest. This is the same trick as VTM's
        time-aware "anchor" selection -- it just spreads A more sparsely so a
        single merge step can shrink M -> a_size in one pass.
        """
        all_idx = torch.arange(M, device=device)
        if a_size is None:
            return all_idx[0::2], all_idx[1::2]
        if a_size <= 0 or a_size >= M:
            raise ValueError(f"a_size must be in (0, M); got a_size={a_size}, M={M}")
        # Evenly-spaced A indices in [0, M).
        idx_a = torch.linspace(
            0, M - 1, steps=a_size, device=device
        ).round().long().unique()
        # Edge: round/unique may collapse adjacent picks; fall back to a fresh
        # contiguous stride if we lost tokens.
        if idx_a.numel() != a_size:
            stride = max(1, M // a_size)
            idx_a = torch.arange(0, M, stride, device=device)[:a_size]
        a_mask = torch.zeros(M, dtype=torch.bool, device=device)
        a_mask[idx_a] = True
        idx_b = torch.nonzero(~a_mask, as_tuple=False).squeeze(-1)
        return idx_a, idx_b

    def _keys(self, x_flat: Tensor) -> Tensor:
        """Return similarity-space keys for ``x_flat`` of shape (B, M, D)."""
        if self.key_proj is None:
            return x_flat
        # Cast through the projection in input dtype.
        return self.key_proj(x_flat)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, frames: Tensor) -> Tensor:
        if frames.dim() != 4:
            raise ValueError(
                f"expected frames of shape (B, T, N, D), got {tuple(frames.shape)}"
            )
        B, T, N, D = frames.shape
        M = T * N
        target = self.target_tokens
        if target > M:
            raise ValueError(
                f"target_tokens={target} exceeds T*N={M}; cannot grow tokens"
            )

        # 1. Flatten time + spatial axes.
        x_flat = frames.reshape(B, M, D)  # (B, M, D)

        # 2. Bipartite split. Two regimes:
        #    (a) target >= ceil(M/2): standard ToMe alternating split, then
        #        keep top-(target - |A|) dissimilar B tokens unchanged.
        #    (b) target  < ceil(M/2): VTM-style sparse-anchor split where
        #        |A| == target, every B merges into A (no "keep" set). This
        #        lets us hit aggressive compression ratios (e.g. T*N=2240
        #        down to N'=140, a 16x cut) in a single vectorised pass.
        half = (M + 1) // 2  # ceil(M/2)
        if target >= half:
            idx_a, idx_b = self._bipartite_split(M, frames.device)
        else:
            idx_a, idx_b = self._bipartite_split(
                M, frames.device, a_size=target
            )
        a_size = idx_a.numel()
        b_size = idx_b.numel()
        # Re-check the post-split size in case the sparse selector lost a
        # token to rounding; treat as a hard internal error.
        if a_size + b_size != M:
            raise RuntimeError(
                f"internal: bipartite split lost tokens (|A|={a_size}, "
                f"|B|={b_size}, M={M})"
            )

        a_tokens = x_flat.index_select(dim=1, index=idx_a)  # (B, a_size, D)
        b_tokens = x_flat.index_select(dim=1, index=idx_b)  # (B, b_size, D)

        # 3. Cosine similarity B -> A in fp32 for stability.
        keys = self._keys(x_flat)
        a_keys = keys.index_select(dim=1, index=idx_a).float()
        b_keys = keys.index_select(dim=1, index=idx_b).float()
        a_n = torch.nn.functional.normalize(a_keys, dim=-1, eps=1e-6)
        b_n = torch.nn.functional.normalize(b_keys, dim=-1, eps=1e-6)
        # sim[b, i, j] = cos(b_i, a_j); shape (B, b_size, a_size)
        sim = torch.matmul(b_n, a_n.transpose(-1, -2))

        # Best match in A for each B token.
        best_sim, best_a = sim.max(dim=-1)  # (B, b_size), (B, b_size)

        # 4. Determine which B tokens to KEEP (top-r dissimilar) vs MERGE.
        #    We need (a_size + r) == target  =>  r = target - a_size.
        r_keep = target - a_size
        r_merge = b_size - r_keep
        if r_keep < 0 or r_merge < 0:
            # Should be impossible given the guards above, but defensive.
            raise RuntimeError(
                f"internal: r_keep={r_keep}, r_merge={r_merge} (target={target}, "
                f"a_size={a_size}, b_size={b_size})"
            )

        # Sort B by similarity ASCENDING: lowest sim = most dissimilar = keep.
        # We use argsort over the batched best_sim. Top-r_keep with LOWEST
        # similarity are kept unchanged; the remaining r_merge highest-sim
        # tokens get merged into their matched A token.
        sort_idx = best_sim.argsort(dim=-1, descending=False)  # (B, b_size)
        keep_idx = sort_idx[:, :r_keep]                        # (B, r_keep)
        merge_idx = sort_idx[:, r_keep:]                       # (B, r_merge)

        # 5a. Gather the "keep" B tokens unchanged.
        if r_keep > 0:
            keep_tokens = torch.gather(
                b_tokens,
                dim=1,
                index=keep_idx.unsqueeze(-1).expand(-1, -1, D),
            )  # (B, r_keep, D)
        else:
            keep_tokens = b_tokens.new_empty((B, 0, D))

        # 5b. Merge the rest into their matched A token.
        if r_merge > 0:
            # gather the merge-side B features and their A-match indices/sims
            merge_b_feats = torch.gather(
                b_tokens,
                dim=1,
                index=merge_idx.unsqueeze(-1).expand(-1, -1, D),
            )  # (B, r_merge, D)
            merge_dst = torch.gather(best_a, dim=1, index=merge_idx)   # (B, r_merge)
            merge_sim = torch.gather(best_sim, dim=1, index=merge_idx)  # (B, r_merge)

            if self.merge_type == "weighted":
                # Similarity in [-1, 1]; rescale to (0, 1] so we never drop a
                # token entirely. Using (sim + 1) / 2 keeps gradients smooth.
                weights = (merge_sim + 1.0) * 0.5
                # Numerical floor so dst contributions can't vanish.
                weights = weights.clamp(min=1e-3).to(merge_b_feats.dtype)
            else:  # "mean"
                weights = torch.ones_like(merge_sim, dtype=merge_b_feats.dtype)

            # Scatter-add weighted B contributions into a_tokens, plus the dst
            # itself with weight 1.0. Then divide by total weight per dst.
            #
            # Accumulators have shape (B, a_size, D) and (B, a_size, 1).
            weighted_feats = merge_b_feats * weights.unsqueeze(-1)
            sum_buf = a_tokens.clone()
            cnt_buf = torch.ones(
                (B, a_size, 1), dtype=merge_b_feats.dtype, device=frames.device
            )
            dst_index_feats = merge_dst.unsqueeze(-1).expand(-1, -1, D)
            dst_index_cnt = merge_dst.unsqueeze(-1)
            sum_buf = sum_buf.scatter_add(1, dst_index_feats, weighted_feats)
            cnt_buf = cnt_buf.scatter_add(1, dst_index_cnt, weights.unsqueeze(-1))
            merged_a = sum_buf / cnt_buf.clamp(min=1e-6)
        else:
            merged_a = a_tokens

        # 6. Concatenate -- final shape (B, a_size + r_keep, D) = (B, target, D).
        out = torch.cat([merged_a, keep_tokens], dim=1)
        assert out.shape == (B, target, D), (out.shape, B, target, D)
        return out

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------
    @staticmethod
    def output_token_count(T: int, N: int, target_tokens: Optional[int] = None) -> int:
        """Output token count for the VTM compressor.

        Unlike :class:`TemporalMeanPoolCompressor`, the VTM target is a
        constructor argument rather than derived from ``(T, N)``. Callers who
        only have ``(T, N)`` and the default convention can pass
        ``target_tokens=None`` and get back ``N`` (the default).
        """
        if target_tokens is None:
            return N
        return int(target_tokens)

    def extra_repr(self) -> str:  # pragma: no cover - trivial
        bits = [f"target_tokens={self.target_tokens}", f"merge_type={self.merge_type}"]
        if self.use_learnable_key:
            bits.append("use_learnable_key=True")
        return ", ".join(bits)
