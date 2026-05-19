"""LongVU-style adaptive cross-frame token pruning compressor.

Reference
---------
LongVU: Spatiotemporal Adaptive Compression for Long Video-Language
Understanding. Meta AI, 2024. arXiv:2410.17434.
https://arxiv.org/abs/2410.17434

Key idea
--------
Not every frame carries equal information. Use feature similarity between
consecutive frames as a redundancy signal: a frame that is very similar to its
predecessor mostly repeats content already seen, so most of its tokens can be
dropped. A frame that is very different brings novel content and should keep
most of its tokens. This module implements that intuition for our Qwen2.5-VL
VLA stack with a fixed output budget so downstream LM padding stays static.

Algorithm
---------
Input:  ``frames`` of shape ``(B, T, N, D)``.
1. Build a frame-level descriptor by mean-pooling tokens within each frame
   -> ``(B, T, D)``.
2. Compute per-frame similarity to the previous frame
   (frame 0 is treated as fully novel, similarity = 0):
   - ``"cosine"`` (default): cosine between consecutive descriptors.
   - ``"l2"``: Gaussian-kernel similarity ``exp(-||a - b||^2 / D)``.
   - ``"learned"``: a small Linear over the absolute difference of consecutive
     descriptors produces a similarity logit, squashed by sigmoid.
3. Map similarity -> per-frame keep ratio:
   ``keep_ratio = clamp(1 - similarity, min_keep, 1.0)`` -> ``(B, T)``.
4. Compute a per-token importance score for every token:
   - ``"norm"`` (default): L2 norm of the token's feature vector.
   - ``"uniform"``: random scores (kill-baseline / ablation).
   - ``"attn_rollout"`` is intentionally not implemented here; it requires the
     vision tower's attention maps, which we do not have in this module's
     signature. Requesting it raises ``NotImplementedError``.
5. Combine per-token importance with per-frame keep-ratio so frames with
   more redundancy bias their tokens downward:
   ``score[b, t, n] = importance[b, t, n] + log(keep_ratio[b, t])``.
   Adding a (broadcasted) per-frame log-bias is monotone in keep_ratio and
   preserves intra-frame importance ordering.
6. Take the global top-N tokens by score across the flattened ``(T * N)``
   axis. This is the fixed-budget variant -- it guarantees the output is
   exactly ``(B, N, D)`` regardless of how the budget splits across frames.
   The per-frame quota emerges from the score, biased by keep_ratio.
7. Scale each kept token by its frame's ``keep_ratio`` (a soft gate). With
   ``min_keep`` close to 1 this is near-identity; with smaller ``min_keep``
   tokens from redundant frames contribute proportionally less. This soft
   gate also lets gradient flow through the similarity branch (in
   particular through the ``"learned"`` head), since the hard top-N
   selection on its own is non-differentiable.

Output ``(B, N, D)`` matches the 4-frame baseline shape so the LM input
sequence length is unchanged.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from . import CrossFrameCompressor, register

_VALID_SIM = ("cosine", "l2", "learned")
_VALID_IMP = ("norm", "uniform", "attn_rollout")


@register("longvu")
class LongVUCompressor(CrossFrameCompressor):
    """Adaptive frame-level token pruning with a fixed output budget.

    Parameters
    ----------
    similarity_metric : {"cosine", "l2", "learned"}
        How to score per-frame redundancy. Default ``"cosine"``.
    importance_score : {"norm", "uniform", "attn_rollout"}
        How to rank tokens within a frame. ``"attn_rollout"`` is optional and
        currently raises ``NotImplementedError`` -- it needs the vision
        tower's attention maps, which are not available in the
        ``(B, T, N, D)`` -> ``(B, N, D)`` interface.
    embed_dim : int, optional
        Required when ``similarity_metric="learned"``: width of the Linear
        used to predict the per-frame similarity logit.
    min_keep : float
        Floor on per-frame keep ratio, in ``[0, 1]``. Default ``0.1``.
        Prevents a frame from being completely starved.
    """

    def __init__(
        self,
        similarity_metric: str = "cosine",
        importance_score: str = "norm",
        embed_dim: Optional[int] = None,
        min_keep: float = 0.1,
    ) -> None:
        super().__init__()
        if similarity_metric not in _VALID_SIM:
            raise ValueError(
                f"similarity_metric must be one of {_VALID_SIM}, "
                f"got {similarity_metric!r}"
            )
        if importance_score not in _VALID_IMP:
            raise ValueError(
                f"importance_score must be one of {_VALID_IMP}, "
                f"got {importance_score!r}"
            )
        if not (0.0 <= min_keep <= 1.0):
            raise ValueError(f"min_keep must be in [0, 1], got {min_keep}")

        self.similarity_metric = similarity_metric
        self.importance_score = importance_score
        self.min_keep = float(min_keep)

        if similarity_metric == "learned":
            if embed_dim is None or embed_dim < 1:
                raise ValueError(
                    "similarity_metric='learned' requires embed_dim >= 1"
                )
            # Maps abs-difference of consecutive frame descriptors to a scalar
            # similarity logit. Init close to zero so initial similarity ~0.5.
            self.sim_head = nn.Linear(embed_dim, 1)
            nn.init.zeros_(self.sim_head.weight)
            nn.init.zeros_(self.sim_head.bias)
        self.embed_dim = embed_dim

    # ------------------------------------------------------------------
    # Frame-level similarity
    # ------------------------------------------------------------------
    def _frame_similarity(self, frame_repr: Tensor) -> Tensor:
        """``(B, T, D)`` -> ``(B, T)`` similarity to previous frame.

        Frame 0 is treated as fully novel (similarity = 0).
        """
        B, T, D = frame_repr.shape
        prev = torch.roll(frame_repr, shifts=1, dims=1)  # (B, T, D)
        # Compute pairwise similarity (current, prev), then zero out t=0.
        if self.similarity_metric == "cosine":
            sim = torch.nn.functional.cosine_similarity(frame_repr, prev, dim=-1)
            # Cosine is in [-1, 1]; map to [0, 1] for a "redundancy" reading.
            sim = (sim + 1.0) * 0.5
        elif self.similarity_metric == "l2":
            diff = frame_repr - prev
            d2 = (diff * diff).sum(dim=-1)  # (B, T)
            sim = torch.exp(-d2 / float(max(D, 1)))
        else:  # learned
            diff = (frame_repr - prev).abs()
            logit = self.sim_head(diff).squeeze(-1)  # (B, T)
            sim = torch.sigmoid(logit)

        # Frame 0 has no predecessor -> novel by definition.
        if T > 0:
            mask = torch.ones(T, device=sim.device, dtype=sim.dtype)
            mask[0] = 0.0
            sim = sim * mask.view(1, T)
        return sim

    # ------------------------------------------------------------------
    # Per-token importance
    # ------------------------------------------------------------------
    def _token_importance(self, frames: Tensor) -> Tensor:
        """``(B, T, N, D)`` -> ``(B, T, N)`` importance scores."""
        if self.importance_score == "norm":
            return frames.norm(dim=-1)
        if self.importance_score == "uniform":
            return torch.rand(
                frames.shape[:3], device=frames.device, dtype=frames.dtype
            )
        # attn_rollout: needs external attention; intentionally not supported
        # via this interface. Callers that have rollout maps should subclass
        # and override _token_importance.
        raise NotImplementedError(
            "importance_score='attn_rollout' requires vision-tower attention "
            "maps not available in (B, T, N, D)->(B, N, D); subclass and "
            "override _token_importance."
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, frames: Tensor) -> Tensor:
        if frames.dim() != 4:
            raise ValueError(
                f"expected frames of shape (B, T, N, D), got {tuple(frames.shape)}"
            )
        B, T, N, D = frames.shape
        if T == 0 or N == 0:
            raise ValueError(f"T and N must be >= 1, got T={T}, N={N}")

        # 1) Frame descriptor and per-frame similarity.
        frame_repr = frames.mean(dim=2)  # (B, T, D)
        sim = self._frame_similarity(frame_repr)  # (B, T)

        # 2) Per-frame keep ratio in [min_keep, 1.0].
        keep_ratio = (1.0 - sim).clamp(min=self.min_keep, max=1.0)  # (B, T)

        # 3) Per-token importance.
        importance = self._token_importance(frames)  # (B, T, N)

        # 4) Combine: monotone in importance within a frame, biased across
        #    frames by log(keep_ratio). Add a small eps so log is finite.
        eps = 1e-6
        frame_bias = torch.log(keep_ratio + eps).unsqueeze(-1)  # (B, T, 1)
        score = importance + frame_bias  # (B, T, N)

        # 5) Global top-N over (T * N) tokens per sample.
        flat_score = score.reshape(B, T * N)  # (B, T*N)
        flat_tokens = frames.reshape(B, T * N, D)  # (B, T*N, D)
        topk = torch.topk(flat_score, k=N, dim=1, largest=True, sorted=False)
        idx = topk.indices  # (B, N)
        # Sort selected indices so output ordering is deterministic (chrono).
        idx_sorted, _ = torch.sort(idx, dim=1)
        gather_idx = idx_sorted.unsqueeze(-1).expand(B, N, D)  # (B, N, D)
        out = torch.gather(flat_tokens, dim=1, index=gather_idx)  # (B, N, D)

        # 6) Soft per-token gating by the kept token's frame keep_ratio so
        #    gradient flows through the similarity path (learned metric and
        #    upstream features) and tokens from redundant frames contribute
        #    proportionally less -- consistent with LongVU's intent.
        keep_per_token = keep_ratio.unsqueeze(-1).expand(B, T, N).reshape(B, T * N)
        keep_gather = torch.gather(keep_per_token, dim=1, index=idx_sorted)  # (B, N)
        out = out * keep_gather.unsqueeze(-1)
        return out

    @staticmethod
    def output_token_count(T: int, N: int) -> int:
        # Fixed-budget design: output exactly N tokens regardless of T.
        return N

    def extra_repr(self) -> str:  # pragma: no cover - trivial
        bits = [
            f"similarity_metric={self.similarity_metric}",
            f"importance_score={self.importance_score}",
            f"min_keep={self.min_keep}",
        ]
        if self.similarity_metric == "learned":
            bits.append(f"embed_dim={self.embed_dim}")
        return ", ".join(bits)
