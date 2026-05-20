"""BLIP-2-style Q-Former projector for HF Qwen2.5-VL — HF Accelerate port.

This is a near-1:1 port of
``torchtitan_qwen25/torchtitan/models/qwen3_vl/qformer_projector.py`` (commit
``5ee6380`` after the internal_dim decoupling fix, 117.6M params at
``internal_dim=1024``) re-implemented against vanilla ``torch.nn`` / no
torchtitan ``Module`` dependency, so it can be loaded from the HF Accelerate
``scripts/train_lora.py`` stack.

Why this exists
---------------
For 3-cam x 4-frame nuScenes planning the stock Qwen2.5-VL pipeline emits
~1680 visual tokens per sample (3 cams * 4 frames * ~140 post-merger tokens
per frame at min/max_pixels=109760). A 64-query Q-Former compresses the
visual token budget by ~26x. This is the same **deployment-efficiency**
trade documented in the torchtitan version — KV-cache scales with context
length, so fewer visual tokens => lower TTFT and smaller KV-cache on TRT
edge deploy. From-scratch Q-Former on ~24K nuScenes samples will lose to
the pretrained Qwen2.5-VL linear projection at equal token budget; that
is expected and acceptable.

Dim contract (Qwen2.5-VL-3B specifically)
-----------------------------------------
The HF model's ``model.visual`` exposes:

    visual.patch_embed    : (raw_pixel_patches, 1280) — ViT input dim
    visual.blocks[...]    : 32 ViT layers, all at hidden_size=1280
    visual.merger.ln_q    : Qwen2_5_VLRMSNorm(1280)
    visual.merger.mlp     : Linear(5120 -> 2048) -> GELU -> Linear(2048 -> 2048)

(The merger reshapes the (N, 1280) ViT output to (N/4, 5120) via the 2x2
spatial-merge, then runs an MLP into the LM dim 2048.)

Our Q-Former is a SWITCHABLE alternative to the merger MLP. It takes
post-``ln_q`` features at ``vit_dim=1280`` and produces
``num_queries`` (default 64) tokens at ``lm_dim=2048`` ready to be
scattered into the LM input at the ``<|video_pad|>`` placeholder positions.

Architecture (mirrored from torchtitan, simplified)
---------------------------------------------------
* ``num_queries`` learnable parameters of shape ``(num_queries, internal_dim)``
  where ``internal_dim`` (default 1024) is the Q-Former's internal width.
  ``internal_dim`` is decoupled from ``lm_dim`` so the param count stays
  ~80-130M instead of growing geometrically with ``lm_dim`` (e.g. for the
  7B model at ``lm_dim=3584`` the same 6-layer Q-Former would otherwise be
  ~600M).
* ``num_layers`` cross-attention blocks. Each block is pre-LN::

      q_norm   = LayerNorm(queries)           # (B, Nq, internal_dim)
      kv_norm  = LayerNorm(kv)                # (B, Nkv, in_features)
      attn_out = MultiheadAttention(q=q_norm, k=kv_norm, v=kv_norm)
      queries  = queries + attn_out
      ffn_out  = FFN(LayerNorm(queries))
      queries  = queries + ffn_out

  No self-attention between queries (BLIP-2 has it; we don't need it at
  this token budget for driving scenes). FFN follows the standard ViT
  pattern: Linear(internal_dim, ffn_mult * internal_dim) -> GELU ->
  Linear(ffn_mult * internal_dim, internal_dim).
* Final ``LayerNorm`` on the queries followed by a Linear projection
  ``out_proj: internal_dim -> lm_dim``.

Implementation notes
--------------------
* Uses ``nn.MultiheadAttention(batch_first=True)`` instead of a hand-rolled
  scaled_dot_product_attention. This is the HF-version simplification — the
  torchtitan version split q/k/v/o into separate Linear configs to
  integrate with its DTensor / Linear.Config wiring; in vanilla torch
  ``nn.MultiheadAttention`` packs all four projections into one module and
  is enough for our purposes.
* The ``in_features != internal_dim`` case is handled by passing
  ``kdim=in_features`` and ``vdim=in_features`` to ``MultiheadAttention``.
  This is the standard PyTorch idiom for cross-attention with mismatched
  q/k dims. Param count is identical: q_proj at (internal_dim, internal_dim),
  k_proj at (in_features, internal_dim), v_proj at (in_features, internal_dim),
  o_proj at (internal_dim, internal_dim).
* The ``key_padding_mask`` argument (``(B, Nkv)`` bool, True at PAD
  positions) is forwarded directly to MultiheadAttention's
  ``key_padding_mask`` — same semantics.

Shape contract
--------------
* Input  ``vision_features``: ``(B, N_vision, in_features)`` where
  ``N_vision`` may be padded per batch.
* Output ``compressed``: ``(B, num_queries, lm_dim)``.

Placeholder-count alignment (consumer-side responsibility)
----------------------------------------------------------
The HF ``Qwen2_5_VLModel.forward`` builds the post-vision LM input by
scattering vision features into ``inputs_embeds`` at every position where
``input_ids == video_token_id``. The placeholder count is set by the chat
template / processor BEFORE this Q-Former runs, and there is no
back-channel. So callers MUST make ``input_ids`` carry exactly
``B * num_queries`` ``<|video_pad|>`` placeholders per video block, OR
trim placeholders in the forward shim. See
``scripts/train_lora.forward_with_qformer_projection`` for one such shim
(it mirrors ``forward_with_video_xframe_compression``: monkey-patches
``inner.get_video_features`` and trims input_ids before
``model.forward``).
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class _QFormerBlock(nn.Module):
    """One Q-Former cross-attn + FFN block (pre-LN, no self-attn).

    All internal tensors are at ``internal_dim`` width except the KV input,
    which is at ``in_features``. The lift to ``lm_dim`` is done once at the
    top-level projector boundary (``out_proj``), not per-block.
    """

    def __init__(
        self,
        internal_dim: int,
        in_features: int,
        n_heads: int,
        ffn_mult: int = 4,
        layer_norm_eps: float = 1e-6,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if internal_dim % n_heads != 0:
            raise ValueError(
                f"internal_dim ({internal_dim}) must be divisible by "
                f"n_heads ({n_heads})"
            )
        self.norm_q = nn.LayerNorm(internal_dim, eps=layer_norm_eps)
        self.norm_kv = nn.LayerNorm(in_features, eps=layer_norm_eps)
        self.norm_ffn = nn.LayerNorm(internal_dim, eps=layer_norm_eps)

        # nn.MultiheadAttention with mismatched q/k/v dims:
        # - embed_dim = internal_dim (query dim, also the output dim)
        # - kdim, vdim = in_features (key/value input dim)
        # - bias=True everywhere (BLIP-2 default; matches our torchtitan port
        #   which has bias on every Linear sub-projection).
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=internal_dim,
            kdim=in_features,
            vdim=in_features,
            num_heads=n_heads,
            dropout=dropout,
            bias=True,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(internal_dim, ffn_mult * internal_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_mult * internal_dim, internal_dim),
        )

    def forward(
        self,
        queries: Tensor,
        kv: Tensor,
        *,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Pre-LN cross-attn + FFN with residuals.

        Args:
            queries: (B, Nq, internal_dim)
            kv: (B, Nkv, in_features)
            key_padding_mask: (B, Nkv) bool, True at PAD positions.

        Returns:
            (B, Nq, internal_dim)
        """
        q_norm = self.norm_q(queries)
        kv_norm = self.norm_kv(kv)
        attn_out, _ = self.cross_attn(
            query=q_norm,
            key=kv_norm,
            value=kv_norm,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        queries = queries + attn_out
        queries = queries + self.ffn(self.norm_ffn(queries))
        return queries


# ---------------------------------------------------------------------------
# Top-level projector
# ---------------------------------------------------------------------------


class Qwen2VLQFormerProjector(nn.Module):
    """BLIP-2-style Q-Former projector for HF Qwen2.5-VL.

    Mirror of ``torchtitan.models.qwen3_vl.Qwen3VLQFormerProjector`` re-built
    on vanilla torch.nn so it can be wired into the HF Accelerate training
    stack without a torchtitan dependency.

    Param-count design
    ------------------
    At the default config (Qwen2.5-VL-3B: ``in_features=1280``, ``lm_dim=2048``,
    ``internal_dim=1024``, ``num_layers=6``, ``n_heads=8``, ``ffn_mult=4``)::

        per_block (nn.MultiheadAttention + FFN + 3 LNs):
            cross_attn:
                in_proj_weight = q_proj(1024,1024) ~ 1.05M
                                 k_proj(1280,1024) ~ 1.31M
                                 v_proj(1280,1024) ~ 1.31M
                in_proj_bias                       ~ 3.07K
                out_proj   (1024 -> 1024)          ~ 1.05M
            ffn:
                fc1 (1024 -> 4096)                 ~ 4.20M
                fc2 (4096 -> 1024)                 ~ 4.20M
            LNs (3x)                                ~ 7K
            subtotal                                ~ 13.13M
        x 6 layers ≈ 78.8M
        + queries  : 64 * 1024 = 65K
        + out_proj : 1024 * 2048 + 2048 ≈ 2.10M
        + norm_out LN: 2K
        Total ≈ 81M params for the 3B model.

    For Qwen2.5-VL-7B (lm_dim=3584) the total is ~83M (only the final
    out_proj scales with lm_dim, so the projector stays ~BLIP-2-sized
    regardless of the LM).

    Parameters
    ----------
    vit_dim : int
        Cross-attn KV input dim. Set to the ViT hidden / context dim when
        the caller is feeding pre-merger features (1280 for Qwen2.5-VL-3B/7B
        and 8B; vision_config.hidden_size). Set to the LM hidden dim when
        the caller routes through the in-encoder merger first.
    internal_dim : int, default 1024
        Q-Former internal width. BLIP-2-base uses 768; we default to 1024
        to give a bit of headroom on driving-scene complexity. KEEP THIS
        DECOUPLED from ``lm_dim`` — the whole point is to bound the param
        count to ~80-130M instead of letting lm_dim drag every projection
        up to 2048/3584/4096.
    lm_dim : int
        Final output dim of the projector (LM hidden_size). For
        Qwen2.5-VL-3B this is 2048; for the 7B it is 3584.
    num_queries : int, default 64
        Compressed output token count. BLIP-2 standard is 32; we use 64
        for the 4-second multi-camera driving setting which packs more
        scene context per query.
    num_layers : int, default 6
        Number of cross-attn + FFN blocks. BLIP-2-base uses 12; we use 6
        for the from-scratch nuScenes training to keep the trainable param
        budget bounded.
    n_heads : int, default 8
        Number of attention heads. ``internal_dim`` must be divisible.
    ffn_mult : int, default 4
        FFN expansion factor.
    layer_norm_eps : float, default 1e-6
        LayerNorm epsilon.
    dropout : float, default 0.0
        Attention dropout. Kept at 0 by default since we're training from
        scratch on a small dataset where dropout often hurts.
    """

    def __init__(
        self,
        vit_dim: int,
        internal_dim: int = 1024,
        lm_dim: int = 2048,
        num_queries: int = 64,
        num_layers: int = 6,
        n_heads: int = 8,
        ffn_mult: int = 4,
        layer_norm_eps: float = 1e-6,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if internal_dim % n_heads != 0:
            raise ValueError(
                f"internal_dim ({internal_dim}) must be divisible by "
                f"n_heads ({n_heads})"
            )
        self.vit_dim = int(vit_dim)
        self.internal_dim = int(internal_dim)
        self.lm_dim = int(lm_dim)
        self.num_queries = int(num_queries)
        self.num_layers = int(num_layers)
        self.n_heads = int(n_heads)
        self.ffn_mult = int(ffn_mult)
        # Backwards-compat with the xframe_compressor protocol used by
        # `forward_with_video_xframe_compression`: it inspects
        # ``compressor.target_tokens`` to compute the placeholder budget.
        self.target_tokens = self.num_queries

        # Learnable queries. Initialised with trunc-normal std=0.02 (BLIP-2 +
        # PyTorch ViT standard); kept at internal_dim, not lm_dim.
        self.queries = nn.Parameter(torch.empty(num_queries, internal_dim))

        self.layers = nn.ModuleList(
            [
                _QFormerBlock(
                    internal_dim=internal_dim,
                    in_features=vit_dim,
                    n_heads=n_heads,
                    ffn_mult=ffn_mult,
                    layer_norm_eps=layer_norm_eps,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_out = nn.LayerNorm(internal_dim, eps=layer_norm_eps)
        # Final boundary projection: internal_dim -> lm_dim. This is the
        # ONLY place lm_dim shows up in the whole projector; everything
        # else (queries, attention, FFN) is sized by internal_dim.
        self.out_proj = nn.Linear(internal_dim, lm_dim, bias=True)

        self._init_weights()

    def _init_weights(self) -> None:
        # Queries: trunc-normal std=0.02 (BLIP-2 / ViT convention).
        nn.init.trunc_normal_(self.queries, mean=0.0, std=0.02)

        # All Linear weights: xavier_uniform (PyTorch MHA's default for its
        # in_proj is xavier_uniform; we follow suit for the FFN linears and
        # the final out_proj for consistency).
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        vision_features: Tensor,
        *,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compress a variable-length vision feature sequence to ``num_queries``.

        Args:
            vision_features: ``(B, N_vision, vit_dim)``. ``vit_dim`` must
                equal the projector's ``vit_dim`` configured at __init__.
            key_padding_mask: ``(B, N_vision)`` bool, ``True`` at PAD
                positions. Optional; pass when the input is padded.

        Returns:
            ``(B, num_queries, lm_dim)`` — compressed visual tokens, ready
            to be scattered into the LLM input at ``<|video_pad|>``
            placeholder positions.
        """
        if vision_features.dim() != 3:
            raise ValueError(
                f"vision_features must be 3D (B, N, D); got shape "
                f"{tuple(vision_features.shape)}"
            )
        if vision_features.shape[-1] != self.vit_dim:
            raise ValueError(
                f"vision_features last dim {vision_features.shape[-1]} != "
                f"projector vit_dim {self.vit_dim}"
            )
        B = vision_features.shape[0]
        # Broadcast the learnable queries across batch.
        queries = self.queries.unsqueeze(0).expand(B, -1, -1).contiguous()

        for layer in self.layers:
            queries = layer(
                queries,
                vision_features,
                key_padding_mask=key_padding_mask,
            )
        # LayerNorm at internal_dim, then lift to lm_dim.
        return self.out_proj(self.norm_out(queries))

    # ------------------------------------------------------------------
    # xframe_compressor compatibility (legacy 4D input)
    # ------------------------------------------------------------------

    @staticmethod
    def output_token_count(T: int, N: int) -> int:  # noqa: ARG004
        """Number of LM-input tokens this projector emits per sample.

        Q-Former always returns ``num_queries`` (default 64) regardless of
        input length. Set on the instance via ``target_tokens`` so the
        ``forward_with_video_xframe_compression`` shim picks it up without
        calling this static method.
        """
        # Static fallback; callers prefer the instance ``target_tokens`` set
        # in __init__. We default to 64 if this is invoked.
        return 64

    def forward_4d(self, frames: Tensor) -> Tensor:
        """Adapter for the ``CrossFrameCompressor``-style ``(B, T, N, D)`` API.

        ``forward_with_video_xframe_compression`` (scripts/train_lora.py)
        builds a 4D ``frames`` tensor with shape ``(B, T_post, N, D)`` from
        the vision tower's pooler_output and passes it to
        ``compressor(frames)``. This method flattens ``T*N`` into a single
        sequence axis and dispatches to the 3D forward path.

        Note: ``D`` here is the LM-dim (2048) because the caller is
        feeding POST-merger features. This requires the projector to be
        constructed with ``vit_dim=lm_dim`` for this entry-point — i.e.
        the Q-Former runs DOWNSTREAM of the linear merger when invoked this
        way. See module docstring for the alternative pre-merger path.
        """
        if frames.dim() != 4:
            raise ValueError(
                f"forward_4d expects (B, T, N, D); got shape "
                f"{tuple(frames.shape)}"
            )
        B, T, N, D = frames.shape
        flat = frames.view(B, T * N, D)
        return self.forward(flat)


__all__ = ["Qwen2VLQFormerProjector"]
