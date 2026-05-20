"""HF-Accelerate port of the Flamingo-style Perceiver Resampler projector.

Port of the torchtitan ``Qwen3VLPerceiverResamplerProjector`` (commit
``0026ea5`` on the ``perceiver_resampler_projector`` branch of
``torchtitan_qwen25``) to the HF Qwen2.5-VL-3B stack (lm_dim=2048).

Track A.3 in the projector-school comparison
--------------------------------------------
* A.0 = Linear (Qwen native PatchMerger).
* A.1 = Q-Former-64 (BLIP-2-style; no temporal pos).
* A.2 = PixelShuffle 2x + Linear.
* A.3 = Perceiver Resampler-64 (THIS FILE; Flamingo-style, WITH temporal pos).

Key architectural deltas vs. the Q-Former sibling
-------------------------------------------------
* **Latent self-attention.** Each Perceiver block does
  ``(latent_self_attn -> cross_attn(latents <- visual + temporal_pos) -> FFN)``.
* **Temporal positional encoding on inputs.** Vision inputs receive a
  learnable ``nn.Embedding(T_max, in_features)`` added BEFORE the cross-attn
  KV projection. Latents see frame-distinguishable features.

Dim decoupling (vs torchtitan: lm_dim=4096 -> 1.21B; ours ~90M)
---------------------------------------------------------------
The torchtitan production runs the resampler at ``lm_dim`` (4096 for 8B),
~1.21B params. For HF Qwen2.5-VL-3B (lm_dim=2048) we DECOUPLE: latents,
self-attn, cross-attn-Q, FFN at ``internal_dim=1024``; KV input dim =
``in_features=2048`` projected to internal_dim by k_proj/v_proj; temporal-pos
table dim = in_features (matching torchtitan); final ``Linear(1024 -> 2048)``
lifts to lm_dim. Measured: 90.44M params (80-110M target).

Multi-cam temporal-index policy (matches torchtitan)
----------------------------------------------------
Temporal index RESETS at every visual-item boundary: cam-A frame-0 and
cam-B frame-0 BOTH get ``temporal_pos[0]``. For 3-cam x 4f nuScenes,
shared T = {0,1,2,3}.

Shape contract
--------------
* Input ``vision_features``: (B, N_vision, in_features) flattened.
* Input ``grid_thw``: (num_visual_items, 3) POST-merger (t, h, w).
* Output: (B, num_latents, lm_dim).
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResamplerSelfAttention(nn.Module):
    """Multi-head self-attention over the latents at ``internal_dim``."""

    def __init__(self, internal_dim: int, n_heads: int) -> None:
        super().__init__()
        if internal_dim % n_heads != 0:
            raise ValueError(
                f"internal_dim ({internal_dim}) must be divisible by n_heads ({n_heads})"
            )
        self.internal_dim = internal_dim
        self.n_heads = n_heads
        self.head_dim = internal_dim // n_heads
        self.q_proj = nn.Linear(internal_dim, internal_dim, bias=True)
        self.k_proj = nn.Linear(internal_dim, internal_dim, bias=True)
        self.v_proj = nn.Linear(internal_dim, internal_dim, bias=True)
        self.o_proj = nn.Linear(internal_dim, internal_dim, bias=True)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        B, Nl, _ = latents.shape
        H, D = self.n_heads, self.head_dim
        q = self.q_proj(latents).view(B, Nl, H, D).transpose(1, 2)
        k = self.k_proj(latents).view(B, Nl, H, D).transpose(1, 2)
        v = self.v_proj(latents).view(B, Nl, H, D).transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, Nl, self.internal_dim)
        return self.o_proj(attn_out)


class _ResamplerCrossAttention(nn.Module):
    """Multi-head cross-attention from latents (internal_dim) to KV (in_features)."""

    def __init__(self, internal_dim: int, kv_in_features: int, n_heads: int) -> None:
        super().__init__()
        if internal_dim % n_heads != 0:
            raise ValueError(
                f"internal_dim ({internal_dim}) must be divisible by n_heads ({n_heads})"
            )
        self.internal_dim = internal_dim
        self.kv_in_features = kv_in_features
        self.n_heads = n_heads
        self.head_dim = internal_dim // n_heads
        self.q_proj = nn.Linear(internal_dim, internal_dim, bias=True)
        self.k_proj = nn.Linear(kv_in_features, internal_dim, bias=True)
        self.v_proj = nn.Linear(kv_in_features, internal_dim, bias=True)
        self.o_proj = nn.Linear(internal_dim, internal_dim, bias=True)

    def forward(
        self,
        latents: torch.Tensor,
        kv: torch.Tensor,
        *,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, Nl, _ = latents.shape
        Nkv = kv.shape[1]
        H, D = self.n_heads, self.head_dim
        q = self.q_proj(latents).view(B, Nl, H, D).transpose(1, 2)
        k = self.k_proj(kv).view(B, Nkv, H, D).transpose(1, 2)
        v = self.v_proj(kv).view(B, Nkv, H, D).transpose(1, 2)
        attn_bias = None
        if key_padding_mask is not None:
            attn_bias = torch.zeros(
                B, 1, 1, Nkv, dtype=latents.dtype, device=latents.device
            )
            attn_bias = attn_bias.masked_fill(
                key_padding_mask.view(B, 1, 1, Nkv), float("-inf")
            )
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        attn_out = attn_out.transpose(1, 2).reshape(B, Nl, self.internal_dim)
        return self.o_proj(attn_out)


class _ResamplerFFN(nn.Module):
    """Position-wise FFN with GELU at internal_dim width."""

    def __init__(self, internal_dim: int, ffn_mult: int = 2) -> None:
        super().__init__()
        hidden = ffn_mult * internal_dim
        self.fc1 = nn.Linear(internal_dim, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, internal_dim, bias=True)
        self.act = nn.GELU(approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class _ResamplerBlock(nn.Module):
    """One Perceiver Resampler block (Flamingo Sec. 3.1).

    Pre-LN throughout::

        latents = latents + self_attn(LN(latents))
        latents = latents + cross_attn(LN_q(latents), LN_kv(kv))
        latents = latents + ffn(LN(latents))
    """

    def __init__(
        self,
        internal_dim: int,
        kv_in_features: int,
        n_heads: int,
        ffn_mult: int,
        layer_norm_eps: float,
    ) -> None:
        super().__init__()
        self.norm_self = nn.LayerNorm(internal_dim, eps=layer_norm_eps)
        self.norm_cross_q = nn.LayerNorm(internal_dim, eps=layer_norm_eps)
        self.norm_cross_kv = nn.LayerNorm(kv_in_features, eps=layer_norm_eps)
        self.norm_ffn = nn.LayerNorm(internal_dim, eps=layer_norm_eps)
        self.self_attn = _ResamplerSelfAttention(internal_dim, n_heads)
        self.cross_attn = _ResamplerCrossAttention(internal_dim, kv_in_features, n_heads)
        self.ffn = _ResamplerFFN(internal_dim, ffn_mult=ffn_mult)

    def forward(
        self,
        latents: torch.Tensor,
        kv: torch.Tensor,
        *,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        latents = latents + self.self_attn(self.norm_self(latents))
        latents = latents + self.cross_attn(
            self.norm_cross_q(latents),
            self.norm_cross_kv(kv),
            key_padding_mask=key_padding_mask,
        )
        latents = latents + self.ffn(self.norm_ffn(latents))
        return latents


def _compute_temporal_indices(
    grid_thw: torch.Tensor,
    n_vision: int,
    device: torch.device,
    t_max: int,
) -> torch.Tensor:
    """Per-token temporal frame indices, resetting per visual item.

    For each item ``(t, h, w)``, emit ``t`` blocks of ``h*w`` identical frame
    indices (0..t-1). Concatenate across items; clamp to ``[0, t_max-1]``.
    """
    parts: List[torch.Tensor] = []
    total = 0
    for i in range(grid_thw.shape[0]):
        t = int(grid_thw[i, 0].item())
        h = int(grid_thw[i, 1].item())
        w = int(grid_thw[i, 2].item())
        spatial = h * w
        idx = torch.arange(t, device=device, dtype=torch.long).repeat_interleave(spatial)
        parts.append(idx)
        total += t * spatial
    if total != n_vision:
        raise ValueError(
            f"grid_thw rows imply {total} tokens but vision_features carries "
            f"{n_vision} along the N axis."
        )
    temporal_idx = torch.cat(parts, dim=0)
    return temporal_idx.clamp_(max=t_max - 1)


class Qwen2VLPerceiverResamplerProjector(nn.Module):
    """Flamingo-style Perceiver Resampler projector for Qwen2.5-VL (HF stack).

    Replaces the in-encoder ``model.visual.merger``. Duck-types the same call
    contract as the sibling Q-Former / PixelShuffle ports:

        out = projector(vision_features, grid_thw=post_merger_grid_thw)

    Returns a fixed ``num_latents`` tokens at ``lm_dim`` regardless of input
    length, ready to scatter into the LM input at ``<|video_pad|>`` placeholders.

    Param count at the defaults (in_features=2048, lm_dim=2048, internal_dim=1024,
    num_latents=64, num_layers=6, n_heads=8, ffn_mult=2): ~90.44M
    (FFN-light because internal_dim = lm_dim / 2).
    """

    def __init__(
        self,
        in_features: int,
        lm_dim: int,
        *,
        internal_dim: int = 1024,
        num_latents: int = 64,
        num_layers: int = 6,
        n_heads: int = 8,
        ffn_mult: int = 2,
        layer_norm_eps: float = 1e-6,
        t_max: int = 32,
    ) -> None:
        super().__init__()
        if internal_dim % n_heads != 0:
            raise ValueError(
                f"internal_dim ({internal_dim}) must be divisible by n_heads ({n_heads})"
            )
        self.in_features = in_features
        self.lm_dim = lm_dim
        self.internal_dim = internal_dim
        self.num_latents = num_latents
        self.num_layers = num_layers
        self.n_heads = n_heads
        self.ffn_mult = ffn_mult
        self.t_max = t_max

        # Learnable latent tokens at internal_dim (NOT lm_dim).
        self.latents = nn.Parameter(torch.empty(num_latents, internal_dim))
        nn.init.normal_(self.latents, mean=0.0, std=0.02)
        # Temporal positional embedding on KV inputs, table dim = in_features.
        self.temporal_pos = nn.Embedding(t_max, in_features)
        nn.init.normal_(self.temporal_pos.weight, mean=0.0, std=0.02)
        # Stack of blocks.
        self.layers = nn.ModuleList(
            [
                _ResamplerBlock(
                    internal_dim=internal_dim,
                    kv_in_features=in_features,
                    n_heads=n_heads,
                    ffn_mult=ffn_mult,
                    layer_norm_eps=layer_norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_out = nn.LayerNorm(internal_dim, eps=layer_norm_eps)
        # Output projection: internal_dim -> lm_dim.
        self.out_proj = nn.Linear(internal_dim, lm_dim, bias=True)

    @property
    def target_tokens(self) -> int:
        return self.num_latents

    @staticmethod
    def output_token_count(T: int, N: int, num_latents: int = 64) -> int:  # noqa: ARG004
        return int(num_latents)

    def _add_temporal_pos(
        self,
        vision_features: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        B, N, _ = vision_features.shape
        temporal_idx = _compute_temporal_indices(
            grid_thw, n_vision=N, device=vision_features.device, t_max=self.t_max,
        )
        temp_emb = self.temporal_pos(temporal_idx).to(vision_features.dtype)
        return vision_features + temp_emb.unsqueeze(0)

    def forward(
        self,
        vision_features: torch.Tensor,
        *,
        grid_thw: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """(B, N_vision, in_features) -> (B, num_latents, lm_dim)."""
        if vision_features.dim() != 3:
            raise ValueError(
                f"expected (B, N, in_features), got {tuple(vision_features.shape)}"
            )
        if vision_features.shape[-1] != self.in_features:
            raise ValueError(
                f"expected in_features={self.in_features}, got {vision_features.shape[-1]}"
            )
        B = vision_features.shape[0]
        latents = self.latents.unsqueeze(0).expand(B, -1, -1).contiguous()
        latents = latents.to(vision_features.dtype)
        kv = self._add_temporal_pos(vision_features, grid_thw)
        for layer in self.layers:
            latents = layer(latents, kv, key_padding_mask=key_padding_mask)
        latents = self.norm_out(latents)
        return self.out_proj(latents)

    def forward_xframe(
        self,
        frames: torch.Tensor,
        *,
        grid_thw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Adapter for the xframe-compressor contract: (B, T, N, D) -> (B, num_latents, D)."""
        if frames.dim() != 4:
            raise ValueError(
                f"forward_xframe expects (B, T, N, D), got {tuple(frames.shape)}"
            )
        B, T, N, D = frames.shape
        if grid_thw is None:
            grid_thw = torch.tensor(
                [[T, 1, N]], dtype=torch.long, device=frames.device,
            )
        flat = frames.reshape(B, T * N, D)
        return self.forward(flat, grid_thw=grid_thw)


__all__ = ["Qwen2VLPerceiverResamplerProjector"]
