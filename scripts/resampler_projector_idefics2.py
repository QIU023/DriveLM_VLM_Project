"""IDEFICS-2 Connector projector (Perceiver Resampler + Modality MLP) with PRETRAINED
init for Qwen2.5/3-VL — A.3 v2.

Wraps HF transformers' Idefics2Connector (modality_projection MLP + perceiver_resampler)
from `HuggingFaceM4/idefics2-8b`. Total ~743M pretrained params + ~6M random adapters
at the boundaries.

Architecture
------------
Qwen vision features at `vit_dim=2048` (post-merger)
  -> Linear(2048 -> 1152) input adapter [random, ~2.4M]
  -> IDEFICS-2 modality_projection (1152 -> 14336 -> 4096) [pretrained, ~75M]
  -> IDEFICS-2 perceiver_resampler (3 layers, 4096 internal, 64 latents) [pretrained, ~660M]
  -> Linear(4096 -> 2048) output adapter [random, ~8.4M]

Total pretrained ≈ 735M / 743M = 99% pretrained.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn


class Idefics2ResamplerProjector(nn.Module):
    """IDEFICS-2 Connector (pretrained) + boundary adapters for Qwen-VL backbones."""

    def __init__(
        self,
        vit_dim: int,
        lm_dim: int,
        num_queries: int = 64,
        pretrained_repo: str = "HuggingFaceM4/idefics2-8b",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        # Load the Connector weights from shard 1 only (we don't need the LM/vision).
        import os, json, safetensors.torch as st
        from transformers import AutoConfig

        cache_root = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")) + "/hub"
        snap_dir = None
        for d in os.listdir(f"{cache_root}/models--HuggingFaceM4--idefics2-8b/snapshots"):
            snap_dir = f"{cache_root}/models--HuggingFaceM4--idefics2-8b/snapshots/{d}"
            break
        if snap_dir is None:
            raise RuntimeError(f"IDEFICS-2 cache not found under {cache_root}")
        shard0_path = os.path.join(snap_dir, "model-00001-of-00007.safetensors")
        if not os.path.exists(shard0_path):
            raise RuntimeError(f"IDEFICS-2 shard 1 not found at {shard0_path}; download first")

        full_state = st.load_file(shard0_path)
        # Extract connector weights, strip "model.connector." prefix
        conn_state = {
            k.replace("model.connector.", ""): v
            for k, v in full_state.items()
            if k.startswith("model.connector.")
        }
        del full_state
        if not conn_state:
            raise RuntimeError("No connector weights found in shard 1")

        # Instantiate the connector module structure ourselves (avoid full Idefics2 load).
        cfg = AutoConfig.from_pretrained(pretrained_repo)
        from transformers.models.idefics2.modeling_idefics2 import Idefics2Connector
        self.connector = Idefics2Connector(cfg)
        # Load pretrained weights
        missing, unexpected = self.connector.load_state_dict(conn_state, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected keys: {unexpected[:5]}")
        if missing:
            # Some keys might be expected to differ (e.g. position embeddings if any)
            print(f"[Idefics2Resampler] missing keys (allowed): {missing[:5]}")
        self.connector.to(dtype=dtype)

        # IDEFICS-2 constants
        self.idefics_vision_dim = 1152   # SigLIP output
        self.idefics_lm_dim = 4096       # Llama-3-8B hidden
        self.idefics_num_queries = 64

        if num_queries != self.idefics_num_queries:
            raise ValueError(
                f"num_queries={num_queries} != IDEFICS-2 native 64; not supported"
            )

        self.vit_dim = int(vit_dim)
        self.lm_dim = int(lm_dim)
        self.num_queries = int(num_queries)

        # Boundary adapters — only random-init params
        self.input_adapter = nn.Linear(self.vit_dim, self.idefics_vision_dim, bias=True)
        self.output_adapter = nn.Linear(self.idefics_lm_dim, self.lm_dim, bias=True)
        for adp in (self.input_adapter, self.output_adapter):
            nn.init.xavier_uniform_(adp.weight)
            nn.init.zeros_(adp.bias)
            adp.to(dtype=dtype)

        # API parity with custom Resampler/Q-Former projectors
        self.target_tokens = self.num_queries
        # forward_with_video_resampler_projector reads .num_latents (legacy attr name)
        self.num_latents = self.num_queries
        # in_features alias for parity with v1 Qwen2VLPerceiverResamplerProjector
        self.in_features = self.vit_dim

    def forward(
        self,
        vision_features: Tensor,
        *,
        key_padding_mask: Optional[Tensor] = None,
        grid_thw=None,  # accepted for resampler shim parity (temporal pos), ignored — IDEFICS-2 doesn't use grid_thw
        **kwargs,
    ) -> Tensor:
        """(B, N_vision, vit_dim) -> (B, num_queries, lm_dim)."""
        if vision_features.dim() != 3:
            raise ValueError(
                f"vision_features must be 3D (B, N, D); got {tuple(vision_features.shape)}"
            )
        if vision_features.shape[-1] != self.vit_dim:
            raise ValueError(
                f"vision_features last dim {vision_features.shape[-1]} != vit_dim {self.vit_dim}"
            )
        # Project Qwen features to IDEFICS-2 vision dim
        kv = self.input_adapter(vision_features)  # (B, N, 1152)
        # IDEFICS-2 connector expects (B, N, vision_dim) and returns (B, num_queries, lm_dim_4096)
        # Use attention_mask of all-ones (no padding) — key_padding_mask not used here.
        B, N, _ = kv.shape
        attn_mask = torch.ones(B, N, dtype=torch.long, device=kv.device)
        out = self.connector(image_hidden_states=kv, attention_mask=attn_mask)
        # out: (B, num_queries=64, 4096)
        return self.output_adapter(out)  # (B, 64, lm_dim)

    @staticmethod
    def output_token_count(T: int, N: int) -> int:  # noqa: ARG004
        return 64

    def forward_4d(self, frames: Tensor) -> Tensor:
        if frames.dim() != 4:
            raise ValueError(f"forward_4d expects (B, T, N, D); got {tuple(frames.shape)}")
        B, T, N, D = frames.shape
        flat = frames.view(B, T * N, D)
        return self.forward(flat)


__all__ = ["Idefics2ResamplerProjector"]
