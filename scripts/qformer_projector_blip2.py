"""BLIP-2 Q-Former projector with PRETRAINED init for Qwen2.5/3-VL — A.1 v2.

Wraps HF transformers' `Blip2QFormerModel` (`Salesforce/blip2-opt-2.7b`) as a
drop-in replacement for our prior random-init `Qwen2VLQFormerProjector`. The
A.1/A.2/A.3 v1 line underperformed the linear baseline R1' by L2 +0.06-0.10
(0.68-0.71 vs 0.62); per `feedback_qformer_pretrained_init_only`, the root
cause was random-init starvation on 24K samples. This version loads the BLIP-2
Q-Former weights (pretrained on 129M image-text pairs) and only random-inits
the two boundary adapters.

Architecture
------------
Pre-merger Qwen vision features at `vit_dim` (1280 for Qwen2.5-VL-3B; 1024 for
Qwen3-VL-4B vision encoder) -> input adapter Linear(vit_dim->1408) -> BLIP-2
Q-Former (12 layers, 768 hidden, 32 queries, all weights from BLIP-2) ->
output Linear(768->lm_dim) (Qwen LM hidden_size).

The two adapters (input + output) are the ONLY random-init params:
- input  adapter: vit_dim * 1408 + 1408   (~1.8M for vit_dim=1280)
- output adapter: 768 * lm_dim + lm_dim   (~1.6M for lm_dim=2048)
Total random init: ~3.4M (vs 105M pretrained) = 3.1% random.

API compatibility
-----------------
Mirrors the same forward signature + `target_tokens` attribute as
`Qwen2VLQFormerProjector` so `forward_with_video_qformer_projector` in
train_lora.py works without changes.

Reference rule: [[feedback_pretrained_init_audit_before_sft]]
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn


class Blip2QFormerProjector(nn.Module):
    """BLIP-2 Q-Former (pretrained) + boundary adapters for Qwen-VL backbones."""

    def __init__(
        self,
        vit_dim: int,
        lm_dim: int,
        num_queries: int = 32,
        pretrained_repo: str = "Salesforce/blip2-opt-2.7b",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        from transformers import Blip2ForConditionalGeneration

        # Load full BLIP-2 model once on CPU to extract qformer + query_tokens
        # then drop everything else. We only keep ~105M params total from this
        # 3.7B-param load — fine since it's a one-time init at training start.
        # `local_files_only=True` skips HF Hub re-validation which can spuriously
        # claim the cached safetensors are missing (observed at eval time 2026-05-23).
        try:
            full = Blip2ForConditionalGeneration.from_pretrained(
                pretrained_repo, torch_dtype=dtype, attn_implementation="eager",
                local_files_only=True,
            )
        except (OSError, ValueError):
            # Cache miss — fall back to network fetch (training-time first load).
            full = Blip2ForConditionalGeneration.from_pretrained(
                pretrained_repo, torch_dtype=dtype, attn_implementation="eager",
            )
        # Pretrained Q-Former: 12 layers, hidden=768, num_heads=12, encoder_hidden=1408
        self.qformer = full.qformer
        # query_tokens: (1, 32, 768) learnable, from BLIP-2 pretrained
        # If caller asked for num_queries != 32, expand by repeating/truncating
        # the pretrained queries to maximize init benefit.
        pretrained_q = full.query_tokens.detach().clone()  # (1, 32, 768)
        if num_queries == pretrained_q.shape[1]:
            self.query_tokens = nn.Parameter(pretrained_q)
        elif num_queries < pretrained_q.shape[1]:
            self.query_tokens = nn.Parameter(pretrained_q[:, :num_queries, :].contiguous())
        else:
            # Pad with random init (trunc_normal std=0.02) for extras beyond 32
            extra = torch.empty(1, num_queries - pretrained_q.shape[1], 768, dtype=dtype)
            nn.init.trunc_normal_(extra, mean=0.0, std=0.02)
            self.query_tokens = nn.Parameter(torch.cat([pretrained_q, extra], dim=1))
        del full  # release the LM + vision tower; we only keep qformer + queries

        self.vit_dim = int(vit_dim)
        self.lm_dim = int(lm_dim)
        self.num_queries = int(num_queries)
        self.qformer_hidden = 768  # BLIP-2 const
        self.qformer_encoder_hidden = 1408  # BLIP-2 cross-attn KV dim

        # Boundary adapters — ONLY random-init params.
        # Input: project Qwen vision features (vit_dim) -> BLIP-2 cross-attn KV dim (1408)
        self.input_adapter = nn.Linear(self.vit_dim, self.qformer_encoder_hidden, bias=True)
        # Output: project Q-Former 768 -> Qwen LM hidden (lm_dim)
        self.output_adapter = nn.Linear(self.qformer_hidden, self.lm_dim, bias=True)
        # Adapter weight init (xavier_uniform, zero bias — standard for linear projectors)
        for adp in (self.input_adapter, self.output_adapter):
            nn.init.xavier_uniform_(adp.weight)
            nn.init.zeros_(adp.bias)
            adp.to(dtype=dtype)

        # API parity with the random-init Qwen2VLQFormerProjector:
        # `forward_with_video_qformer_projector` reads `target_tokens` to set
        # the LM-input placeholder budget.
        self.target_tokens = self.num_queries

    def forward(
        self,
        vision_features: Tensor,
        *,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compress (B, N_vision, vit_dim) -> (B, num_queries, lm_dim).

        key_padding_mask: (B, N_vision) bool, True at PAD. Forwarded as
        ``encoder_attention_mask`` to the BLIP-2 Q-Former (HF uses 1=keep,
        0=mask; we invert).
        """
        if vision_features.dim() != 3:
            raise ValueError(
                f"vision_features must be 3D (B, N, D); got {tuple(vision_features.shape)}"
            )
        if vision_features.shape[-1] != self.vit_dim:
            raise ValueError(
                f"vision_features last dim {vision_features.shape[-1]} != vit_dim {self.vit_dim}"
            )
        B = vision_features.shape[0]
        # Project Qwen features into BLIP-2 cross-attn KV dim
        kv = self.input_adapter(vision_features)  # (B, N, 1408)
        # Broadcast query tokens across batch
        q = self.query_tokens.expand(B, -1, -1).contiguous()  # (B, Nq, 768)
        # BLIP-2 expects encoder_attention_mask with 1=valid, 0=mask
        enc_mask = None
        if key_padding_mask is not None:
            enc_mask = (~key_padding_mask).to(dtype=torch.long)
        out = self.qformer(
            query_embeds=q,
            encoder_hidden_states=kv,
            encoder_attention_mask=enc_mask,
            return_dict=True,
        )
        # last_hidden_state: (B, Nq, 768)
        return self.output_adapter(out.last_hidden_state)  # (B, Nq, lm_dim)

    @staticmethod
    def output_token_count(T: int, N: int) -> int:  # noqa: ARG004
        # API parity with random-init projector
        return 32

    def forward_4d(self, frames: Tensor) -> Tensor:
        """(B, T, N, D) -> (B, num_queries, lm_dim) — flatten T*N then forward.

        Used by `forward_with_video_xframe_compression`-style callers that
        pre-collate per-frame features.
        """
        if frames.dim() != 4:
            raise ValueError(
                f"forward_4d expects (B, T, N, D); got {tuple(frames.shape)}"
            )
        B, T, N, D = frames.shape
        flat = frames.view(B, T * N, D)
        return self.forward(flat)


__all__ = ["Blip2QFormerProjector"]
