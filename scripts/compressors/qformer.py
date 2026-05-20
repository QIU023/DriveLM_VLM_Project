"""Q-Former cross-frame projector (HF Accelerate port).

Wraps :class:`scripts.qformer_projector_hf.Qwen2VLQFormerProjector` and
registers it under the ``"qformer"`` name in the cross-frame compressor
registry, so it can be activated from a YAML config block like::

    cross_frame_compressor:
      name: qformer
      kwargs:
        vit_dim: 2048        # post-merger LM hidden dim for Qwen2.5-VL-3B
        lm_dim:  2048
        internal_dim: 1024
        num_queries: 64
        num_layers: 6
        n_heads: 8

The default kwargs target Qwen2.5-VL-3B at the **post-merger** entry point
(`vit_dim == lm_dim == 2048`). The existing
``forward_with_video_xframe_compression`` (scripts/train_lora.py) runs the
vision tower including the linear merger, reshapes the pooler_output to
``(B, T, N, lm_dim)``, and passes that into ``compressor(frames)``. Our
``forward_4d`` flattens ``(T, N) -> T*N`` and runs the 64-query
cross-attention to produce ``(B, 64, lm_dim)``, which is then scattered
back into the LM input at the ``<|video_pad|>`` placeholder positions.

This wires the Q-Former into the SAME forward shim already used by VTM
and LongVU, so the placeholder-count alignment (drop excess
``<|video_pad|>`` tokens via the ``target_tokens=64`` attribute) is
handled for free.

Pre-merger variant (open issue)
-------------------------------
A more faithful port of the torchtitan Q-Former would skip the linear
merger entirely and feed pre-merger ViT features at ``vit_dim=1280``
(after only ``ln_q``). That requires a deeper monkey-patch of
``Qwen2_5_VLVisionModel.forward`` (not just ``get_video_features``) and
is left as a follow-up. See ``docs/upstream_prs/008_hf_qformer_projector.md``
for the open-issues list.
"""
from __future__ import annotations

import os
import sys
from typing import Optional

import torch
from torch import Tensor, nn

# scripts.qformer_projector_hf lives one level up (in scripts/), so add the
# parent to sys.path if not present. This mirrors how `compressors.longvu`
# imports `compressors.temporal_pool` — keeps the registry self-contained.
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from qformer_projector_hf import Qwen2VLQFormerProjector  # type: ignore  # noqa: E402

from . import CrossFrameCompressor, register


@register("qformer")
class QFormerCompressor(CrossFrameCompressor):
    """Q-Former projector wrapped as a CrossFrameCompressor.

    Constructor kwargs are forwarded verbatim to
    :class:`scripts.qformer_projector_hf.Qwen2VLQFormerProjector`. See its
    docstring for the dim / depth / head argument semantics.

    Token-budget contract (used by ``forward_with_video_xframe_compression``
    to trim ``<|video_pad|>`` placeholders): ``target_tokens = num_queries``
    (default 64).
    """

    def __init__(
        self,
        vit_dim: int = 2048,
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
        self.projector = Qwen2VLQFormerProjector(
            vit_dim=vit_dim,
            internal_dim=internal_dim,
            lm_dim=lm_dim,
            num_queries=num_queries,
            num_layers=num_layers,
            n_heads=n_heads,
            ffn_mult=ffn_mult,
            layer_norm_eps=layer_norm_eps,
            dropout=dropout,
        )
        # Surface target_tokens at top level so the xframe forward shim
        # (`forward_with_video_xframe_compression`) sees it without
        # introspecting `.projector`.
        self.target_tokens: int = int(num_queries)

    def forward(self, frames: Tensor) -> Tensor:
        """Adapter: ``(B, T, N, D) -> (B, num_queries, D)``.

        The caller is `forward_with_video_xframe_compression`, which feeds
        the post-merger pooler_output reshaped to (B, T, N, D) where D is
        the LM hidden dim (== `lm_dim`). For Qwen2.5-VL-3B that's 2048.
        """
        return self.projector.forward_4d(frames)

    @staticmethod
    def output_token_count(T: int, N: int, target_tokens: Optional[int] = None) -> int:  # noqa: ARG004
        """Number of LM-input tokens this compressor emits per sample.

        The Q-Former always returns ``num_queries`` (default 64). Callers
        should prefer the instance attribute ``target_tokens``; this static
        fallback returns 64 for callers that don't have an instance.
        """
        return int(target_tokens) if target_tokens is not None else 64


__all__ = ["QFormerCompressor"]
