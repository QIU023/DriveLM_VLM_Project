"""Temporal mean-pool cross-frame token compressor.

Takes per-frame visual embeddings ``(B, T, N, D)`` and collapses the time
axis ``T`` into a single set of ``N`` tokens, so LM step time stays roughly
constant as we extend from 4 past frames to 16-32.

Four pool variants are exposed via the ``pool_type`` constructor arg:

* ``"mean"``        -- uniform average over T (baseline; zero params).
* ``"last"``        -- copy the last frame (sanity reference; zero params).
* ``"weighted"``    -- learnable per-frame logits, softmax over T (~T params).
* ``"exponential"`` -- fixed exponential decay biased toward recent frames.

Output shape is always ``(B, N, D)`` -- spatial token count is preserved.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from . import CrossFrameCompressor, register

_VALID_POOLS = ("mean", "last", "weighted", "exponential")


@register("temporal_pool")
class TemporalMeanPoolCompressor(CrossFrameCompressor):
    """Collapse the time axis with a configurable pooling rule.

    Parameters
    ----------
    pool_type : {"mean", "last", "weighted", "exponential"}
        Pooling rule. See module docstring.
    num_frames : int, optional
        Required for ``pool_type="weighted"`` so we can allocate the per-frame
        weight parameter; ignored otherwise.
    decay : float, default ``0.5``
        Exponential decay base for ``pool_type="exponential"``; must satisfy
        ``0 < decay < 1``. Higher decay -> longer memory.
    """

    def __init__(
        self,
        pool_type: str = "mean",
        num_frames: Optional[int] = None,
        decay: float = 0.5,
    ) -> None:
        super().__init__()
        if pool_type not in _VALID_POOLS:
            raise ValueError(
                f"pool_type must be one of {_VALID_POOLS}, got {pool_type!r}"
            )
        self.pool_type = pool_type
        self.num_frames = num_frames
        self.decay = decay

        if pool_type == "weighted":
            if num_frames is None or num_frames < 1:
                raise ValueError(
                    "pool_type='weighted' requires num_frames >= 1"
                )
            # Logits initialised to zero -> softmax gives uniform mean at init.
            self.frame_logits = nn.Parameter(torch.zeros(num_frames))
        elif pool_type == "exponential":
            if not (0.0 < decay < 1.0):
                raise ValueError("decay must be in (0, 1) for exponential pool")
            # Weights resolved lazily in forward (depends on actual T).

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _exponential_weights(self, T: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        # weights[t] proportional to decay ** (T - 1 - t); normalise to sum 1.
        idx = torch.arange(T, device=device, dtype=dtype)
        w = self.decay ** ((T - 1) - idx)
        return w / w.sum()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, frames: Tensor) -> Tensor:
        if frames.dim() != 4:
            raise ValueError(
                f"expected frames of shape (B, T, N, D), got {tuple(frames.shape)}"
            )
        _, T, _, _ = frames.shape

        if self.pool_type == "mean":
            return frames.mean(dim=1)

        if self.pool_type == "last":
            return frames[:, -1]

        if self.pool_type == "weighted":
            if T != self.num_frames:
                raise ValueError(
                    f"weighted pool was built for T={self.num_frames} but got T={T}"
                )
            w = torch.softmax(self.frame_logits, dim=0)  # (T,)
            # (B, T, N, D) * (T, 1, 1) -> sum over T -> (B, N, D)
            return (frames * w.view(1, T, 1, 1)).sum(dim=1)

        # exponential
        w = self._exponential_weights(T, frames.device, frames.dtype)
        return (frames * w.view(1, T, 1, 1)).sum(dim=1)

    @staticmethod
    def output_token_count(T: int, N: int) -> int:
        # Spatial tokens are preserved; time axis collapsed.
        return N

    def extra_repr(self) -> str:  # pragma: no cover - trivial
        bits = [f"pool_type={self.pool_type}"]
        if self.pool_type == "weighted":
            bits.append(f"num_frames={self.num_frames}")
        if self.pool_type == "exponential":
            bits.append(f"decay={self.decay}")
        return ", ".join(bits)
