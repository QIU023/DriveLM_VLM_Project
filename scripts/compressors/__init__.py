"""Cross-frame visual token compressors.

These modules consume per-frame visual embeddings of shape ``(B, T, N, D)``
(``T`` past frames, ``N`` tokens per frame, dim ``D``) and produce a smaller
``(B, N', D)`` set of LM-input tokens so that LM sequence length stays roughly
constant as the number of past frames grows.

Public API
----------
* :class:`CrossFrameCompressor` -- abstract base class.
* :func:`make_compressor` -- factory keyed by registry name.

Registered compressors:
* ``"temporal_pool"`` -- :class:`TemporalMeanPoolCompressor` (4 pool variants).

The training script obtains a compressor via::

    from scripts.compressors import make_compressor
    comp = make_compressor("temporal_pool", pool_type="mean")
    out  = comp(frames)  # (B, T, N, D) -> (B, N, D)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict

import torch
from torch import Tensor, nn


class CrossFrameCompressor(nn.Module, ABC):
    """Abstract base class for cross-frame visual token compressors.

    Concrete subclasses must implement :meth:`forward` accepting a tensor of
    shape ``(B, T, N, D)`` and returning ``(B, N', D)`` where ``N'`` is
    computed by :meth:`output_token_count`.
    """

    @abstractmethod
    def forward(self, frames: Tensor) -> Tensor:  # pragma: no cover - abstract
        """Compress per-frame embeddings.

        Parameters
        ----------
        frames : Tensor
            Shape ``(B, T, N, D)``.

        Returns
        -------
        Tensor
            Shape ``(B, N', D)``.
        """

    @staticmethod
    @abstractmethod
    def output_token_count(T: int, N: int) -> int:  # pragma: no cover - abstract
        """Number of LM-input tokens this compressor emits per sample.

        Lets callers (LM padding, attention-mask construction) compute the
        post-compression sequence length without running a forward pass.
        """


# ---------------------------------------------------------------------------
# Registry / factory
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, Callable[..., CrossFrameCompressor]] = {}


def register(name: str) -> Callable[[Callable[..., CrossFrameCompressor]], Callable[..., CrossFrameCompressor]]:
    """Decorator: register a compressor constructor under ``name``."""

    def _wrap(ctor: Callable[..., CrossFrameCompressor]) -> Callable[..., CrossFrameCompressor]:
        if name in _REGISTRY:
            raise ValueError(f"compressor name already registered: {name!r}")
        _REGISTRY[name] = ctor
        return ctor

    return _wrap


def make_compressor(name: str, **kwargs: Any) -> CrossFrameCompressor:
    """Build a compressor by registry name.

    Raises
    ------
    KeyError
        If ``name`` is not a registered compressor.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown compressor {name!r}; registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name](**kwargs)


# Import side-effect: register built-in compressors.
from . import temporal_pool  # noqa: E402,F401  (registers TemporalMeanPoolCompressor)
from . import vtm  # noqa: E402,F401  (registers VTMCompressor)
from . import longvu  # noqa: E402,F401  (registers LongVUCompressor)

__all__ = [
    "CrossFrameCompressor",
    "make_compressor",
    "register",
]
