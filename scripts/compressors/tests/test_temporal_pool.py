"""Tests for the temporal mean-pool cross-frame compressor."""
from __future__ import annotations

import pytest
import torch

from scripts.compressors import CrossFrameCompressor, make_compressor
from scripts.compressors.temporal_pool import TemporalMeanPoolCompressor


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------
def test_mean_pool_output_shape() -> None:
    """[2, 8, 140, 1024] -> [2, 140, 1024] for mean pool."""
    comp = make_compressor("temporal_pool", pool_type="mean")
    x = torch.randn(2, 8, 140, 1024)
    y = comp(x)
    assert y.shape == (2, 140, 1024)
    assert TemporalMeanPoolCompressor.output_token_count(T=8, N=140) == 140


def test_factory_returns_abc_subtype() -> None:
    comp = make_compressor("temporal_pool", pool_type="mean")
    assert isinstance(comp, CrossFrameCompressor)


# ---------------------------------------------------------------------------
# Numerical correctness
# ---------------------------------------------------------------------------
def test_mean_pool_matches_manual_mean() -> None:
    """Deterministic mean output equals manual `.mean(1)`."""
    torch.manual_seed(0)
    comp = make_compressor("temporal_pool", pool_type="mean")
    x = torch.randn(3, 4, 10, 16)
    y = comp(x)
    torch.testing.assert_close(y, x.mean(dim=1))


def test_last_pool_returns_last_frame() -> None:
    comp = make_compressor("temporal_pool", pool_type="last")
    x = torch.randn(2, 5, 7, 8)
    torch.testing.assert_close(comp(x), x[:, -1])


def test_weighted_pool_last_frame_identity() -> None:
    """Weighted pool returns last frame when logits force softmax to [0,..,0,1]."""
    T = 6
    comp = make_compressor("temporal_pool", pool_type="weighted", num_frames=T)
    # Drive softmax to a near one-hot on the last frame.
    with torch.no_grad():
        big = 1e4
        comp.frame_logits.copy_(torch.tensor([-big] * (T - 1) + [0.0]))
    x = torch.randn(2, T, 12, 8)
    y = comp(x)
    torch.testing.assert_close(y, x[:, -1], atol=1e-5, rtol=1e-5)


def test_weighted_pool_uniform_init_matches_mean() -> None:
    """At init (all-zero logits), weighted pool == uniform mean."""
    T = 4
    comp = make_compressor("temporal_pool", pool_type="weighted", num_frames=T)
    x = torch.randn(2, T, 5, 8)
    torch.testing.assert_close(comp(x), x.mean(dim=1))


def test_exponential_pool_normalised_and_recency_biased() -> None:
    decay = 0.5
    comp = make_compressor(
        "temporal_pool", pool_type="exponential", decay=decay
    )
    T = 4
    x = torch.ones(1, T, 1, 1)
    y = comp(x)
    # Weights are normalised, so pooled all-ones still equals one.
    torch.testing.assert_close(y, torch.ones(1, 1, 1))

    # Now check recency bias: token-axis basis frames -> y is a convex combo
    # where the last frame dominates.
    basis = torch.eye(T).view(1, T, T, 1)  # (1, T, N=T, D=1)
    y2 = comp(basis).squeeze(0).squeeze(-1)  # (T,)
    # y2[t] equals the normalised weight on frame t.
    assert torch.argmax(y2).item() == T - 1
    assert y2[-1] > y2[0]


# ---------------------------------------------------------------------------
# Autograd
# ---------------------------------------------------------------------------
def test_weighted_pool_gradient_flows() -> None:
    T = 4
    comp = make_compressor("temporal_pool", pool_type="weighted", num_frames=T)
    x = torch.randn(2, T, 5, 8, requires_grad=True)
    y = comp(x)
    loss = y.sum()
    loss.backward()
    assert comp.frame_logits.grad is not None
    assert torch.isfinite(comp.frame_logits.grad).all()
    # The gradient should not be identically zero for a generic input.
    assert comp.frame_logits.grad.abs().sum().item() > 0.0
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def test_invalid_pool_type_raises() -> None:
    with pytest.raises(ValueError):
        make_compressor("temporal_pool", pool_type="bogus")


def test_weighted_requires_num_frames() -> None:
    with pytest.raises(ValueError):
        make_compressor("temporal_pool", pool_type="weighted")


def test_weighted_t_mismatch_raises() -> None:
    comp = make_compressor("temporal_pool", pool_type="weighted", num_frames=4)
    x = torch.randn(1, 5, 3, 8)
    with pytest.raises(ValueError):
        comp(x)


def test_bad_input_rank_raises() -> None:
    comp = make_compressor("temporal_pool", pool_type="mean")
    with pytest.raises(ValueError):
        comp(torch.randn(2, 4, 8))  # only 3 dims


def test_exponential_decay_bounds() -> None:
    with pytest.raises(ValueError):
        make_compressor("temporal_pool", pool_type="exponential", decay=0.0)
    with pytest.raises(ValueError):
        make_compressor("temporal_pool", pool_type="exponential", decay=1.0)


def test_unknown_compressor_name_raises() -> None:
    with pytest.raises(KeyError):
        make_compressor("not_a_real_compressor")
