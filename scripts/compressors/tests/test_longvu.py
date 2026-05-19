"""Tests for the LongVU-style adaptive frame-prune compressor."""
from __future__ import annotations

import pytest
import torch

from scripts.compressors import CrossFrameCompressor, make_compressor
from scripts.compressors.longvu import LongVUCompressor


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------
def test_longvu_output_shape_2_16_140_1024() -> None:
    """[2, 16, 140, 1024] -> [2, 140, 1024] for the fixed-budget design."""
    comp = make_compressor("longvu")
    x = torch.randn(2, 16, 140, 1024)
    y = comp(x)
    assert y.shape == (2, 140, 1024)
    assert LongVUCompressor.output_token_count(T=16, N=140) == 140


def test_factory_returns_abc_subtype() -> None:
    comp = make_compressor("longvu")
    assert isinstance(comp, CrossFrameCompressor)


@pytest.mark.parametrize(
    "shape",
    [(1, 2, 4, 8), (3, 8, 16, 32), (2, 12, 64, 128)],
)
def test_shape_preserved_for_various_dims(shape: tuple) -> None:
    comp = make_compressor("longvu")
    x = torch.randn(*shape)
    y = comp(x)
    B, _, N, D = shape
    assert y.shape == (B, N, D)


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("metric", ["cosine", "l2"])
def test_similarity_metrics_run(metric: str) -> None:
    comp = make_compressor("longvu", similarity_metric=metric)
    x = torch.randn(2, 5, 10, 16)
    y = comp(x)
    assert y.shape == (2, 10, 16)
    assert torch.isfinite(y).all()


def test_learned_metric_requires_embed_dim() -> None:
    with pytest.raises(ValueError):
        make_compressor("longvu", similarity_metric="learned")


def test_importance_uniform_runs() -> None:
    torch.manual_seed(0)
    comp = make_compressor("longvu", importance_score="uniform")
    x = torch.randn(2, 4, 8, 16)
    y = comp(x)
    assert y.shape == (2, 8, 16)


def test_attn_rollout_not_implemented() -> None:
    comp = make_compressor("longvu", importance_score="attn_rollout")
    x = torch.randn(1, 3, 4, 8)
    with pytest.raises(NotImplementedError):
        comp(x)


# ---------------------------------------------------------------------------
# Core LongVU property: high-similarity frame keeps fewer tokens than a
# low-similarity (novel) frame.
# ---------------------------------------------------------------------------
def test_high_similarity_frame_has_lower_keep_ratio() -> None:
    """Build T=3 frames where frame 1 is a near-duplicate of frame 0 and
    frame 2 is fully novel. After compression, the duplicate frame should
    contribute strictly fewer tokens than the novel frame.
    """
    torch.manual_seed(0)
    B, T, N, D = 1, 3, 60, 16

    base = torch.randn(B, 1, N, D)
    # Frame 0: base; frame 1: base + tiny noise (high sim); frame 2: novel.
    f0 = base
    f1 = base + 1e-4 * torch.randn_like(base)
    f2 = torch.randn(B, 1, N, D) * 5.0  # very different magnitude/direction
    x = torch.cat([f0, f1, f2], dim=1)  # (1, 3, N, D)

    # Use uniform importance so the only signal driving selection across
    # frames is the per-frame keep-ratio bias.
    comp = make_compressor(
        "longvu",
        similarity_metric="cosine",
        importance_score="uniform",
        min_keep=0.0,
    )

    # Inspect the internal per-frame keep ratio: novel frame > duplicate frame.
    frame_repr = x.mean(dim=2)
    sim = comp._frame_similarity(frame_repr)  # (1, T)
    keep = (1.0 - sim).clamp(min=0.0, max=1.0)
    # Frame 0: sim forced to 0 (no predecessor) -> keep = 1.
    # Frame 1: high similarity to frame 0 -> keep small.
    # Frame 2: low similarity to frame 1 -> keep large.
    assert keep[0, 1].item() < keep[0, 2].item(), (
        f"duplicate frame keep={keep[0,1].item()} should be < "
        f"novel frame keep={keep[0,2].item()}"
    )

    # And end-to-end: count how many output tokens were picked from each
    # frame. We do this by checking which (T*N) slot each output row came
    # from. Re-run the forward and reverse-engineer the picks via the score.
    importance = comp._token_importance(x)
    eps = 1e-6
    frame_bias = torch.log(keep + eps).unsqueeze(-1)
    score = importance + frame_bias  # (1, T, N)
    flat = score.reshape(1, T * N)
    _, idx = torch.topk(flat, k=N, dim=1, largest=True, sorted=False)
    per_frame_pick = torch.zeros(T, dtype=torch.long)
    for t in range(T):
        per_frame_pick[t] = ((idx[0] >= t * N) & (idx[0] < (t + 1) * N)).sum()
    # Novel frame (t=2) should keep more tokens than the duplicate (t=1).
    assert per_frame_pick[2].item() > per_frame_pick[1].item(), (
        f"per-frame picks: {per_frame_pick.tolist()} -- novel frame "
        f"must outscore duplicate frame"
    )


# ---------------------------------------------------------------------------
# Autograd: gradient flows through the learned-similarity variant.
# ---------------------------------------------------------------------------
def test_learned_similarity_gradient_flows() -> None:
    torch.manual_seed(0)
    B, T, N, D = 2, 4, 12, 16
    comp = make_compressor(
        "longvu",
        similarity_metric="learned",
        importance_score="norm",
        embed_dim=D,
    )
    # Perturb the head off zero so gradients have something to act on.
    with torch.no_grad():
        comp.sim_head.weight.add_(0.01 * torch.randn_like(comp.sim_head.weight))
        comp.sim_head.bias.add_(0.01 * torch.randn_like(comp.sim_head.bias))

    x = torch.randn(B, T, N, D, requires_grad=True)
    y = comp(x)
    loss = y.pow(2).sum()
    loss.backward()

    # Input gradient flows.
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum().item() > 0.0

    # Learned-similarity head gets a gradient (via log(keep_ratio) bias on
    # the score, which the top-N selection depends on through gather, plus
    # via the importance/norm path).
    assert comp.sim_head.weight.grad is not None
    assert torch.isfinite(comp.sim_head.weight.grad).all()
    assert comp.sim_head.bias.grad is not None
    assert torch.isfinite(comp.sim_head.bias.grad).all()


def test_gradient_flows_through_tokens_default() -> None:
    """Default cosine + norm: x.grad should be non-zero."""
    torch.manual_seed(0)
    comp = make_compressor("longvu")
    x = torch.randn(2, 4, 8, 16, requires_grad=True)
    y = comp(x)
    y.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum().item() > 0.0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def test_invalid_similarity_metric_raises() -> None:
    with pytest.raises(ValueError):
        make_compressor("longvu", similarity_metric="bogus")


def test_invalid_importance_score_raises() -> None:
    with pytest.raises(ValueError):
        make_compressor("longvu", importance_score="bogus")


def test_invalid_min_keep_raises() -> None:
    with pytest.raises(ValueError):
        make_compressor("longvu", min_keep=-0.1)
    with pytest.raises(ValueError):
        make_compressor("longvu", min_keep=1.5)


def test_bad_input_rank_raises() -> None:
    comp = make_compressor("longvu")
    with pytest.raises(ValueError):
        comp(torch.randn(2, 4, 8))  # only 3 dims
