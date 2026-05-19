"""Tests for the VTM (Video Token Merging) cross-frame compressor."""
from __future__ import annotations

import pytest
import torch

from scripts.compressors import CrossFrameCompressor, make_compressor
from scripts.compressors.vtm import VTMCompressor


# ---------------------------------------------------------------------------
# Shape contract
# ---------------------------------------------------------------------------
def test_vtm_target_shape_2_16_140_1024() -> None:
    """Headline VLA shape: [2, 16, 140, 1024] -> [2, 140, 1024]."""
    B, T, N, D = 2, 16, 140, 1024
    comp = make_compressor("vtm", target_tokens=N, dim=D)
    x = torch.randn(B, T, N, D)
    y = comp(x)
    assert y.shape == (B, N, D)
    assert VTMCompressor.output_token_count(T=T, N=N) == N


def test_vtm_smaller_target() -> None:
    """Asking for fewer than N output tokens still works as long as target >= ceil(M/2)."""
    B, T, N, D = 2, 4, 8, 16
    M = T * N
    target = M // 2 + 4  # > |A| = M/2
    comp = VTMCompressor(target_tokens=target)
    y = comp(torch.randn(B, T, N, D))
    assert y.shape == (B, target, D)


def test_factory_returns_abc_subtype() -> None:
    comp = make_compressor("vtm", target_tokens=8)
    assert isinstance(comp, CrossFrameCompressor)


# ---------------------------------------------------------------------------
# Bipartite split correctness
# ---------------------------------------------------------------------------
def test_bipartite_split_alternating() -> None:
    """A set = even indices, B set = odd indices, both disjoint and cover all M."""
    idx_a, idx_b = VTMCompressor._bipartite_split(10, torch.device("cpu"))
    assert idx_a.tolist() == [0, 2, 4, 6, 8]
    assert idx_b.tolist() == [1, 3, 5, 7, 9]
    union = torch.cat([idx_a, idx_b]).sort().values
    assert torch.equal(union, torch.arange(10))


def test_bipartite_split_odd_M() -> None:
    """For odd M=11, |A| = 6 (ceil), |B| = 5 (floor)."""
    idx_a, idx_b = VTMCompressor._bipartite_split(11, torch.device("cpu"))
    assert idx_a.numel() == 6
    assert idx_b.numel() == 5
    assert torch.equal(
        torch.cat([idx_a, idx_b]).sort().values, torch.arange(11)
    )


def test_bipartite_split_sparse_anchor() -> None:
    """With explicit a_size, A is a stride-spread subset of size a_size."""
    M, a_size = 32, 4
    idx_a, idx_b = VTMCompressor._bipartite_split(
        M, torch.device("cpu"), a_size=a_size
    )
    assert idx_a.numel() == a_size
    assert idx_b.numel() == M - a_size
    # Union covers [0, M).
    assert torch.equal(
        torch.cat([idx_a, idx_b]).sort().values, torch.arange(M)
    )


# ---------------------------------------------------------------------------
# Merge behaviour
# ---------------------------------------------------------------------------
def test_vtm_reduces_to_target_token_count() -> None:
    """Output token dim equals target_tokens for several sizes."""
    B, T, N, D = 3, 4, 12, 8
    for target in (24, 30, 40, 48):
        comp = VTMCompressor(target_tokens=target)
        y = comp(torch.randn(B, T, N, D))
        assert y.shape == (B, target, D), (target, y.shape)


def test_vtm_identity_when_target_equals_M() -> None:
    """If target == T*N, no merging happens and tokens are preserved (as a set).

    The output ordering interleaves (A_tokens, kept-B-tokens-sorted-by-sim),
    so we verify the multiset equality rather than positional identity:
    every input token must appear exactly once in the output.
    """
    B, T, N, D = 2, 2, 4, 6
    M = T * N
    comp = VTMCompressor(target_tokens=M, merge_type="mean")
    x = torch.randn(B, T, N, D)
    y = comp(x)
    x_flat = x.reshape(B, M, D)
    # Sort by the first feature dim so multiset equality is testable.
    y_sorted = y[..., 0].sort(dim=1).values
    x_sorted = x_flat[..., 0].sort(dim=1).values
    torch.testing.assert_close(y_sorted, x_sorted)


def test_vtm_merges_duplicates_into_one() -> None:
    """When B-side tokens are identical to A-side tokens, mean merge preserves them.

    Construct a case where odd-indexed tokens equal even-indexed tokens; the
    cosine similarity for each B->A is 1, so merge_type='mean' should
    produce the same A-token after averaging.
    """
    B, T, N, D = 1, 2, 4, 5
    M = T * N
    base = torch.randn(B, M // 2, D)
    # interleave: even positions = base[i], odd positions = base[i] (clone)
    x_flat = torch.zeros(B, M, D)
    x_flat[:, 0::2] = base
    x_flat[:, 1::2] = base
    x = x_flat.view(B, T, N, D)
    # target = M//2 so r_keep=0, r_merge=|B|=M/2 -> all B merge into A.
    comp = VTMCompressor(target_tokens=M // 2, merge_type="mean")
    y = comp(x)
    # Each A token is averaged with one identical B token => unchanged.
    torch.testing.assert_close(y, base, atol=1e-5, rtol=1e-5)


def test_vtm_weighted_merge_runs_and_shapes() -> None:
    B, T, N, D = 2, 4, 5, 8
    target = (T * N) // 2 + 3
    comp = VTMCompressor(target_tokens=target, merge_type="weighted")
    y = comp(torch.randn(B, T, N, D))
    assert y.shape == (B, target, D)


# ---------------------------------------------------------------------------
# Autograd
# ---------------------------------------------------------------------------
def test_vtm_gradient_flows_through_input() -> None:
    """Input grad is non-trivial after a mean-merge forward+backward."""
    B, T, N, D = 2, 4, 6, 8
    target = (T * N) // 2 + 4
    comp = VTMCompressor(target_tokens=target, merge_type="mean")
    x = torch.randn(B, T, N, D, requires_grad=True)
    loss = comp(x).pow(2).sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum().item() > 0.0


def test_vtm_gradient_flows_through_weighted_merge() -> None:
    B, T, N, D = 2, 4, 6, 8
    target = (T * N) // 2 + 4
    comp = VTMCompressor(target_tokens=target, merge_type="weighted")
    x = torch.randn(B, T, N, D, requires_grad=True)
    loss = comp(x).pow(2).sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum().item() > 0.0


def test_vtm_gradient_flows_through_learnable_key() -> None:
    """Learnable projection receives gradient through similarity computation."""
    B, T, N, D = 2, 4, 6, 8
    target = (T * N) // 2 + 4
    comp = VTMCompressor(
        target_tokens=target, merge_type="weighted", dim=D, use_learnable_key=True
    )
    x = torch.randn(B, T, N, D, requires_grad=True)
    loss = comp(x).pow(2).sum()
    loss.backward()
    assert comp.key_proj.weight.grad is not None
    assert torch.isfinite(comp.key_proj.weight.grad).all()
    assert comp.key_proj.weight.grad.abs().sum().item() > 0.0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def test_invalid_merge_type_raises() -> None:
    with pytest.raises(ValueError):
        VTMCompressor(target_tokens=8, merge_type="bogus")


def test_target_tokens_too_large_raises() -> None:
    comp = VTMCompressor(target_tokens=1000)
    with pytest.raises(ValueError):
        # T*N = 4*4 = 16 < 1000
        comp(torch.randn(1, 4, 4, 8))


def test_heavy_compression_uses_sparse_anchors() -> None:
    """target < M/2 falls back to the VTM sparse-anchor split (|A|==target)."""
    B, T, N, D = 1, 4, 8, 4  # M=32
    comp = VTMCompressor(target_tokens=8)  # target < ceil(M/2)=16
    y = comp(torch.randn(B, T, N, D))
    assert y.shape == (B, 8, D)


def test_bad_input_rank_raises() -> None:
    comp = VTMCompressor(target_tokens=4)
    with pytest.raises(ValueError):
        comp(torch.randn(2, 4, 8))


def test_learnable_key_requires_dim() -> None:
    with pytest.raises(ValueError):
        VTMCompressor(target_tokens=4, use_learnable_key=True)


def test_output_token_count_static_helper() -> None:
    """Helper returns N when target is None, else the explicit target."""
    assert VTMCompressor.output_token_count(T=8, N=140) == 140
    assert VTMCompressor.output_token_count(T=8, N=140, target_tokens=64) == 64
