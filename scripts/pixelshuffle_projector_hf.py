"""PixelShuffle + Linear projector for HF Qwen2.5-VL.

A CPU-friendly HF Accelerate port of the torchtitan
``Qwen3VLPixelShufflePlusLinearProjector``
(see ``torchtitan_qwen25/torchtitan/models/qwen3_vl/pixelshuffle_projector.py``).

Why this exists
---------------
For our 3-cam x 4-frame nuScenes planning recipe the stock Qwen2.5-VL pipeline
emits ~140 tokens per frame after the in-encoder ``PatchMerger`` (2x2 spatial
merge), i.e. ~140 * 4 * 3 = ~1680 visual tokens / sample at the LM boundary.

PixelShuffle 2x deterministically merges every 2x2 spatial block on top of the
merger's output, giving a further 4x compression -> ~420 visual tokens per
sample (~35 per frame x 4 x 3). No learnable query pool, no cross-attention;
just one Linear ``(4*in_dim) -> lm_dim``. This sits between the Q-Former-64
(64 tokens / item, ~26x compression) variant and the stock linear baseline
(~1680 tokens) on our fusion-mechanism school comparison (A.0 / A.1 / A.2 /
A.3).

Architecture (HF flow)
----------------------
Qwen2.5-VL's vision tower emits **post-merger** features of shape
``(total_post_tokens, lm_dim=2048)`` where ``total_post_tokens`` is the sum of
``T_pre * (H_pre // merge) * (W_pre // merge)`` across all video clips in the
batch. We:

  1. Reshape per-video (post-merger) features back to ``(B, t, h, w, lm_dim)``
     using ``grid_thw // merge_size``.
  2. Flatten the temporal axis into batch -> ``(B*t, h, w, lm_dim)``.
  3. Apply space-to-depth (PixelUnshuffle 2x) -> ``(B*t, h/2, w/2, 4*lm_dim)``.
  4. ``Linear(4*lm_dim, lm_dim)`` -> ``(B*t, h/2, w/2, lm_dim)``.
  5. Flatten back to ``(B, t*(h/2)*(w/2), lm_dim)``.

Net effect: a *stacked* projector = merger 2x2 + pixelshuffle 2x2 = 16x raw
ViT patch compression at the LM boundary. The torchtitan version uses the same
stacked design (see its docstring "path-(a) ready").

Param count (target ~17M)
-------------------------
At Qwen2.5-VL-3B's ``lm_dim=2048`` and ``shuffle_ratio=2``:

    Linear(4 * 2048, 2048) = 8192 * 2048 + 2048 ~ 16.78M

This matches the agent spec's ~17M target.

Even (H, W) requirement
-----------------------
PixelShuffle 2x requires both H and W to be even AFTER the in-encoder merger.
Qwen2.5-VL's collator pads images to multiples of ``patch_size *
spatial_merge_size = 28``, so post-merger ``(h, w)`` is always at least 2.
We assert evenness at forward time and raise a clear error if not.

Shape contract
--------------
``forward(merged_features, grid_thw_post)`` where:

    merged_features   : (B, N_post, in_features=lm_dim)
                        per-sample post-merger features. ``B`` here is the
                        *number of video items* (one per cam in 3-cam mode), so
                        a real LM batch of ``B_lm`` samples lifts to a
                        ``B_lm * num_cams`` projector batch.
    grid_thw_post     : (B, 3) — [t, h_post, w_post] in *post-merger* units.

    out               : (B, N_post // 4, lm_dim)

The HF wiring (see ``train_lora.forward_with_pixelshuffle_projector``) feeds
this projector the model's own ``get_video_features`` output (which already
applies the in-encoder merger), reshaped per-item, and the post-merger grid.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


__all__ = ["Qwen2VLPixelShufflePlusLinearProjector"]


class Qwen2VLPixelShufflePlusLinearProjector(nn.Module):
    """PixelShuffle 2x + Linear vision-to-LM projector for HF Qwen2.5-VL.

    Operates on **post-merger** features (the output of ``model.visual``'s
    built-in ``PatchMerger``). Deterministic 4x spatial token compression.

    Parameters
    ----------
    in_features : int
        Channel dim of incoming visual features. For Qwen2.5-VL-3B this is
        ``lm_dim = 2048`` (post-merger features come out at LM hidden dim).
    lm_dim : int
        LM hidden dim (output channel dim of the projector). 2048 for 3B.
    shuffle_ratio : int
        Spatial compression ratio. Must be >= 1. Each output token pools a
        ``shuffle_ratio x shuffle_ratio`` block. Default 2 (LLaVA-NeXT).

    Notes
    -----
    Unlike Q-Former, there are NO learnable query parameters; the projector is
    a single Linear sized ``(in_features * shuffle_unit) -> lm_dim``. For the
    default config (in_features=2048, lm_dim=2048, shuffle_ratio=2) this is
    ~16.78M params + 2048 bias.
    """

    @dataclass
    class Config:
        in_features: int = 2048
        lm_dim: int = 2048
        shuffle_ratio: int = 2

    def __init__(
        self,
        in_features: int = 2048,
        lm_dim: int = 2048,
        shuffle_ratio: int = 2,
    ):
        super().__init__()
        if shuffle_ratio < 1:
            raise ValueError(
                f"shuffle_ratio must be >= 1, got {shuffle_ratio}"
            )
        self.in_features = int(in_features)
        self.lm_dim = int(lm_dim)
        self.shuffle_ratio = int(shuffle_ratio)
        self.shuffle_unit = self.shuffle_ratio ** 2
        # Single Linear: (in_features * r^2) -> lm_dim
        self.proj = nn.Linear(self.in_features * self.shuffle_unit, self.lm_dim)

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _pixel_unshuffle_2d(x: torch.Tensor, shuffle_ratio: int) -> torch.Tensor:
        """Space-to-depth on a (B, H, W, C) tensor.

        Reorders so that each output token covers a ``r x r`` spatial block.
        Intra-block layout is row-major (top-left, top-right, bottom-left,
        bottom-right) — matches LLaVA-NeXT / InternVL convention.

        Args:
            x: ``(B, H, W, C)`` with H, W divisible by ``shuffle_ratio``.
            shuffle_ratio: ``r``.

        Returns:
            ``(B, H // r, W // r, C * r * r)``.
        """
        B, H, W, C = x.shape
        r = shuffle_ratio
        if H % r != 0 or W % r != 0:
            raise ValueError(
                f"PixelShuffle requires H ({H}) and W ({W}) divisible by "
                f"shuffle_ratio ({r}). Pad upstream (collator) if needed."
            )
        x = x.view(B, H // r, r, W // r, r, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # (B, H/r, W/r, r, r, C)
        x = x.view(B, H // r, W // r, r * r * C)
        return x

    # --------------------------------------------------------------- forward
    def forward(
        self,
        vision_features: torch.Tensor,
        *,
        grid_thw_post: torch.Tensor,
    ) -> torch.Tensor:
        """Apply PixelShuffle + Linear.

        Args:
            vision_features: ``(B, N_post, in_features)`` per-sample
                post-merger features. ``N_post`` must equal
                ``t * h_post * w_post`` for the matching grid_thw row (all
                items in the batch must share the same (t, h_post, w_post)).
            grid_thw_post: ``(B, 3)`` per-item ``[t, h_post, w_post]`` patch
                counts in **post-merger units** (i.e. raw grid_thw // 2 on
                the spatial axes for Qwen2.5-VL's spatial_merge_size=2).

        Returns:
            ``(B, N_post // shuffle_unit, lm_dim)`` — compressed and
            LM-dim-projected visual features.
        """
        if vision_features.dim() != 3:
            raise ValueError(
                f"vision_features must be (B, N, C); got shape "
                f"{tuple(vision_features.shape)}"
            )
        B, N, C = vision_features.shape
        if C != self.in_features:
            raise ValueError(
                f"Expected in_features={self.in_features}, got channel dim "
                f"{C} in vision_features of shape {tuple(vision_features.shape)}"
            )
        if grid_thw_post.ndim != 2 or grid_thw_post.shape[1] != 3:
            raise ValueError(
                f"grid_thw_post must have shape (B, 3); got "
                f"{tuple(grid_thw_post.shape)}"
            )
        if grid_thw_post.shape[0] != B:
            raise ValueError(
                f"grid_thw_post batch dim ({grid_thw_post.shape[0]}) does not "
                f"match vision_features batch dim ({B})"
            )

        # All items must share the same (t, h, w) for the per-batch rearrange
        # to be valid. This is the standard nuScenes case (all cams + frames
        # at fixed resolution under min_pixels==max_pixels). For
        # mixed-resolution batches the caller must group by shape.
        first = grid_thw_post[0]
        if not torch.equal(
            grid_thw_post, first.unsqueeze(0).expand_as(grid_thw_post)
        ):
            raise ValueError(
                "PixelShuffle projector requires all items in the batch to "
                "share the same (t, h_post, w_post). For mixed-resolution "
                "batches, group by shape and call this projector per group. "
                f"grid_thw_post = {grid_thw_post.tolist()}"
            )
        t = int(first[0].item())
        h = int(first[1].item())
        w = int(first[2].item())

        valid_len = t * h * w
        if valid_len != N:
            raise ValueError(
                f"grid_thw_post implies {valid_len} tokens but "
                f"vision_features has N={N}. PixelShuffle projector expects "
                f"the per-item length to exactly equal t*h_post*w_post (no "
                f"padding tail). Caller should split the padded vision output "
                f"per-item before calling this projector."
            )

        r = self.shuffle_ratio
        if h % r != 0 or w % r != 0:
            raise ValueError(
                f"PixelShuffle ratio {r} requires post-merger h ({h}) and w "
                f"({w}) to be divisible by {r}. Got grid_thw_post[0]="
                f"{[t, h, w]}. Qwen2.5-VL's collator pads images to multiples "
                f"of patch_size * spatial_merge_size = 28, so post-merger "
                f"(h, w) should be even; check the wiring layer's grid_thw "
                f"computation."
            )

        # 1. Flatten temporal into batch so PixelShuffle is spatial-only.
        x = vision_features.view(B * t, h, w, C)
        # 2. Space-to-depth: (B*t, h, w, C) -> (B*t, h/r, w/r, C*r*r)
        x = self._pixel_unshuffle_2d(x, r)
        # 3. Flatten the per-frame spatial grid back into a sequence and
        #    re-collapse t into the per-sample sequence dim.
        x = x.view(B, t * (h // r) * (w // r), C * r * r)
        # 4. Project to lm_dim.
        out = self.proj(x)  # (B, t*(h/r)*(w/r), lm_dim)
        return out

    # ----------------------------------------------------------- introspect
    def output_token_count(self, N_post: int) -> int:
        """Number of LM tokens this projector emits per item, given a
        per-item post-merger token count ``N_post``.

        Used by the wiring layer to size the trimmed ``<|video_pad|>`` block.
        """
        if N_post % self.shuffle_unit != 0:
            raise ValueError(
                f"N_post={N_post} not divisible by shuffle_unit="
                f"{self.shuffle_unit}; (h_post, w_post) must both be even."
            )
        return N_post // self.shuffle_unit

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"in_features={self.in_features}, lm_dim={self.lm_dim}, "
            f"shuffle_ratio={self.shuffle_ratio}"
        )


# ---------------------------------------------------------------------------
# CPU smoke test — run as `python pixelshuffle_projector_hf.py`
# ---------------------------------------------------------------------------

def _smoke() -> None:
    """Quick CPU smoke test: shape, gradient flow, param count."""
    torch.manual_seed(0)
    # Construct production-ish projector. 3-cam x 4-frame nuScenes after
    # 2x2 spatial merger:
    #   per-cam: t=4, h_post=10, w_post=14  -> 4*10*14 = 560 tokens / cam
    #   3 cams = 1680 total tokens / sample (matches our config doc).
    # We test on a per-cam-item batch of 2 samples (B_proj = 2).
    in_features = 2048
    lm_dim = 2048
    proj = Qwen2VLPixelShufflePlusLinearProjector(
        in_features=in_features, lm_dim=lm_dim, shuffle_ratio=2,
    )

    # Param count check (10M < count < 30M).
    n_param = sum(p.numel() for p in proj.parameters())
    assert 10_000_000 < n_param < 30_000_000, (
        f"Param count {n_param:,} outside expected envelope (10M, 30M)"
    )
    print(f"[smoke] params: {n_param:,} ({n_param/1e6:.2f}M)")

    # Per-cam shape (single item / cam): t=4, h=10, w=14 -> 560 tokens
    # Pack 2 items into one batch for the projector.
    t, h, w = 4, 10, 14
    N = t * h * w  # 560
    B = 2
    x = torch.randn(B, N, in_features, requires_grad=True)
    grid_thw_post = torch.tensor([[t, h, w]] * B, dtype=torch.long)
    out = proj(x, grid_thw_post=grid_thw_post)
    expected_N_out = t * (h // 2) * (w // 2)  # 4 * 5 * 7 = 140
    assert out.shape == (B, expected_N_out, lm_dim), (
        f"unexpected out shape: {tuple(out.shape)}, "
        f"expected ({B}, {expected_N_out}, {lm_dim})"
    )
    print(
        f"[smoke] forward OK: in={tuple(x.shape)} -> out={tuple(out.shape)} "
        f"(4x compression)"
    )

    # Gradient flow: backward through proj.
    loss = out.float().pow(2).mean()
    loss.backward()
    assert proj.proj.weight.grad is not None, "no grad on proj.weight"
    grad_norm = proj.proj.weight.grad.float().norm().item()
    assert grad_norm > 0, f"zero grad norm: {grad_norm}"
    print(f"[smoke] backward OK: proj.weight grad_norm={grad_norm:.3e}")

    # Full-rig shape test from the agent spec: B=2, N=1680, in=2048
    # (3-cam concat into one projector batch). For this case we pretend
    # the 3 cams have been concatenated along the temporal axis so a
    # single (t=12, h=10, w=14) grid fits — only for shape verification.
    proj2 = Qwen2VLPixelShufflePlusLinearProjector(
        in_features=in_features, lm_dim=lm_dim, shuffle_ratio=2,
    )
    t2, h2, w2 = 12, 10, 14  # 12 * 10 * 14 = 1680
    x2 = torch.randn(B, t2 * h2 * w2, in_features)
    g2 = torch.tensor([[t2, h2, w2]] * B, dtype=torch.long)
    o2 = proj2(x2, grid_thw_post=g2)
    expected2 = t2 * (h2 // 2) * (w2 // 2)  # 12 * 5 * 7 = 420
    assert o2.shape == (B, expected2, lm_dim), (
        f"3-cam concat shape mismatch: {tuple(o2.shape)} vs "
        f"({B}, {expected2}, {lm_dim})"
    )
    print(
        f"[smoke] 3-cam-stack shape OK: 1680 -> {expected2} tokens "
        f"(matches spec)"
    )

    # Odd (H,W) rejection.
    try:
        bad = torch.randn(B, 3 * 5 * 5, in_features)  # h=5 odd
        proj(bad, grid_thw_post=torch.tensor([[3, 5, 5]] * B, dtype=torch.long))
    except ValueError as e:
        print(f"[smoke] odd-H rejection OK: {e!s:.80s}...")
    else:
        raise AssertionError("odd H should have raised")

    print("[smoke] PASS")


if __name__ == "__main__":
    _smoke()
