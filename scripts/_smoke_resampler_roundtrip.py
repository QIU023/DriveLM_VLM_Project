"""Round-trip smoke test for the Perceiver Resampler external-projector save/load.

Verifies that:
  1. Qwen2VLPerceiverResamplerProjector saved via the train_lora
     _save_external_projector mechanism (state_dict + meta JSON)
  2. ... can be loaded back via planning_eval._maybe_load_external_projector
  3. ... and produces identical outputs on the same input.

CPU-only; takes a couple of seconds. Run:
  /usr/bin/python3 scripts/_smoke_resampler_roundtrip.py

ALSO exercises the projector forward at the REAL post-merger grid shape used
by the 1-cam x 4f Track A.3 launch config — 4 frames x 8 (h_post) x 15 (w_post)
= 480 tokens/sample at base min/max_pixels=109760. The Perceiver Resampler is
shape-agnostic (cross-attention pools over arbitrary N), so this should work
without any min/max_pixels override (unlike A.2 PixelShuffle, which required
an even-grid budget bump). This smoke is what proves that claim.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from perceiver_resampler_projector_hf import Qwen2VLPerceiverResamplerProjector  # noqa: E402


def _projector_constructor_kwargs(projector, projector_type: str) -> dict:
    """Inlined copy of train_lora._projector_constructor_kwargs for the
    smoke test (avoids importing train_lora's heavy torch/accelerate path).
    Must stay byte-equal to the train_lora resampler branch."""
    t = projector_type.lower()
    if t == "resampler":
        return {
            "in_features": int(projector.in_features),
            "lm_dim": int(projector.lm_dim),
            "internal_dim": int(projector.internal_dim),
            "num_latents": int(projector.num_latents),
            "num_layers": int(projector.num_layers),
            "n_heads": int(projector.n_heads),
            "ffn_mult": int(projector.ffn_mult),
            "layer_norm_eps": float(getattr(projector.norm_out, "eps", 1e-6)),
            "t_max": int(projector.t_max),
        }
    raise ValueError(f"Unknown projector_type={projector_type!r}")


def _save(projector, projector_type: str, save_dir: str) -> None:
    """Mirror of train_lora._save_external_projector but without an Accelerator.
    The Accelerator wrap is a no-op on rank 0 in single-process mode, so this
    is a faithful round-trip of the on-disk artifact format."""
    os.makedirs(save_dir, exist_ok=True)
    sd = projector.state_dict()
    meta = {
        "type": projector_type.lower(),
        "config": _projector_constructor_kwargs(projector, projector_type),
    }
    torch.save(sd, os.path.join(save_dir, "projector.pt"))
    with open(os.path.join(save_dir, "projector_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def _load(load_dir: str):
    """Mirror of planning_eval._maybe_load_external_projector (CPU path)."""
    meta_path = os.path.join(load_dir, "projector_meta.json")
    weights_path = os.path.join(load_dir, "projector.pt")
    with open(meta_path) as f:
        meta = json.load(f)
    assert meta["type"] == "resampler"
    projector = Qwen2VLPerceiverResamplerProjector(**meta["config"])
    state = torch.load(weights_path, map_location="cpu")
    projector.load_state_dict(state)
    projector.eval()
    return projector


def main() -> int:
    torch.manual_seed(0)

    # ---- Stage 1: toy dims --------------------------------------------------
    # Small enough for CPU to run in <1 s but large enough to exercise every
    # constructor knob (decoupled internal_dim, multi-layer stack, per-frame
    # temporal_pos lookup).
    cfg = dict(
        in_features=64,
        lm_dim=48,
        internal_dim=32,
        num_latents=8,
        num_layers=2,
        n_heads=4,
        ffn_mult=2,
        layer_norm_eps=1e-6,
        t_max=8,
    )

    orig = Qwen2VLPerceiverResamplerProjector(**cfg)
    orig.eval()

    # Toy input: B=2, per-item grid (t=2, h=3, w=5) -> N=30, in_features=64.
    # NOTE: h, w are intentionally ODD to confirm the resampler has NO
    # even-grid requirement (the A.2 PixelShuffle lesson).
    B = 2
    t_toy, h_toy, w_toy = 2, 3, 5
    N_toy = t_toy * h_toy * w_toy
    x_toy = torch.randn(B, N_toy, cfg["in_features"])
    grid_thw_toy = torch.tensor([[t_toy, h_toy, w_toy]], dtype=torch.long)

    with torch.no_grad():
        out_orig_toy = orig(x_toy, grid_thw=grid_thw_toy)
    assert out_orig_toy.shape == (B, cfg["num_latents"], cfg["lm_dim"]), (
        f"unexpected toy output shape: {out_orig_toy.shape}"
    )

    with tempfile.TemporaryDirectory() as tmp:
        _save(orig, "resampler", tmp)
        assert os.path.exists(os.path.join(tmp, "projector.pt"))
        assert os.path.exists(os.path.join(tmp, "projector_meta.json"))

        loaded = _load(tmp)
        with torch.no_grad():
            out_loaded_toy = loaded(x_toy, grid_thw=grid_thw_toy)

    if not torch.allclose(out_orig_toy, out_loaded_toy, atol=0, rtol=0):
        max_abs = (out_orig_toy - out_loaded_toy).abs().max().item()
        if not torch.allclose(out_orig_toy, out_loaded_toy, atol=1e-6, rtol=1e-5):
            print(f"FAIL (toy): outputs differ; max_abs={max_abs}")
            return 1
        else:
            print(f"WARN (toy): outputs differ within tolerance; max_abs={max_abs}")

    # Verify recovered config round-trips exactly.
    with tempfile.TemporaryDirectory() as tmp:
        _save(orig, "resampler", tmp)
        with open(os.path.join(tmp, "projector_meta.json")) as f:
            meta = json.load(f)
        for k, v in cfg.items():
            if meta["config"][k] != v:
                print(f"FAIL: meta[{k}]={meta['config'][k]!r} != {v!r}")
                return 1

    # ---- Stage 2: REAL post-merger grid shape (1-cam x 4f, base 109760) -----
    # Driving config: 1 cam x 4 frames at min/max_pixels=109760. nuScenes
    # 16:9 (1600x900) smart_resize -> 224x420 pixels -> patch 16x30 -> after
    # in-encoder PatchMerger 2x2 -> 8x15 post-merger spatial grid per frame.
    # Per-item visual token count: t=4 * h_post=8 * w_post=15 = 480.
    # Out: (B, num_latents=64, lm_dim=2048) regardless of input N — proving
    # the resampler is data-shape agnostic (no even-grid req like A.2).
    real_cfg = dict(
        in_features=2048,
        lm_dim=2048,
        internal_dim=1024,
        num_latents=64,
        num_layers=6,
        n_heads=8,
        ffn_mult=2,
        layer_norm_eps=1e-6,
        t_max=32,
    )
    real = Qwen2VLPerceiverResamplerProjector(**real_cfg)
    real.eval()
    n_params = sum(p.numel() for p in real.parameters())
    print(f"[stage2] real-config param count = {n_params/1e6:.2f}M "
          f"(target band 80-110M; commit 579d9dc claims 90.44M)")
    if not (80e6 <= n_params <= 110e6):
        print(f"FAIL: param count {n_params/1e6:.2f}M outside 80-110M band")
        return 1

    B_real = 1
    t_real, h_real, w_real = 4, 8, 15  # 1-cam x 4f at base 109760
    N_real = t_real * h_real * w_real  # 480
    x_real = torch.randn(B_real, N_real, real_cfg["in_features"])
    grid_thw_real = torch.tensor([[t_real, h_real, w_real]], dtype=torch.long)
    with torch.no_grad():
        out_real = real(x_real, grid_thw=grid_thw_real)
    if out_real.shape != (B_real, real_cfg["num_latents"], real_cfg["lm_dim"]):
        print(f"FAIL (real grid 4x8x15): output shape {out_real.shape} != "
              f"({B_real}, {real_cfg['num_latents']}, {real_cfg['lm_dim']})")
        return 1
    print(f"[stage2] real-grid forward OK: N_in={N_real} (4x8x15, ODD w) "
          f"-> N_out={out_real.shape[1]} (= num_latents=64)")

    # Save -> load -> parity on the REAL config too.
    with tempfile.TemporaryDirectory() as tmp:
        _save(real, "resampler", tmp)
        loaded_real = _load(tmp)
        with torch.no_grad():
            out_real_loaded = loaded_real(x_real, grid_thw=grid_thw_real)
    if not torch.allclose(out_real, out_real_loaded, atol=1e-6, rtol=1e-5):
        max_abs = (out_real - out_real_loaded).abs().max().item()
        print(f"FAIL (real): outputs differ post-roundtrip; max_abs={max_abs}")
        return 1
    print("[stage2] real-grid round-trip parity OK")

    # ---- Stage 3: sanity that the projector is data-shape agnostic ----------
    # Different post-merger grid, same projector -> still emits num_latents.
    # This is the property that lets A.3 use base 109760 unlike A.2 which
    # needed 150528 for even h/w. Smoke a few non-divisible shapes.
    for (tt, hh, ww) in [(4, 10, 18), (4, 7, 14), (1, 1, 7)]:
        n = tt * hh * ww
        xx = torch.randn(B_real, n, real_cfg["in_features"])
        gg = torch.tensor([[tt, hh, ww]], dtype=torch.long)
        with torch.no_grad():
            yy = real(xx, grid_thw=gg)
        if yy.shape != (B_real, real_cfg["num_latents"], real_cfg["lm_dim"]):
            print(f"FAIL (shape-agnosticism {tt}x{hh}x{ww}): {yy.shape}")
            return 1
    print("[stage3] shape-agnostic across (4,10,18) / (4,7,14) / (1,1,7) OK")

    print("ROUND-TRIP OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
