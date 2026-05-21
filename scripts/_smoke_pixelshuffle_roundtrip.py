"""Round-trip smoke test for the PixelShuffle external-projector save/load.

Verifies that:
  1. Qwen2VLPixelShufflePlusLinearProjector saved via the train_lora
     _save_external_projector mechanism (state_dict + meta JSON)
  2. ... can be loaded back via planning_eval._maybe_load_external_projector
  3. ... and produces identical outputs on the same input.

CPU-only; takes <1 s. Run:
  /usr/bin/python3 scripts/_smoke_pixelshuffle_roundtrip.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from pixelshuffle_projector_hf import Qwen2VLPixelShufflePlusLinearProjector  # noqa: E402


def _projector_constructor_kwargs(projector, projector_type: str) -> dict:
    """Inlined copy of train_lora._projector_constructor_kwargs for the
    smoke test (avoids importing train_lora's heavy torch/accelerate path).
    Must stay byte-equal to the train_lora pixelshuffle branch."""
    t = projector_type.lower()
    if t == "pixelshuffle":
        return {
            "in_features": int(projector.in_features),
            "lm_dim": int(projector.lm_dim),
            "shuffle_ratio": int(projector.shuffle_ratio),
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
    assert meta["type"] == "pixelshuffle"
    projector = Qwen2VLPixelShufflePlusLinearProjector(**meta["config"])
    state = torch.load(weights_path, map_location="cpu")
    projector.load_state_dict(state)
    projector.eval()
    return projector


def main() -> int:
    torch.manual_seed(0)

    # Toy dims — small enough for CPU to run in <1 s but large enough to
    # exercise every constructor knob.
    cfg = dict(
        in_features=64,
        lm_dim=48,
        shuffle_ratio=2,
    )

    orig = Qwen2VLPixelShufflePlusLinearProjector(**cfg)
    orig.eval()

    # Known input: B=2, per-item grid (t=2, h=4, w=6) -> N=48, in_features=64.
    # h and w are even so shuffle_ratio=2 divides cleanly.
    B = 2
    t, h, w = 2, 4, 6
    N = t * h * w
    x = torch.randn(B, N, cfg["in_features"])
    grid_thw_post = torch.tensor([[t, h, w]] * B, dtype=torch.long)

    with torch.no_grad():
        out_orig = orig(x, grid_thw_post=grid_thw_post)
    expected_N_out = t * (h // 2) * (w // 2)
    assert out_orig.shape == (B, expected_N_out, cfg["lm_dim"]), (
        f"unexpected output shape: {out_orig.shape}"
    )

    with tempfile.TemporaryDirectory() as tmp:
        _save(orig, "pixelshuffle", tmp)
        # Sanity: both files exist.
        assert os.path.exists(os.path.join(tmp, "projector.pt"))
        assert os.path.exists(os.path.join(tmp, "projector_meta.json"))

        loaded = _load(tmp)
        with torch.no_grad():
            out_loaded = loaded(x, grid_thw_post=grid_thw_post)

    if not torch.allclose(out_orig, out_loaded, atol=0, rtol=0):
        max_abs = (out_orig - out_loaded).abs().max().item()
        if not torch.allclose(out_orig, out_loaded, atol=1e-6, rtol=1e-5):
            print(f"FAIL: outputs differ; max_abs={max_abs}")
            return 1
        else:
            print(f"WARN: outputs differ within tolerance; max_abs={max_abs}")

    # Verify the recovered config round-trips.
    with tempfile.TemporaryDirectory() as tmp:
        _save(orig, "pixelshuffle", tmp)
        with open(os.path.join(tmp, "projector_meta.json")) as f:
            meta = json.load(f)
        for k, v in cfg.items():
            if meta["config"][k] != v:
                print(f"FAIL: meta[{k}]={meta['config'][k]!r} != {v!r}")
                return 1

    # output_token_count helper sanity (used by the planning_eval trim path).
    assert orig.output_token_count(N) == expected_N_out, (
        f"output_token_count({N}) = {orig.output_token_count(N)} != {expected_N_out}"
    )

    print("ROUND-TRIP OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
