"""Round-trip smoke test for the external-projector save/load mechanism.

Verifies that:
  1. Qwen2VLQFormerProjector saved via the train_lora _save_external_projector
     mechanism (state_dict + meta JSON)
  2. ... can be loaded back via planning_eval._maybe_load_external_projector
  3. ... and produces identical outputs on the same input.

CPU-only; takes a few seconds. Run:
  /usr/bin/python3 scripts/_smoke_qformer_roundtrip.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from qformer_projector_hf import Qwen2VLQFormerProjector  # noqa: E402


def _projector_constructor_kwargs(projector, projector_type: str) -> dict:
    """Inlined copy of train_lora._projector_constructor_kwargs for the
    smoke test (avoids importing train_lora's heavy torch/accelerate path).
    Must stay byte-equal to the train_lora version."""
    t = projector_type.lower()
    if t == "qformer":
        return {
            "vit_dim": int(projector.vit_dim),
            "internal_dim": int(projector.internal_dim),
            "lm_dim": int(projector.lm_dim),
            "num_queries": int(projector.num_queries),
            "num_layers": int(projector.num_layers),
            "n_heads": int(projector.n_heads),
            "ffn_mult": int(projector.ffn_mult),
            "layer_norm_eps": float(getattr(projector.norm_out, "eps", 1e-6)),
            "dropout": 0.0,
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
    assert meta["type"] == "qformer"
    projector = Qwen2VLQFormerProjector(**meta["config"])
    state = torch.load(weights_path, map_location="cpu")
    projector.load_state_dict(state)
    projector.eval()
    return projector


def main() -> int:
    torch.manual_seed(0)

    # Toy dims — small enough for CPU to run in <1s but large enough to
    # exercise every constructor knob.
    cfg = dict(
        vit_dim=128,
        internal_dim=64,
        lm_dim=96,
        num_queries=8,
        num_layers=2,
        n_heads=4,
        ffn_mult=2,
    )

    orig = Qwen2VLQFormerProjector(**cfg)
    orig.eval()

    # Known input: (B=2, N_vision=10, vit_dim=128).
    x = torch.randn(2, 10, cfg["vit_dim"])

    with torch.no_grad():
        out_orig = orig(x)
    assert out_orig.shape == (2, cfg["num_queries"], cfg["lm_dim"]), (
        f"unexpected output shape: {out_orig.shape}"
    )

    with tempfile.TemporaryDirectory() as tmp:
        _save(orig, "qformer", tmp)
        # Sanity: both files exist.
        assert os.path.exists(os.path.join(tmp, "projector.pt"))
        assert os.path.exists(os.path.join(tmp, "projector_meta.json"))

        loaded = _load(tmp)
        with torch.no_grad():
            out_loaded = loaded(x)

    if not torch.allclose(out_orig, out_loaded, atol=0, rtol=0):
        # Try with tiny tolerance in case of nondeterministic kernels.
        max_abs = (out_orig - out_loaded).abs().max().item()
        if not torch.allclose(out_orig, out_loaded, atol=1e-6, rtol=1e-5):
            print(f"FAIL: outputs differ; max_abs={max_abs}")
            return 1
        else:
            print(f"WARN: outputs differ within tolerance; max_abs={max_abs}")

    # Verify the recovered config round-trips.
    with tempfile.TemporaryDirectory() as tmp:
        _save(orig, "qformer", tmp)
        with open(os.path.join(tmp, "projector_meta.json")) as f:
            meta = json.load(f)
        for k, v in cfg.items():
            if meta["config"][k] != v:
                print(f"FAIL: meta[{k}]={meta['config'][k]!r} != {v!r}")
                return 1

    print("ROUND-TRIP OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
