"""Collect calibrated per-tensor input AMAX for the Qwen3-VL ViT export wrapper,
in PURE PYTORCH (no ORT -> no [2,16,5600,5600] OOM).

This is modelopt's "max" calibration algorithm done by hand: run the
VisualExportWrapper over the real calib set, hook every nn.Linear, and keep the
running max of |input|.amax() per layer. Attention scores live in torch (<1GB),
so the ORT-calibration OOM never happens.

Output: a JSON {node_basename: input_amax} keyed by the ONNX MatMul/Gemm node
path (e.g. "/blocks.0/attn/qkv"), consumed by inject_vit_qdq.py.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import apply_venv_shims  # noqa
apply_venv_shims()

import numpy as np
import torch
import torch.nn as nn

# reuse the EXACT export wrapper so amax matches the exported graph's activations
from build_vit_trt_engine import (
    VisualExportWrapper, precompute_constants, CKPT, GRID, SEQ,
)
from transformers import Qwen3VLForConditionalGeneration

HERE = os.path.dirname(os.path.abspath(__file__))

# torch module name (in wrapper) -> ONNX node path basename. The wrapper renames
# nothing; module `blocks.0.attn.qkv` exports as node "/blocks.0/attn/qkv/MatMul".
# Mergers are Gemm "/merger/linear_fc1" etc. We key by the torch module name and
# convert to "/...." path in the injector.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ncalib", type=int, default=6)
    ap.add_argument("--out", default=f"{HERE}/results/vit_input_amax.json")
    args = ap.parse_args()

    device = "cuda:0"
    print("[1] loading HF model (fp32 for faithful amax)...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        CKPT, dtype=torch.float32, device_map=device)
    visual = model.model.visual.eval()
    visual.config._attn_implementation = "sdpa"

    pos_embeds, cos_f, sin_f, num_frames = precompute_constants(visual, device)
    wrapper = VisualExportWrapper(
        visual, pos_embeds.float(), cos_f.float(), sin_f.float(), num_frames
    ).eval().to(device)

    # hook every Linear in the wrapper for per-tensor input amax (max over calib)
    amax = {}

    def mk(name):
        def h(mod, inp, out):
            v = float(inp[0].detach().abs().amax())
            amax[name] = max(amax.get(name, 0.0), v)
        return h

    hooks = []
    for n, m in wrapper.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(mk(n)))
    print(f"[2] hooked {len(hooks)} Linear layers")

    calib = np.load("/tmp/vit_calib.npy", mmap_mode="r")
    ncalib = min(args.ncalib, calib.shape[0])
    print(f"[3] calibrating over {ncalib} samples (max algorithm)...")
    with torch.inference_mode():
        for i in range(ncalib):
            x = torch.from_numpy(np.ascontiguousarray(calib[i])).to(device, torch.float32)
            wrapper(x)
            print(f"    sample {i} done")
    for h in hooks:
        h.remove()

    print(f"[4] collected input amax for {len(amax)} layers")
    # sanity: show range
    vals = sorted(amax.values())
    print(f"    input-amax min={vals[0]:.3f} max={vals[-1]:.3f} median={vals[len(vals)//2]:.3f}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(amax, f, indent=1)
    print("[saved]", args.out)


if __name__ == "__main__":
    main()
