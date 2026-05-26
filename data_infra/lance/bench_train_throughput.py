"""Tier-2 REAL training-step A/B throughput bench (3-cam NATIVE planning SFT).

Compares the per-step cost of the LIVE vs CACHED vision path:
  Arm A (LIVE)  : forward_with_video_compression_free  -> frozen ViT runs every
                  step (8400 video patches through the vision tower) + FasterVLM
                  x4 + deepstack, then the LM fwd/bwd.
  Arm B (CACHED): forward_with_cached_vision_tokens     -> ViT NEVER runs; cached
                  int8 vision tokens scattered into inputs_embeds; same LM fwd/bwd.

Each arm runs in its OWN subprocess (clean GPU memory) and prints a parseable
RESULT line. The orchestrator collects them and reports steps/sec + speedup.

NATIVE (700 tok/cam) measurements:
  * LIVE full fwd+bwd+opt is expected to OOM on a single 32G GPU: real 3-cam
    NATIVE full_sft trains all 4B params under 8-GPU FSDP (sharded optimizer);
    one GPU cannot hold the full-model backward at native resolution. We RECORD
    the OOM (cached enables a config the live single-GPU path cannot run).
  * FORWARD-ONLY native A/B (both fit) isolates exactly the per-step cost the
    cache removes (the ViT forward over 8400 patches + FasterVLM).

REDUCED (240 tok/cam @ video_max_pixels=524288) both-arms-fit A/B: full
fwd+bwd+opt for a clean number where the LIVE path also fits.

Usage:
    export HF_HOME=/workspace/.hf_home; unset HF_HUB_OFFLINE
    CUDA_VISIBLE_DEVICES=0 /usr/bin/python3 data_infra/lance/bench_train_throughput.py \
        --lance data_infra/lance/nusc_3cam_native.lance --steps 30 --batch-size 1
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
_SCRIPTS = os.path.join(_REPO, "scripts")
for _p in (_SCRIPTS, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CONFIG = os.path.join(_REPO, "configs", "nuscenes_planning_3cam_qwen3vl_NATIVE.yaml")


def _freeze_vision(model):
    visual = getattr(getattr(model, "model", model), "visual", None)
    if visual is not None:
        for p in visual.parameters():
            p.requires_grad = False


def _time_loop(batches, fwd_fn, steps, backward, opt, warmup=3):
    times = []
    torch.cuda.reset_peak_memory_stats()
    ctx = torch.enable_grad() if backward else torch.no_grad()
    for s in range(steps + warmup):
        b = batches[s % len(batches)]
        torch.cuda.synchronize(); t0 = time.perf_counter()
        if backward:
            opt.zero_grad(set_to_none=True)
        with ctx:
            out = fwd_fn(b)
        if backward:
            out.loss.backward(); opt.step()
        torch.cuda.synchronize()
        if s >= warmup:
            times.append(time.perf_counter() - t0)
    return float(np.median(times)), torch.cuda.max_memory_allocated() / 1e9


# ----------------------------------------------------------------------------
# Worker: run ONE arm and print a RESULT line.
# arm in {live_native_train, live_native_fwd, cached_native_fwd,
#         live_reduced_train, cached_reduced_train}
# ----------------------------------------------------------------------------
def run_arm(arm, lance_path, steps, bs):
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from train_lora import (
        load_config, collate_fn,
        forward_with_video_compression_free, forward_with_cached_vision_tokens,
    )
    from multimodal_planning_dataset import build_multimodal_planning_dataset
    from cached_native_dataset import CachedNativeDataset, collate_cached

    reduced = arm.startswith("live_reduced") or arm.startswith("cached_reduced")
    cfg = load_config(CONFIG)
    model_id = cfg["model_id"]
    device = "cuda"
    proc = AutoProcessor.from_pretrained(model_id)
    proc.image_processor.min_pixels = cfg["min_pixels"]
    proc.image_processor.max_pixels = cfg["max_pixels"]
    if reduced:
        vp = proc.video_processor
        for attr in ("min_pixels", "max_pixels"):
            if hasattr(vp, attr):
                setattr(vp, attr, 524288)
        if hasattr(vp, "size") and vp.size is not None:
            if hasattr(vp.size, "shortest_edge"):
                vp.size.shortest_edge = 524288
            if hasattr(vp.size, "longest_edge"):
                vp.size.longest_edge = 524288
    vt = proc.tokenizer.convert_tokens_to_ids("<|video_pad|>")
    it = proc.tokenizer.convert_tokens_to_ids("<|image_pad|>")

    live_ds = build_multimodal_planning_dataset(cfg, proc, split="train")
    model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16).to(device)
    _freeze_vision(model)
    nb = max(4, steps // 4)

    is_cached = arm.startswith("cached")
    backward = arm.endswith("_train")

    if backward:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
        opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-6)
    else:
        model.eval(); opt = None

    # Build the batches for this arm.
    if is_cached and not reduced:
        cache_ds = CachedNativeDataset(lance_path)
        batches = []
        for j in range(nb):
            idxs = [(j * bs + k) % len(cache_ds) for k in range(bs)]
            cb = collate_cached([cache_ds[i] for i in idxs])
            batches.append({k: (v.to(device) if torch.is_tensor(v) else v) for k, v in cb.items()})
        fwd = lambda b: forward_with_cached_vision_tokens(model, b, vt, it)
    elif is_cached and reduced:
        # build an in-memory 240-tok cache (int8 round-trip) for these samples
        from native_cache_common import (MERGE_SIZE, COMPRESS_RATIO, compress_video_features,
                                          trim_native_layout, quantize_int8, dequantize_int8)
        inner = model.model
        batches = []
        for j in range(nb):
            items = [live_ds[(j * bs + k) % len(live_ds)] for k in range(bs)]
            cvp, cvd, cip, cid = [], [], [], []
            c_ids, c_lab, c_attn, c_mm, c_vg, c_ig = [], [], [], [], [], []
            for item in items:
                ids = item["input_ids"].to(device); attn = item["attention_mask"].to(device)
                lab = item["labels"].to(device)
                mm = item.get("mm_token_type_ids"); mm = mm.to(device) if mm is not None else None
                vg = item["video_grid_thw"].to(device)
                if vg.dim() == 1: vg = vg.unsqueeze(0)
                ig = item["image_grid_thw"].to(device)
                if ig.dim() == 1: ig = ig.unsqueeze(0)
                pv = item["pixel_values_videos"].to(device, torch.bfloat16)
                pvi = item["pixel_values"].to(device, torch.bfloat16)
                po = [int(vg[i,0])*(int(vg[i,1])//MERGE_SIZE)*(int(vg[i,2])//MERGE_SIZE) for i in range(vg.shape[0])]
                pc = [max(1, c//COMPRESS_RATIO) for c in po]
                tr = trim_native_layout(ids, attn, lab, mm, vg, vt, compress_ratio=COMPRESS_RATIO, merge_size=MERGE_SIZE)
                with torch.no_grad():
                    vo = inner.get_video_features(pv, vg); io = inner.get_image_features(pvi, ig)
                vpool = vo.pooler_output
                if not isinstance(vpool, (list, tuple)): vpool = list(torch.split(vpool, po))
                cp, cd = compress_video_features(list(vpool), list(vo.deepstack_features), po, pc, COMPRESS_RATIO)
                ipool = io.pooler_output; ipool = ipool[0] if isinstance(ipool,(list,tuple)) else ipool
                def _rt(x):
                    a=x.float().cpu().numpy(); b,s,sh=quantize_int8(a); return torch.from_numpy(dequantize_int8(b,s,sh).copy())
                cvp.append(_rt(cp)); cvd.append(torch.stack([_rt(d) for d in cd]))
                cip.append(_rt(ipool)); cid.append(torch.stack([_rt(d) for d in io.deepstack_features]))
                c_ids.append(tr["input_ids"]); c_lab.append(tr["labels"]); c_attn.append(tr["attention_mask"])
                c_mm.append(tr["mm_token_type_ids"]); c_vg.append(tr["video_grid_thw"]); c_ig.append(ig)
            maxl = max(t.shape[0] for t in c_ids)
            def _pad(lst, fill):
                return torch.stack([torch.cat([t, torch.full((maxl-t.shape[0],), fill, dtype=t.dtype, device=t.device)]) if t.shape[0]<maxl else t for t in lst])
            batches.append({
                "input_ids": _pad(c_ids,0), "labels": _pad(c_lab,-100), "attention_mask": _pad(c_attn,0),
                "mm_token_type_ids": _pad(c_mm,0),
                "video_grid_thw": torch.cat(c_vg,0), "image_grid_thw": torch.cat(c_ig,0),
                "cached_video_pooler": torch.stack(cvp).to(device),
                "cached_video_deepstack": torch.stack(cvd).to(device),
                "cached_image_pooler": torch.stack(cip).to(device),
                "cached_image_deepstack": torch.stack(cid).to(device),
            })
        fwd = lambda b: forward_with_cached_vision_tokens(model, b, vt, it)
    else:  # live
        batches = []
        for j in range(nb):
            items = [live_ds[(j * bs + k) % len(live_ds)] for k in range(bs)]
            lb = collate_fn(items)
            batches.append({k: (v.to(device) if torch.is_tensor(v) else v) for k, v in lb.items()})
        fwd = lambda b: forward_with_video_compression_free(model, dict(b), vt, "fastervlm", 4, 4)

    sec, mem = _time_loop(batches, fwd, steps, backward, opt)
    print(f"RESULT arm={arm} sec={sec:.6f} mem={mem:.3f}", flush=True)


# ----------------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------------
def _spawn(arm, lance_path, steps, bs):
    env = dict(os.environ)
    env.setdefault("HF_HOME", "/workspace/.hf_home")
    env.pop("HF_HUB_OFFLINE", None)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    cmd = [sys.executable, os.path.abspath(__file__), "--_arm", arm,
           "--lance", lance_path, "--steps", str(steps), "--batch-size", str(bs)]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    out = p.stdout + p.stderr
    m = re.search(r"RESULT arm=\S+ sec=([\d.]+) mem=([\d.]+)", out)
    if m:
        return float(m.group(1)), float(m.group(2)), out
    oom = "OutOfMemoryError" in out
    return (None, None, out) if not oom else ("OOM", None, out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lance", default=os.path.join(_HERE, "nusc_3cam_native.lance"))
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--_arm", default=None)
    args = ap.parse_args()

    if args._arm:
        run_arm(args._arm, args.lance, args.steps, args.batch_size)
        return 0

    print(f"=== Tier-2 A/B training-step bench | steps={args.steps} bs={args.batch_size} ===")
    summary = {}

    print("\n[NATIVE 700 tok/cam] (1) FULL fwd+bwd+opt A/B (SGD, grad-ckpt, single GPU):")
    sec, mem, out = _spawn("live_native_train", args.lance, args.steps, args.batch_size)
    secc, memc, outc = _spawn("cached_native_train", args.lance, args.steps, args.batch_size)
    live_native_oom = (sec == "OOM" or sec is None)
    if live_native_oom:
        print("  LIVE native fwd+bwd+opt: OOM on single 32G GPU.")
        print("  -> cached path removes the ViT forward AND its activations from the per-step graph.")
    else:
        print(f"  LIVE   native fwd+bwd+opt: {sec:.4f}s/step  peak={mem:.1f}GB")
    if isinstance(secc, float):
        print(f"  CACHED native fwd+bwd+opt: {secc:.4f}s/step  peak={memc:.1f}GB")
    if isinstance(sec, float) and isinstance(secc, float):
        summary["native_train"] = (sec, secc, sec / secc)
        print(f"  NATIVE full-train speedup = {sec/secc:.3f}x")

    print("\n[NATIVE 700 tok/cam] (2) FORWARD-ONLY A/B (isolates ViT-skip):")
    fa, mfa, _ = _spawn("live_native_fwd", args.lance, args.steps, args.batch_size)
    fb, mfb, _ = _spawn("cached_native_fwd", args.lance, args.steps, args.batch_size)
    if isinstance(fa, float) and isinstance(fb, float):
        summary["native_fwd"] = (fa, fb, fa / fb)
        print(f"  LIVE   native fwd-only: {fa:.4f}s/step  peak={mfa:.1f}GB")
        print(f"  CACHED native fwd-only: {fb:.4f}s/step  peak={mfb:.1f}GB")
        print(f"  NATIVE forward-only speedup = {fa/fb:.3f}x")
    else:
        print(f"  native fwd A={fa} B={fb} (see logs)")

    print("\n[REDUCED 240 tok/cam @ 524288] both-arms-fit full fwd+bwd+opt A/B:")
    ra, mra, _ = _spawn("live_reduced_train", args.lance, args.steps, args.batch_size)
    rb, mrb, _ = _spawn("cached_reduced_train", args.lance, args.steps, args.batch_size)
    if isinstance(ra, float) and isinstance(rb, float):
        summary["reduced_train"] = (ra, rb, ra / rb)
        print(f"  LIVE   reduced fwd+bwd+opt: {ra:.4f}s/step  peak={mra:.1f}GB")
        print(f"  CACHED reduced fwd+bwd+opt: {rb:.4f}s/step  peak={mrb:.1f}GB")
        print(f"  REDUCED full-train speedup = {ra/rb:.3f}x")
    else:
        print(f"  reduced train A={ra} B={rb} (see logs)")

    print("\n=== SUMMARY ===")
    print(f"live_native_train_OOM = {live_native_oom}")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
