#!/usr/bin/env /venv/trt_llm/bin/python
"""Measured FLOPs / BOPs + roofline for the B.5'' 1-cam deploy pipeline.

Uses torch.profiler(with_flops=True) on a REAL multimodal sample to count the
FLOPs actually executed by (1) the ViT vision tower, (2) one LM prefill over the
full prompt, (3) one LM decode step. FLOPs are precision-INDEPENDENT (same MAC
count for bf16/fp8/fp4); we report BOPs = FLOPs x bit-width and the bs=1 decode
roofline (memory-bandwidth bound) to explain why fp8/fp4 help via bytes, not FLOPs.

Run on ONE GPU: CUDA_VISIBLE_DEVICES=N /venv/trt_llm/bin/python profile_flops.py
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from _common import apply_venv_shims
apply_venv_shims()
from _common import DEFAULT_CKPT, DEFAULT_CONFIG_YAML, build_calib_dataset, load_hf_model_bf16

import torch
from torch.profiler import profile, ProfilerActivity

DEV = "cuda:0"
BW_GBs = 1792.0          # RTX 5090 GDDR7 ~1.79 TB/s
TFLOPS = {"bf16": 210.0, "fp8": 419.0, "fp4": 838.0}   # Blackwell dense, approx
BYTES  = {"bf16": 2.0,  "fp8": 1.0,   "fp4": 0.5}


def _flops_of(prof) -> float:
    tot = 0
    for e in prof.key_averages():
        f = getattr(e, "flops", 0) or 0
        if f > 0:
            tot += f
    return tot


def main() -> int:
    print("[flops] loading bf16 model ...")
    model, processor = load_hf_model_bf16(DEFAULT_CKPT, device=DEV)
    model.eval()
    ds = build_calib_dataset(processor=processor, n_samples=1, split="val",
                             config_yaml=DEFAULT_CONFIG_YAML)
    s = ds[0]
    batch = {}
    for k, v in s.items():
        if k.startswith("_meta_") or not isinstance(v, torch.Tensor):
            continue
        if k in ("input_ids", "attention_mask", "labels", "mm_token_type_ids"):
            v = v.unsqueeze(0)
        if k in ("image_grid_thw", "video_grid_thw") and v.ndim == 1:
            v = v.unsqueeze(0)
        batch[k] = v.to(DEV, dtype=torch.bfloat16) if v.dtype.is_floating_point else v.to(DEV)
    batch.pop("labels", None)
    n_prompt = int(batch["input_ids"].shape[-1])
    print(f"[flops] real sample: prompt_len={n_prompt} tokens")

    lm_params = sum(p.numel() for p in model.model.language_model.parameters())
    print(f"[flops] LM params = {lm_params/1e9:.2f}B")

    # ---- full forward (ViT + LM prefill) with FLOP counting ----
    print("[flops] profiling full forward (ViT + prefill) ...")
    with torch.inference_mode():
        _ = model(**batch)   # warm
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     with_flops=True) as prof:
            _ = model(**batch)
            torch.cuda.synchronize()
    fwd_flops = _flops_of(prof)

    # ---- decode step (1 token, with cache) ----
    print("[flops] profiling 1 decode step ...")
    with torch.inference_mode():
        out = model(**batch, use_cache=True)
        pkv = out.past_key_values
        nxt = out.logits[:, -1:].argmax(-1)
        dec_in = {"input_ids": nxt, "past_key_values": pkv, "use_cache": True}
        _ = model(**dec_in)   # warm
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     with_flops=True) as prof_d:
            _ = model(**dec_in)
            torch.cuda.synchronize()
    dec_flops = _flops_of(prof_d)

    vit_plus_prefill_T = fwd_flops / 1e12
    dec_T = dec_flops / 1e12
    print("\n==== MEASURED FLOPs (torch.profiler with_flops) ====")
    print(f"full forward (ViT + {n_prompt}-tok prefill): {vit_plus_prefill_T:.2f} TFLOP")
    print(f"1 decode step:                               {dec_T*1000:.2f} GFLOP")
    print(f"14-tok trajectory decode:                    {dec_T*14*1000:.1f} GFLOP")

    print("\n==== BOPs per precision (= FLOPs x bit-width); FLOPs identical ====")
    for p in ("bf16", "fp8", "fp4"):
        bits = {"bf16":16,"fp8":8,"fp4":4}[p]
        print(f"{p}: full-fwd BOPs = {vit_plus_prefill_T*bits:.1f} TBOP "
              f"(x{bits}b)  decode/14tok BOPs = {dec_T*14*bits*1000:.1f} GBOP")

    print("\n==== bs=1 decode roofline (RTX 5090 BW=1.79TB/s, ridge bf16=117 FLOP/B) ====")
    dec_flop_tok = dec_flops
    for p in ("bf16","fp8","fp4"):
        wbytes = lm_params * BYTES[p]
        t_mem = wbytes / (BW_GBs*1e9) * 1000
        t_cmp = (dec_flop_tok/1e12) / TFLOPS[p] * 1000
        ai = dec_flop_tok / wbytes
        bound = "MEMORY" if t_mem > t_cmp else "COMPUTE"
        print(f"{p}: w={wbytes/1e9:.1f}GB/tok AI={ai:.1f}FLOP/B t_mem={t_mem:.2f}ms "
              f"t_compute={t_cmp:.3f}ms -> {bound}-bound  14tok~{t_mem*14:.0f}ms")

    out = {
        "prompt_len": n_prompt,
        "lm_params_B": lm_params/1e9,
        "measured_full_fwd_TFLOP": vit_plus_prefill_T,
        "measured_decode_step_GFLOP": dec_T*1000,
        "measured_14tok_decode_GFLOP": dec_T*14*1000,
        "note": "FLOPs precision-independent; BOPs=FLOPsxbits; decode bs=1 is bandwidth-bound",
    }
    Path(_HERE/"results").mkdir(exist_ok=True)
    json.dump(out, open(_HERE/"results"/"flops_profile.json","w"), indent=2)
    print(f"\n[flops] wrote {_HERE}/results/flops_profile.json")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
