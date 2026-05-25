#!/usr/bin/env /venv/trt_llm/bin/python
"""B.5''' TRT-LLM 1.3.0rc15 PyTorch-backend engine load + smoke probe.

TRT-LLM 1.3 PyTorch backend is JIT — it does NOT serialize an engine file to
disk. This script's job is:
  1. Construct `LLM(model=ckpt, dtype=..., tensor_parallel_size=1, ...)` with
     the right kwargs for {bf16, fp8, nvfp4}.
  2. Write a small JSON wrapper config to engines/<name>/build_config.json so
     downstream bench/deploy scripts know which ckpt + precision to load.
  3. Do a load-smoke: `llm.generate("Hello", max_tokens=1)` to verify the
     PTQ ckpt is consumable by TRT-LLM (quant scheme is auto-detected from the
     modelopt-written hf_quant_config.json / safetensors metadata).

Usage:
    /venv/trt_llm/bin/python build_engine.py \
        --ckpt /path/to/{final,quant_fp8,quant_nvfp4} \
        --precision {bf16,fp8,nvfp4} \
        --out engines/b5ppp_bf16 \
        [--max-seq-len 12288] \
        [--max-batch-size 8] \
        [--free-gpu-mem-frac 0.6]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from _common import apply_venv_shims  # noqa: E402
apply_venv_shims()

from _common import (  # noqa: E402
    DEFAULT_CKPT,
    DEFAULT_ENGINE_OUT_DIR,
    DEFAULT_PARENT,
    peak_mem_gb,
    reset_peak_mem,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="B.5''' TRT-LLM PyTorch backend load probe (no on-disk engine — JIT)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt", default=DEFAULT_CKPT,
                   help="HF ckpt (bf16 final/) or quantized (quant_fp8/, quant_nvfp4/)")
    p.add_argument("--precision", choices=["bf16", "fp8", "nvfp4"], required=True)
    p.add_argument("--out", default=None,
                   help="Output dir for wrapper config (defaults to engines/<auto-name>)")
    p.add_argument("--max-seq-len", type=int, default=12288,
                   help="Max prompt+generation length (matches training max_length)")
    p.add_argument("--max-batch-size", type=int, default=8)
    p.add_argument("--max-num-tokens", type=int, default=12288)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--free-gpu-mem-frac", type=float, default=0.6,
                   help="KvCacheConfig.free_gpu_memory_fraction")
    p.add_argument("--smoke-prompt", default="You are a self-driving planner. Answer:",
                   help="Text-only prompt for the load smoke")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    src = Path(args.ckpt)
    if not src.is_dir():
        print(f"[build] FATAL: ckpt not found: {src}", file=sys.stderr)
        return 2

    if args.out is None:
        # Auto-name from ckpt basename + precision
        out_dir = Path(DEFAULT_ENGINE_OUT_DIR) / f"b5ppp_{args.precision}"
    else:
        out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[build] === STEP 1: validate ckpt + precision metadata ===")
    print(f"[build] ckpt = {src}")
    print(f"[build] precision = {args.precision}")
    print(f"[build] out = {out_dir}")

    # For fp8/nvfp4, modelopt writes a `hf_quant_config.json` next to the model
    # safetensors. Surface it so we know TRT will detect the quantization.
    hf_quant_cfg = src / "hf_quant_config.json"
    if args.precision in ("fp8", "nvfp4"):
        if not hf_quant_cfg.is_file():
            print(f"[build] WARN: precision={args.precision} but no hf_quant_config.json "
                  f"at {hf_quant_cfg}. TRT-LLM may fall back to bf16 / fail to detect quant.",
                  file=sys.stderr)
        else:
            try:
                qcfg = json.load(open(hf_quant_cfg))
                print(f"[build] hf_quant_config.json: {json.dumps(qcfg, indent=2)[:600]}")
            except Exception as e:
                print(f"[build] couldn't parse hf_quant_config.json: {e}")
    elif args.precision == "bf16" and hf_quant_cfg.is_file():
        print(f"[build] WARN: precision=bf16 but ckpt has hf_quant_config.json — "
              f"this is a quantized ckpt, downgrade precision flag or pick a different "
              f"ckpt.", file=sys.stderr)

    print(f"[build] === STEP 2: construct LLM ===")
    import torch
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import KvCacheConfig

    # TRT-LLM 1.3.0rc15 PyTorch backend kwargs. dtype is passed via the
    # ckpt's config.json for bf16 — the LLM class infers from config. For
    # fp8/nvfp4 we rely on auto-detection from hf_quant_config.json (no
    # explicit quant_algo kwarg needed in 1.3.0rc15; that was the 0.x API).
    reset_peak_mem()
    t0 = time.perf_counter()
    llm_kwargs = dict(
        model=str(src),
        tensor_parallel_size=int(args.tensor_parallel_size),
        max_batch_size=int(args.max_batch_size),
        max_seq_len=int(args.max_seq_len),
        max_num_tokens=int(args.max_num_tokens),
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=float(args.free_gpu_mem_frac)),
        trust_remote_code=True,
    )
    print(f"[build] LLM kwargs: {{...max_seq_len={args.max_seq_len}, "
          f"max_batch_size={args.max_batch_size}, tp={args.tensor_parallel_size}}}")
    llm = LLM(**llm_kwargs)
    load_secs = time.perf_counter() - t0
    load_peak_gb = peak_mem_gb()
    print(f"[build] LLM loaded in {load_secs:.1f}s, peak GPU mem after load: {load_peak_gb:.2f} GB")

    print(f"[build] === STEP 3: text-only smoke generate (max_tokens=4) ===")
    reset_peak_mem()
    sp = SamplingParams(max_tokens=4, temperature=0.0)
    t0 = time.perf_counter()
    out = llm.generate([{"prompt": args.smoke_prompt}], sampling_params=sp)
    gen_secs = time.perf_counter() - t0
    gen_peak_gb = peak_mem_gb()
    try:
        smoke_tokens = list(out[0].outputs[0].token_ids)
        smoke_text = out[0].outputs[0].text
    except Exception as e:
        print(f"[build] FATAL: smoke generate returned unparseable output: {e}", file=sys.stderr)
        return 4
    print(f"[build] smoke generate OK in {gen_secs*1000:.1f}ms")
    print(f"[build]   generated token_ids: {smoke_tokens}")
    print(f"[build]   generated text: {smoke_text!r}")
    print(f"[build]   peak GPU mem (load+gen): {gen_peak_gb:.2f} GB")

    print(f"[build] === STEP 4: write wrapper config ===")
    wrapper = {
        "ckpt": str(src),
        "precision": args.precision,
        "tensor_parallel_size": int(args.tensor_parallel_size),
        "max_batch_size": int(args.max_batch_size),
        "max_seq_len": int(args.max_seq_len),
        "max_num_tokens": int(args.max_num_tokens),
        "free_gpu_mem_frac": float(args.free_gpu_mem_frac),
        "trt_llm_version": "1.3.0rc15",
        "backend": "pytorch",
        "load_secs": load_secs,
        "load_peak_gb": load_peak_gb,
        "smoke_gen_ms": gen_secs * 1000,
        "smoke_gen_peak_gb": gen_peak_gb,
        "smoke_tokens": smoke_tokens,
        "smoke_text": smoke_text,
        "hf_quant_config_present": hf_quant_cfg.is_file(),
        "note": "TRT-LLM 1.3.0rc15 PyTorch backend is JIT; no on-disk engine. "
                "This wrapper points bench/deploy scripts at the ckpt + precision combo.",
    }
    wrapper_path = out_dir / "build_config.json"
    with open(wrapper_path, "w") as f:
        json.dump(wrapper, f, indent=2)
    print(f"[build] wrapper config → {wrapper_path}")

    print(f"[build] === DONE ===")
    print(f"[build] Next: /venv/trt_llm/bin/python bench_trt.py "
          f"--ckpt {src} --precision {args.precision}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
