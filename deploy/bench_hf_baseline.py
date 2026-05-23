#!/usr/bin/env python3
"""HF bf16 latency benchmark on B.5'' (Qwen3-VL-4B) ckpt — text-only path.

Apples-to-apples vs deploy/bench_trt_qwen3vl.py (TRT-LLM 1.3 PyTorch backend),
which also uses a text-only prompt via LLM.generate(). Same prompt, same ckpt,
same dtype, same hardware → honest TTFT / decode / throughput delta.

Why text-only: TRT-LLM 1.3 LLM API does not currently accept Qwen3-VL visual
inputs through generate(); both benches measure LM forward only. Visual prefill
is a small one-time cost outside the decode loop and doesn't dominate
latency for the 14-token trajectory output.

Run:
  /usr/bin/python3 deploy/bench_hf_baseline.py \\
    --ckpt checkpoints_qwen25/nusc_planning_b5pp_1cam_qwen3vl_multimodal/final \\
    --n-warmup 3 --n-runs 20 --out deploy/trt_bench/B5pp_hf_qwen3vl_bf16.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch


def percentile(values, p):
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


# Exact same prompt as deploy/bench_trt_qwen3vl.py
PROMPT = (
    "You are a self-driving system. Given the current scene context, "
    "predict the next 6 ego waypoints as <traj_start> bin tokens. "
    "Current scene: straight road, 12 m/s, no obstacles. <traj_start>"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-warmup", type=int, default=3)
    ap.add_argument("--n-runs", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=14)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"[bench] loading {args.ckpt} ...")
    t0 = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(
        args.ckpt, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(args.device).eval()
    processor = AutoProcessor.from_pretrained(args.ckpt)
    t_load = time.perf_counter() - t0
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[bench] loaded in {t_load:.1f}s; {n_params/1e9:.3f}B params")

    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    enc = tokenizer(PROMPT, return_tensors="pt")
    input_ids = enc["input_ids"].to(args.device)
    attention_mask = enc["attention_mask"].to(args.device)
    prompt_len = input_ids.shape[1]
    print(f"[bench] prompt_len={prompt_len}")

    # Use the LM submodule directly: text-only, no M-RoPE branch.
    if hasattr(model, "language_model"):
        lm = model.language_model
    elif hasattr(model, "model") and hasattr(model.model, "language_model"):
        lm = model.model.language_model
    else:
        lm = model
    lm = lm.eval()

    def prefill():
        return lm(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)

    def run_full(max_new):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = prefill()
        # last hidden -> logits
        if hasattr(out, "logits") and out.logits is not None:
            logits = out.logits[:, -1:, :]
        else:
            hidden = out.last_hidden_state[:, -1:, :]
            logits = model.lm_head(hidden) if hasattr(model, "lm_head") else hidden
        next_id = logits.argmax(dim=-1)
        torch.cuda.synchronize()
        ttft = time.perf_counter() - t0
        pkv = out.past_key_values
        am = attention_mask.clone()
        for _ in range(max_new - 1):
            am = torch.cat([am, torch.ones((1, 1), dtype=am.dtype, device=am.device)], dim=1)
            out = lm(input_ids=next_id, attention_mask=am, past_key_values=pkv, use_cache=True)
            if hasattr(out, "logits") and out.logits is not None:
                logits = out.logits[:, -1:, :]
            else:
                hidden = out.last_hidden_state[:, -1:, :]
                logits = model.lm_head(hidden) if hasattr(model, "lm_head") else hidden
            next_id = logits.argmax(dim=-1)
            pkv = out.past_key_values
        torch.cuda.synchronize()
        return ttft, time.perf_counter() - t0

    print(f"[bench] warmup {args.n_warmup} runs ...")
    with torch.no_grad():
        for _ in range(args.n_warmup):
            run_full(args.max_new_tokens)

    prefill_times = []
    with torch.no_grad():
        for _ in range(args.n_runs):
            torch.cuda.synchronize()
            t = time.perf_counter()
            _ = prefill()
            torch.cuda.synchronize()
            prefill_times.append(time.perf_counter() - t)

    ttft_times, full_times = [], []
    with torch.no_grad():
        for _ in range(args.n_runs):
            ttft, full = run_full(args.max_new_tokens)
            ttft_times.append(ttft)
            full_times.append(full)

    per_tok_decode = [(f - t) / (args.max_new_tokens - 1) for f, t in zip(full_times, ttft_times)]
    throughput = [args.max_new_tokens / f for f in full_times]

    results = {
        "ckpt": args.ckpt,
        "device": torch.cuda.get_device_name(0),
        "device_cap": list(torch.cuda.get_device_capability(0)),
        "dtype": "bfloat16",
        "backend": "HF transformers SDPA (LM forward, text-only — apples-to-apples vs TRT bench)",
        "params_B": n_params / 1e9,
        "prompt": PROMPT,
        "prompt_tokens": int(prompt_len),
        "max_new_tokens": args.max_new_tokens,
        "n_warmup": args.n_warmup,
        "n_runs": args.n_runs,
        "load_seconds": t_load,
        "prefill_ms": {
            "mean": 1000 * sum(prefill_times) / len(prefill_times),
            "p50": 1000 * percentile(prefill_times, 50),
            "p99": 1000 * percentile(prefill_times, 99),
        },
        "TTFT_ms": {
            "mean": 1000 * sum(ttft_times) / len(ttft_times),
            "p50": 1000 * percentile(ttft_times, 50),
            "p99": 1000 * percentile(ttft_times, 99),
        },
        "per_token_decode_ms": {
            "mean": 1000 * sum(per_tok_decode) / len(per_tok_decode),
            "p50": 1000 * percentile(per_tok_decode, 50),
            "p99": 1000 * percentile(per_tok_decode, 99),
        },
        "full_traj_ms": {
            "mean": 1000 * sum(full_times) / len(full_times),
            "p50": 1000 * percentile(full_times, 50),
            "p99": 1000 * percentile(full_times, 99),
        },
        "throughput_toks_per_s": {
            "mean": sum(throughput) / len(throughput),
            "p50": percentile(throughput, 50),
        },
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[bench] saved → {args.out}")
    print(f"  prefill mean: {results['prefill_ms']['mean']:.1f} ms (p50 {results['prefill_ms']['p50']:.1f} / p99 {results['prefill_ms']['p99']:.1f})")
    print(f"  TTFT    mean: {results['TTFT_ms']['mean']:.1f} ms")
    print(f"  decode  mean: {results['per_token_decode_ms']['mean']:.2f} ms / token")
    print(f"  full    mean: {results['full_traj_ms']['mean']:.1f} ms / {args.max_new_tokens} tokens")
    print(f"  thrpt   mean: {results['throughput_toks_per_s']['mean']:.1f} tok/s")


if __name__ == "__main__":
    main()
