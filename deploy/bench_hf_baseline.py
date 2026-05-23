#!/usr/bin/env python3
"""HF bf16 latency benchmark on the B.5' VLA ckpt — serves as deployment-latency
**reference** for the TRT path we couldn't finish tonight.

Why this exists: TRT-LLM 1.2.1 does NOT support Qwen2.5-VL (model_type lookup
returns None); upstream issues #2794, #10069, #8404 track Qwen2.5-VL FP4 support.
We installed TRT-LLM in /opt/trt_venv but cannot complete the engine build for
this model architecture tonight. HF bf16 on the host gives an UPPER-BOUND latency
number against which a future TRT FP4 engine should be compared.

Measures:
  - prefill latency (model.forward only, no generate)
  - TTFT (time-to-first-token via generate with max_new_tokens=1)
  - per-token decode latency (median over N runs of generate(max_new_tokens=14))
  - total trajectory-token (14 tokens) generation time
  - throughput (tokens/sec)

Run:
  /usr/bin/python3 deploy/bench_hf_baseline.py \\
    --ckpt checkpoints_qwen25/nusc_planning_b5prime_3cam_multimodal/final \\
    --n-warmup 3 --n-runs 20 --out deploy/trt_bench/B5prime_hf_bf16.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image

_HERE = Path(__file__).resolve().parent
_BASE = _HERE.parent
sys.path.insert(0, str(_BASE / "scripts"))


def percentile(values, p):
    """numpy-free percentile."""
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-warmup", type=int, default=3)
    ap.add_argument("--n-runs", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=14,
                    help="trajectory token sequence length")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--use-sample-input", action="store_true", default=True,
                    help="construct a realistic 3-cam multimodal sample from val")
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

    # Build a representative input: simulate 3-cam × 4-frame video + HD-map image + bbox text
    from multimodal_planning_dataset import MultiModalPlanningDataset
    ds = MultiModalPlanningDataset(
        infos_path=os.path.join(_BASE, "data/uniad_infos/nuscenes_infos_temporal_val.pkl"),
        nusc_root=os.path.join(_BASE, "data/nuscenes"),
        processor=processor,
        max_length=12288, num_past_frames=4, num_future_waypoints=6,
        video_fps=2.0, vla_loss_mode="answer_and_traj",
        max_samples=5,
        require_full_future=True,
        planning_cams=["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"],
        require_all_cams=True,
        hdmap_dir=os.path.join(_BASE, "data/preproc/hdmap_bev"),
        bbox_jsonl=os.path.join(_BASE, "data/preproc/bbox_egostate_val.jsonl"),
        split="val", modality_dropout_p=0.0,
    )
    sample = ds[0]
    input_ids = sample["input_ids"].unsqueeze(0).to(args.device)
    attention_mask = sample["attention_mask"].unsqueeze(0).to(args.device)
    pixel_values_videos = sample["pixel_values_videos"].to(args.device, dtype=torch.bfloat16)
    video_grid_thw = sample["video_grid_thw"].to(args.device)
    pixel_values = sample["pixel_values"].to(args.device, dtype=torch.bfloat16)
    image_grid_thw = sample["image_grid_thw"].to(args.device)
    # MultiModalPlanningDataset returns image_grid_thw shape=(3,) for single image;
    # model.rot_pos_emb iterates assuming shape=(N, 3) → unsqueeze if 1-D.
    if image_grid_thw.dim() == 1:
        image_grid_thw = image_grid_thw.unsqueeze(0)
    second_per_grid_ts = sample["second_per_grid_ts"].to(args.device, dtype=torch.float32)
    # Truncate input_ids to remove trajectory tokens (we'll generate them)
    prompt_len = int(sample["_meta_prompt_len"])
    input_ids = input_ids[:, :prompt_len]
    attention_mask = attention_mask[:, :prompt_len]

    print(f"[bench] input_ids shape: {tuple(input_ids.shape)} (prompt_len={prompt_len})")
    print(f"[bench] video pixel shape: {tuple(pixel_values_videos.shape)}")

    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values_videos=pixel_values_videos,
        video_grid_thw=video_grid_thw,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        second_per_grid_ts=second_per_grid_ts,
        do_sample=False,
    )

    # Warmup
    print(f"[bench] warmup {args.n_warmup} runs ...")
    with torch.no_grad():
        for _ in range(args.n_warmup):
            _ = model.generate(**gen_kwargs, max_new_tokens=args.max_new_tokens)
    torch.cuda.synchronize()

    # 1. prefill latency (forward only, no decode)
    prefill_times = []
    with torch.no_grad():
        for _ in range(args.n_runs):
            torch.cuda.synchronize()
            t = time.perf_counter()
            _ = model(
                input_ids=input_ids, attention_mask=attention_mask,
                pixel_values_videos=pixel_values_videos, video_grid_thw=video_grid_thw,
                pixel_values=pixel_values, image_grid_thw=image_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
            )
            torch.cuda.synchronize()
            prefill_times.append(time.perf_counter() - t)

    # 2. TTFT (generate 1 token)
    ttft_times = []
    with torch.no_grad():
        for _ in range(args.n_runs):
            torch.cuda.synchronize()
            t = time.perf_counter()
            _ = model.generate(**gen_kwargs, max_new_tokens=1)
            torch.cuda.synchronize()
            ttft_times.append(time.perf_counter() - t)

    # 3. Full trajectory generation (14 tokens)
    full_times = []
    with torch.no_grad():
        for _ in range(args.n_runs):
            torch.cuda.synchronize()
            t = time.perf_counter()
            out = model.generate(**gen_kwargs, max_new_tokens=args.max_new_tokens)
            torch.cuda.synchronize()
            full_times.append(time.perf_counter() - t)
    # Per-token decode latency = (full - ttft) / (max_new_tokens - 1)
    per_tok_decode = [(f - t) / (args.max_new_tokens - 1) for f, t in zip(full_times, ttft_times)]
    throughput = [args.max_new_tokens / f for f in full_times]

    results = {
        "ckpt": args.ckpt,
        "device": torch.cuda.get_device_name(0),
        "device_cap": list(torch.cuda.get_device_capability(0)),
        "dtype": "bfloat16",
        "backend": "HF transformers (no TRT engine)",
        "params_B": n_params / 1e9,
        "input_tokens": int(input_ids.shape[1]),
        "video_tokens": int(pixel_values_videos.shape[0] // 4),  # post 2x2 merge
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
    print(f"  full    mean: {results['full_traj_ms']['mean']:.1f} ms / 14 tokens")
    print(f"  thrpt   mean: {results['throughput_toks_per_s']['mean']:.1f} tok/s")


if __name__ == "__main__":
    main()
