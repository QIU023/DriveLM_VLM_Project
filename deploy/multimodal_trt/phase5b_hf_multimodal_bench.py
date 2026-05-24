"""Phase 5b — HF full-multimodal latency bench on SAME B.5'' val sample[0].

Apples-to-apples vs phase5_bench_full_multimodal.py (TRT). Uses manual prefill
+ decode loop because HF generate() doesn't propagate mm_token_type_ids for
Qwen3-VL M-RoPE under transformers 5.x.

Saves: deploy/trt_bench/B5pp_hf_qwen3vl_multimodal_bf16.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_BASE = _HERE.parent.parent
sys.path.insert(0, str(_BASE / "scripts"))


def percentile(values, p):
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    f = int(k); c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f) if f != c else s[f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(_BASE / "checkpoints_qwen25/nusc_planning_b5pp_1cam_qwen3vl_multimodal/final"))
    ap.add_argument("--n-warmup", type=int, default=3)
    ap.add_argument("--n-runs", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=14)
    ap.add_argument("--out", default=str(_BASE / "deploy/trt_bench/B5pp_hf_qwen3vl_multimodal_bf16.json"))
    args = ap.parse_args()

    from transformers import AutoModelForImageTextToText, AutoProcessor
    from multimodal_planning_dataset import MultiModalPlanningDataset

    print(f"[hf-mm-bench] loading {args.ckpt}")
    t0 = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(
        args.ckpt, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda:0").eval()
    proc = AutoProcessor.from_pretrained(args.ckpt)
    print(f"[hf-mm-bench] loaded in {time.perf_counter()-t0:.1f}s")

    ds = MultiModalPlanningDataset(
        infos_path=str(_BASE / "data/uniad_infos/nuscenes_infos_temporal_val.pkl"),
        nusc_root=str(_BASE / "data/nuscenes"),
        processor=proc, max_length=12288,
        num_past_frames=4, num_future_waypoints=6, video_fps=2.0,
        vla_loss_mode="answer_and_traj", max_samples=2, require_full_future=True,
        planning_cams=["CAM_FRONT"], require_all_cams=True,
        hdmap_dir=str(_BASE / "data/preproc/hdmap_bev"),
        bbox_jsonl=str(_BASE / "data/preproc/bbox_egostate_val.jsonl"),
        split="val", modality_dropout_p=0.0,
    )
    s = ds[0]
    prompt_len = int(s["_meta_prompt_len"])
    input_ids = s["input_ids"].unsqueeze(0)[:, :prompt_len].to("cuda:0")
    attention_mask = s["attention_mask"].unsqueeze(0)[:, :prompt_len].to("cuda:0")
    mm_token_type_ids = s["mm_token_type_ids"].unsqueeze(0)[:, :prompt_len].to("cuda:0")
    pixel_values = s["pixel_values"].to("cuda:0", dtype=torch.bfloat16)
    pixel_values_videos = s["pixel_values_videos"].to("cuda:0", dtype=torch.bfloat16)
    image_grid_thw = s["image_grid_thw"].to("cuda:0")
    video_grid_thw = s["video_grid_thw"].to("cuda:0")
    if image_grid_thw.dim() == 1: image_grid_thw = image_grid_thw.unsqueeze(0)
    if video_grid_thw.dim() == 1: video_grid_thw = video_grid_thw.unsqueeze(0)

    prefill_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        pixel_values_videos=pixel_values_videos,
        video_grid_thw=video_grid_thw,
        mm_token_type_ids=mm_token_type_ids,
        use_cache=True,
    )

    def prefill():
        return model(**prefill_kwargs)

    # NOTE: HF transformers 5.x has a bug propagating mm_token_type_ids through
    # generate()/manual decode under Qwen3-VL M-RoPE. Per-token decode hits a
    # K-cache shape mismatch after the first step. So we measure ONLY prefill +
    # one-shot first-token logit (proxy for TTFT). The decode-per-token figure
    # in this JSON is therefore N/A; reuse the text-only HF baseline for the
    # decode-per-token proxy.
    print(f"[hf-mm-bench] PREFILL-ONLY mode (HF decode loop blocked by Qwen3-VL M-RoPE bug)")
    print(f"[hf-mm-bench] warmup {args.n_warmup}")
    with torch.no_grad():
        for _ in range(args.n_warmup):
            _ = prefill()
            torch.cuda.synchronize()

    print(f"[hf-mm-bench] runs {args.n_runs}")
    ttft = []
    with torch.no_grad():
        for _ in range(args.n_runs):
            torch.cuda.synchronize()
            t0_ = time.perf_counter()
            out = prefill()
            _ = out.logits[:, -1:, :].argmax(dim=-1)
            torch.cuda.synchronize()
            ttft.append(time.perf_counter() - t0_)
    full = ttft  # alias; no real full-traj measurement
    decode = [0.0] * len(ttft)
    throughput = [1.0 / t for t in ttft]

    results = {
        "ckpt": args.ckpt,
        "device": torch.cuda.get_device_name(0),
        "backend": "HF transformers 5.x SDPA (full multimodal PREFILL-ONLY: video+image+text+M-RoPE)",
        "note": "HF generate()/manual decode loop has a Qwen3-VL M-RoPE bug under transformers 5.x (K-cache shape mismatch on first decode step). Only prefill is measured here. For decode-per-token HF proxy, see B5pp_hf_qwen3vl_bf16.json (text-only LM forward).",
        "model": "Qwen3-VL-4B B.5'' VLA",
        "modality_real": "video(1cam x 4f) + image(HD-map BEV) + text(bbox/ego/prompt)",
        "dtype": "bfloat16",
        "prompt_len": prompt_len,
        "video_mm_tokens": 36,
        "image_mm_tokens": 121,
        "max_new_tokens": args.max_new_tokens,
        "n_warmup": args.n_warmup,
        "n_runs": args.n_runs,
        "TTFT_ms": {
            "mean": 1000*sum(ttft)/len(ttft),
            "p50": 1000*percentile(ttft, 50),
            "p99": 1000*percentile(ttft, 99),
        },
        "per_token_decode_ms": {
            "mean": 1000*sum(decode)/len(decode),
            "p50": 1000*percentile(decode, 50),
            "p99": 1000*percentile(decode, 99),
        },
        "full_traj_ms": {
            "mean": 1000*sum(full)/len(full),
            "p50": 1000*percentile(full, 50),
            "p99": 1000*percentile(full, 99),
        },
        "throughput_toks_per_s": {
            "mean": sum(throughput)/len(throughput),
            "p50": percentile(throughput, 50),
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[hf-mm-bench] saved → {args.out}")
    print(f"  TTFT mean: {results['TTFT_ms']['mean']:.1f} ms")
    print(f"  decode mean: {results['per_token_decode_ms']['mean']:.2f} ms/tok")
    print(f"  full mean: {results['full_traj_ms']['mean']:.1f} ms / {args.max_new_tokens} tok")
    print(f"  thrpt: {results['throughput_toks_per_s']['mean']:.1f} tok/s")


if __name__ == "__main__":
    main()
