"""Open-loop trajectory evaluation.

Loads a (merged or LoRA) Qwen2.5-VL VLA checkpoint, runs it on the validation
split of the VLA dataset, decodes the generated trajectory tokens, and reports
L2 displacement error (ADE / FDE) against the ground-truth waypoints stored in
each record's `metadata.waypoints_xy_m`.

This is the standard nuScenes open-loop planning metric (see
OpenDriveLab/UniAD eval, or AutoVLA Table 3) restricted to the 3 s horizon we
use:

    ADE_t = mean_{t in [1..T]} || pred_xy[t] - gt_xy[t] ||_2
    FDE   = || pred_xy[-1] - gt_xy[-1] ||_2

This script is **prep only** — it is not run as part of this task because the
ground-truth waypoints currently sourced via the behavior-heuristic fallback
(see scripts/extract_ego_trajectory.py) are not real nuScenes ego poses and
would produce a meaningless number. Once nuScenes v1.0-trainval metadata is on
disk, re-run extract_ego_trajectory.py and then this eval.

Usage:
    python scripts/eval_traj.py \
        --config configs/gb200_vla.yaml \
        --data   data_processed/v1_1_video_n4_with_traj.json \
        --limit  100
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_lora import DriveLMDataset, collate_fn, load_config  # noqa: E402
from trajectory_tokenizer import (  # noqa: E402
    TrajectoryTokenizer,
    TrajectoryTokenizerConfig,
)


def _compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """ADE @ {1s, 2s, 3s} and FDE. Assumes 6 waypoints @ 2 Hz (1s=2, 2s=4, 3s=6)."""
    T = min(pred.shape[0], gt.shape[0])
    if T == 0:
        return {"ade_full": float("nan"), "fde": float("nan"), "ade_1s": float("nan"),
                "ade_2s": float("nan"), "ade_3s": float("nan"), "n": 0}
    err = np.linalg.norm(pred[:T] - gt[:T], axis=1)  # (T,)
    ade_full = err.mean()
    fde = err[-1]
    ade_1s = err[: min(2, T)].mean()
    ade_2s = err[: min(4, T)].mean()
    ade_3s = err[: min(6, T)].mean()
    return {"ade_full": float(ade_full), "fde": float(fde),
            "ade_1s": float(ade_1s), "ade_2s": float(ade_2s), "ade_3s": float(ade_3s), "n": int(T)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True, help="VLA val JSON (with traj waypoints)")
    ap.add_argument("--limit", type=int, default=50, help="Max samples to eval")
    ap.add_argument("--gen-tokens", type=int, default=24,
                    help="Max new tokens (>= 2*num_waypoints + 2)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if not cfg.get("vla_mode", False):
        print("ERROR: config is not in vla_mode.", file=sys.stderr)
        sys.exit(2)

    model_id = cfg["model_id"]
    num_frames = cfg.get("num_frames", 4)
    video_fps = cfg.get("video_fps", 2.0)
    max_length = cfg.get("max_length", 2560)
    dtype = getattr(torch, cfg.get("dtype", "bfloat16"))

    print(f"[eval-traj] model={model_id}")
    processor = AutoProcessor.from_pretrained(model_id)
    if hasattr(processor, "image_processor") and processor.image_processor is not None:
        processor.image_processor.min_pixels = cfg.get("min_pixels", 25088)
        processor.image_processor.max_pixels = cfg.get("max_pixels", 100352)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=dtype, device_map="auto", attn_implementation="sdpa",
    )
    model.eval()

    traj_cfg = TrajectoryTokenizerConfig()
    traj_tok = TrajectoryTokenizer(traj_cfg)

    data_path = args.data
    if not os.path.isabs(data_path):
        data_path = os.path.join(_BASE_DIR, data_path)
    ds = DriveLMDataset(
        data_path, processor, max_length=max_length,
        video_mode=True, num_frames=num_frames, video_fps=video_fps,
        vla_mode=True, traj_start_id=traj_cfg.traj_start_id, traj_end_id=traj_cfg.traj_end_id,
    )

    # We also need the raw records (for ground-truth waypoints).
    with open(data_path, "r") as f:
        raw_records = json.load(f)

    asst_marker = processor.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)

    all_metrics: List[dict] = []
    n = min(args.limit, len(ds))
    for i in range(n):
        try:
            sample = ds[i]
            batch = collate_fn([sample])
        except Exception as e:
            print(f"  sample {i}: skipping ({e})")
            continue

        full_ids = batch["input_ids"][0].tolist()
        cut = None
        for j in range(len(full_ids) - len(asst_marker) + 1):
            if full_ids[j : j + len(asst_marker)] == asst_marker:
                cut = j + len(asst_marker)
        if cut is None:
            print(f"  sample {i}: no assistant marker, skipping")
            continue

        prompt_ids = torch.tensor([full_ids[:cut]], device=model.device, dtype=batch["input_ids"].dtype)
        prompt_mask = torch.ones_like(prompt_ids)
        gen_inputs = {"input_ids": prompt_ids, "attention_mask": prompt_mask}
        for k in ("pixel_values_videos", "video_grid_thw", "second_per_grid_ts",
                  "pixel_values", "image_grid_thw"):
            if k in batch and isinstance(batch[k], torch.Tensor):
                gen_inputs[k] = batch[k].to(model.device,
                                            dtype=dtype if "pixel" in k else batch[k].dtype)
        with torch.no_grad():
            gen = model.generate(**gen_inputs, max_new_tokens=args.gen_tokens, do_sample=False)
        new_tokens = gen[0, prompt_ids.shape[1]:].tolist()
        pred_wp = traj_tok.decode(new_tokens)
        gt_wp = np.asarray(raw_records[i].get("metadata", {}).get("waypoints_xy_m", []),
                           dtype=np.float32)
        m = _compute_metrics(pred_wp, gt_wp)
        m["source"] = raw_records[i].get("metadata", {}).get("traj_source", "unknown")
        all_metrics.append(m)
        if i < 5:
            print(f"  sample {i}: pred[-1]={pred_wp[-1].tolist() if len(pred_wp) else 'EMPTY'} "
                  f"gt[-1]={gt_wp[-1].tolist() if len(gt_wp) else 'EMPTY'} "
                  f"ade={m['ade_full']:.2f} fde={m['fde']:.2f} src={m['source']}")

    if not all_metrics:
        print("No samples evaluated.")
        sys.exit(1)
    agg = {k: float(np.mean([m[k] for m in all_metrics if m["n"] > 0]))
           for k in ("ade_full", "fde", "ade_1s", "ade_2s", "ade_3s")}
    print()
    print(f"Aggregate over {len(all_metrics)} samples:")
    for k, v in agg.items():
        print(f"  {k:10s}: {v:.3f} m")
    print(f"  source histogram: " + str({s: sum(1 for m in all_metrics if m['source'] == s)
                                          for s in set(m['source'] for m in all_metrics)}))


if __name__ == "__main__":
    main()
