"""Audit B.5' (3-cam Qwen2.5-VL multimodal) actual token counts per sample.

Goal: measure what fraction of train+val samples would silently truncate
under max_length=12288, so we can decide if B.5' results are valid or need retraining.

Approach: build the same chat-template content as MultiModalPlanningDataset,
tokenize text portion, add fixed visual expansion (3 × 3648 cam + 64 HD-map),
report distribution.

NO model load. NO actual image/video load. Pure tokenization.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List

BASE = Path("/workspace/DriveLM_VLM_Project")
sys.path.insert(0, str(BASE / "scripts"))

# Visual budget (fixed for native nuScenes 1600x900 + 4 frames + Qwen2.5-VL patch=14)
VIDEO_TOK_PER_CAM = 3648   # video_grid_thw [2, 64, 114] / 4 (spatial merge)
HDMAP_TOK = 64             # 224x224 → 16x16 / 4 = 64
VIDEO_PAD_TOKEN = "<|video_pad|>"
IMAGE_PAD_TOKEN = "<|image_pad|>"

CAMS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"]


def load_bbox_lookup(jsonl_path: Path) -> Dict[str, str]:
    """Build sample_token -> bbox_text dict."""
    d = {}
    with open(jsonl_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
                d[rec["sample_token"]] = rec.get("bbox_text", "")
            except Exception:
                continue
    return d


def format_ego_speed(info: dict) -> str:
    # Match scripts/multimodal_planning_dataset._format_ego_speed_preamble
    can_bus = info.get("can_bus", None)
    if can_bus is None:
        speed = 0.0
    else:
        try:
            import numpy as np
            vx, vy = float(can_bus[13]), float(can_bus[14])
            speed = float(np.hypot(vx, vy))
        except Exception:
            speed = 0.0
    return f"Ego speed at current frame: {speed:.2f} m/s\n"


def planning_prompt() -> str:
    return (
        "Given the above observations, predict the next 6 ego waypoints "
        "as <traj_start>{12 bin tokens}<traj_end>."
    )


def build_multicam_content(bbox_text: str, ego_text: str) -> list:
    """Mirror _build_user_content_multimodal for 3-cam path."""
    trailing = (bbox_text.rstrip("\n") + "\n\n" + ego_text + planning_prompt())
    CAM_LABELS = {
        "CAM_FRONT": "Camera FRONT",
        "CAM_FRONT_LEFT": "Camera FRONT_LEFT",
        "CAM_FRONT_RIGHT": "Camera FRONT_RIGHT",
    }
    content: list = []
    for cam in CAMS:
        content.append({"type": "text", "text": f"{CAM_LABELS[cam]}: "})
        content.append({"type": "video"})
        content.append({"type": "text", "text": " "})
    content.append({"type": "image"})       # HD-map BEV
    content.append({"type": "text", "text": trailing})
    return content


def count_one(processor, tokenizer, info: dict, bbox_lookup: Dict[str, str]):
    sample_token = info["token"]
    bbox_text = bbox_lookup.get(sample_token, "Detected objects in ego frame:\nnone")
    if not bbox_text.strip():
        bbox_text = "Detected objects in ego frame:\nnone"
    ego = format_ego_speed(info)
    user_content = build_multicam_content(bbox_text, ego)
    proc_messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": "Predicted trajectory:"},
    ]
    # Render the chat template as text (no tokenize), then tokenize.
    rendered = processor.apply_chat_template(
        proc_messages, tokenize=False, add_generation_prompt=False
    )
    # Tokenize raw rendered text — includes ONE <|video_pad|> per video and
    # ONE <|image_pad|> per image placeholder. We replace those with the actual
    # post-merger token count expansion.
    ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    n_video_pad = ids.count(tokenizer.convert_tokens_to_ids(VIDEO_PAD_TOKEN))
    n_image_pad = ids.count(tokenizer.convert_tokens_to_ids(IMAGE_PAD_TOKEN))
    base_text_tokens = len(ids)
    # Expansion: each video_pad placeholder expands to VIDEO_TOK_PER_CAM tokens
    # (replacing the 1-token placeholder). Same for image_pad.
    video_expansion = n_video_pad * (VIDEO_TOK_PER_CAM - 1)
    image_expansion = n_image_pad * (HDMAP_TOK - 1)
    total = base_text_tokens + video_expansion + image_expansion
    return {
        "total": total,
        "text_only": base_text_tokens - n_video_pad - n_image_pad,
        "video_tok": VIDEO_TOK_PER_CAM * n_video_pad,
        "image_tok": HDMAP_TOK * n_image_pad,
        "n_video_pad": n_video_pad,
        "n_image_pad": n_image_pad,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/workspace/DriveLM_VLM_Project/checkpoints_qwen25/nusc_planning_b5prime_3cam_multimodal/final")
    ap.add_argument("--infos-train", default="/workspace/DriveLM_VLM_Project/data/uniad_infos/nuscenes_infos_temporal_train.pkl")
    ap.add_argument("--infos-val", default="/workspace/DriveLM_VLM_Project/data/uniad_infos/nuscenes_infos_temporal_val.pkl")
    ap.add_argument("--bbox-train", default="/workspace/DriveLM_VLM_Project/data/preproc/bbox_egostate_train.jsonl")
    ap.add_argument("--bbox-val", default="/workspace/DriveLM_VLM_Project/data/preproc/bbox_egostate_val.jsonl")
    ap.add_argument("--max-length", type=int, default=12288, help="threshold to check truncation against")
    ap.add_argument("--limit", type=int, default=0, help="cap N samples per split for quick test (0=all)")
    ap.add_argument("--out", default="/workspace/DriveLM_VLM_Project/docs/B5prime_token_audit.md")
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoProcessor
    print(f"loading processor + tokenizer from {args.ckpt}", flush=True)
    processor = AutoProcessor.from_pretrained(args.ckpt, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt, trust_remote_code=True)

    results = {}
    for split, infos_path, bbox_path in [
        ("train", args.infos_train, args.bbox_train),
        ("val",   args.infos_val,   args.bbox_val),
    ]:
        print(f"\n=== {split} ===")
        print(f"loading infos: {infos_path}")
        data = pickle.load(open(infos_path, "rb"))
        infos = data["infos"] if isinstance(data, dict) and "infos" in data else data
        bbox_lookup = load_bbox_lookup(Path(bbox_path))
        print(f"  n_infos={len(infos)}  n_bbox_keys={len(bbox_lookup)}")

        n_target = len(infos) if args.limit == 0 else min(args.limit, len(infos))
        t0 = time.time()
        totals: List[int] = []
        n_over = 0
        worst = []
        for i in range(n_target):
            try:
                c = count_one(processor, tokenizer, infos[i], bbox_lookup)
            except Exception as e:
                if len(worst) < 3:
                    print(f"  err at i={i}: {e}")
                continue
            totals.append(c["total"])
            if c["total"] > args.max_length:
                n_over += 1
                if len(worst) < 10:
                    worst.append((i, c["total"], c["text_only"], infos[i]["token"]))
            if (i + 1) % 2000 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (n_target - i - 1) / rate
                print(f"  {i+1}/{n_target}  rate={rate:.0f}/s  ETA={eta:.0f}s  over={n_over}")
        elapsed = time.time() - t0
        print(f"  DONE {n_target} samples in {elapsed:.1f}s")

        import statistics
        totals.sort()
        n = len(totals)
        p50 = totals[n // 2]
        p90 = totals[int(n * 0.9)]
        p99 = totals[int(n * 0.99)]
        mx = totals[-1]
        mn = totals[0]
        results[split] = {
            "n": n,
            "min": mn, "p50": p50, "p90": p90, "p99": p99, "max": mx,
            "n_over": n_over,
            "pct_over": 100.0 * n_over / max(n, 1),
            "max_length_threshold": args.max_length,
            "worst": worst,
        }
        print(f"  min={mn} p50={p50} p90={p90} p99={p99} max={mx}")
        print(f"  over max_length={args.max_length}: {n_over}/{n} = {100.0 * n_over / max(n, 1):.2f}%")
        if worst:
            print(f"  worst (i, total, text, token):")
            for w in worst:
                print(f"    i={w[0]}  total={w[1]}  text={w[2]}  sample={w[3][:30]}")

    # Write report
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("# B.5' Token Audit (3-cam Qwen2.5-VL × multimodal)\n\n")
        f.write(f"**Threshold**: max_length = **{args.max_length}**\n\n")
        f.write(f"**Method**: tokenize chat-template text + add fixed visual expansion (3 cams × {VIDEO_TOK_PER_CAM} + 1 HD-map × {HDMAP_TOK} = {3 * VIDEO_TOK_PER_CAM + HDMAP_TOK} visual tokens). Text varies per sample (bbox count + ego speed). No model load, no pixel I/O.\n\n")
        for split, r in results.items():
            f.write(f"## {split}\n\n")
            f.write(f"- n = {r['n']}\n")
            f.write(f"- min/p50/p90/p99/max = {r['min']} / {r['p50']} / {r['p90']} / {r['p99']} / {r['max']}\n")
            f.write(f"- **samples exceeding {args.max_length}: {r['n_over']} ({r['pct_over']:.2f}%)**\n")
            if r['worst']:
                f.write(f"- worst (first 10): {[(w[0], w[1]) for w in r['worst']]}\n")
            f.write("\n")
        f.write("\n## Decision rule\n")
        f.write("- 0% over → B.5' results valid\n")
        f.write("- <1% over → essentially valid, add caveat\n")
        f.write("- 1-10% over → noisy, trend valid, absolute may be optimistic\n")
        f.write("- >10% over → MUST RETRAIN at higher max_length\n")
    print(f"\nreport written: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
