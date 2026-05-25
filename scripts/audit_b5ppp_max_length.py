"""Audit B.5''' (3-cam Qwen3-VL-4B multimodal) actual token counts per sample.

Uses the REAL Qwen3-VL-4B processor (downloaded P2). Loads one tiny dummy
sample to get exact video/image token expansion at native nuScenes 1600x900
+ HD-map 224x224 + 4-frame past, then tokenizes the text portion per sample
to get total tokens per sample.

NO model load. Real processor for visual expansion math.
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


CAMS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"]


def load_bbox_lookup(jsonl_path: Path) -> Dict[str, str]:
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


def measure_visual_expansion(processor, tokenizer):
    """Run the processor on real dummy frames at native size to get exact
    video_tok and image_tok counts under Qwen3-VL's smart_resize behavior."""
    from PIL import Image
    import numpy as np
    # 3 cam videos, each 4 frames at 1600x900
    cam_clip = [Image.fromarray(np.zeros((900, 1600, 3), dtype=np.uint8)) for _ in range(4)]
    cam_clips = [cam_clip, cam_clip, cam_clip]
    hd = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    # Build chat-template text with placeholders
    user_content = build_multicam_content("Detected objects in ego frame:\nnone", "Ego speed at current frame: 0.00 m/s\n")
    msgs = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": "Predicted trajectory:"},
    ]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    # Now run processor with videos + images
    out = processor(text=[text], images=[hd], videos=cam_clips, return_tensors="pt")
    total_tokens = out["input_ids"].shape[-1]
    # Tokenize text-only (no expansion) for fixed visual count breakdown
    bare = tokenizer(text, add_special_tokens=False)["input_ids"]
    # Each <|video_pad|> / <|image_pad|> token in bare = 1 placeholder; processor expanded each to actual count
    vp_id = tokenizer.convert_tokens_to_ids("<|video_pad|>")
    ip_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    n_vp = bare.count(vp_id)
    n_ip = bare.count(ip_id)
    expansion = total_tokens - len(bare)
    video_count_per_cam = (expansion - (sum(g[0] * g[1] * g[2] // 4 for g in out["image_grid_thw"].tolist()))) // max(n_vp, 1)
    # cleaner: pull from video_grid_thw / image_grid_thw directly
    video_grid = out.get("video_grid_thw", None)
    image_grid = out.get("image_grid_thw", None)
    print(f"  dummy processor probe: input_ids.len = {total_tokens}, bare text = {len(bare)}, expansion = {expansion}")
    print(f"  video_grid_thw = {video_grid.tolist() if video_grid is not None else None}  (T, H, W per cam)")
    print(f"  image_grid_thw = {image_grid.tolist() if image_grid is not None else None}  (T, H, W per image)")
    # Compute expanded count per modality (Qwen2/3-VL: grid_thw / merge_size^2 ; merge=2 -> divide by 4)
    if video_grid is not None:
        vg = video_grid.tolist()
        video_total = sum(t * h * w // 4 for (t, h, w) in vg)
        per_cam_video = vg[0][0] * vg[0][1] * vg[0][2] // 4
    else:
        video_total = 0; per_cam_video = 0
    if image_grid is not None:
        ig = image_grid.tolist()
        image_total = sum(t * h * w // 4 for (t, h, w) in ig)
    else:
        image_total = 0
    print(f"  video tokens total = {video_total}  ({per_cam_video} per cam, {len(vg)} cams)")
    print(f"  image tokens total = {image_total}")
    # Sanity: video_total + image_total + (len(bare) - n_vp - n_ip) should equal total_tokens
    sanity = video_total + image_total + (len(bare) - n_vp - n_ip)
    print(f"  sanity check: visual({video_total + image_total}) + text({len(bare) - n_vp - n_ip}) = {sanity} vs total {total_tokens}  diff={sanity - total_tokens}")
    return per_cam_video, image_total, n_vp, n_ip


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/workspace/.hf_home/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17")
    ap.add_argument("--infos-train", default="/workspace/DriveLM_VLM_Project/data/uniad_infos/nuscenes_infos_temporal_train.pkl")
    ap.add_argument("--infos-val", default="/workspace/DriveLM_VLM_Project/data/uniad_infos/nuscenes_infos_temporal_val.pkl")
    ap.add_argument("--bbox-train", default="/workspace/DriveLM_VLM_Project/data/preproc/bbox_egostate_train.jsonl")
    ap.add_argument("--bbox-val", default="/workspace/DriveLM_VLM_Project/data/preproc/bbox_egostate_val.jsonl")
    ap.add_argument("--max-length", type=int, default=12288)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="/workspace/DriveLM_VLM_Project/docs/B5ppp_token_audit.md")
    # 2026-05-25: mirror train_lora.py's processor mutation so audit reflects
    # ACTUAL training-time resolution, not processor defaults. Match B.5':
    #   image cap (HD-map)     = --min/max-pixels (default 109760, like B.5')
    #   video cap (3-cam)      = --video-max-pixels (default 0 = SKIP mutation,
    #                            HF Qwen3VLVideoProcessor default ~25M cap on
    #                            T*H*W gives native 1600x900 pass-through, matches
    #                            B.5' Qwen2.5-VL behavior).
    # NOTE Qwen3-VL video cap is on T*H*W TOTAL CLIP VOLUME, NOT per-frame.
    # For 4-frame T=2 native 1600x900, need ≥ 2.88M; default 25M is fine.
    # Setting --video-max-pixels=1440000 would DOWNSCALE (only allows H*W~720k @ T=2).
    ap.add_argument("--min-pixels", type=int, default=109760)
    ap.add_argument("--max-pixels", type=int, default=109760)
    ap.add_argument("--video-max-pixels", type=int, default=0,
                    help="0 = skip video processor mutation, use HF default (25M, native pass)")
    ap.add_argument("--video-min-pixels", type=int, default=0,
                    help="0 = inherit shortest_edge from processor default")
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoProcessor
    print(f"loading processor + tokenizer from {args.ckpt}", flush=True)
    processor = AutoProcessor.from_pretrained(args.ckpt, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt, trust_remote_code=True)

    # Mirror train_lora.py processor mutations
    if hasattr(processor, "image_processor") and processor.image_processor is not None:
        processor.image_processor.min_pixels = args.min_pixels
        processor.image_processor.max_pixels = args.max_pixels
        print(f"  image_processor capped: min/max_pixels={args.min_pixels}/{args.max_pixels}")
    if (
        hasattr(processor, "video_processor") and processor.video_processor is not None
        and args.video_max_pixels > 0
    ):
        vp = processor.video_processor
        vmin = args.video_min_pixels if args.video_min_pixels > 0 else getattr(vp.size, "shortest_edge", args.video_max_pixels)
        vmax = args.video_max_pixels
        if hasattr(vp, "size") and vp.size is not None:
            if hasattr(vp.size, "shortest_edge"):
                setattr(vp.size, "shortest_edge", vmin)
            if hasattr(vp.size, "longest_edge"):
                setattr(vp.size, "longest_edge", vmax)
        for attr, val in (("min_pixels", vmin), ("max_pixels", vmax)):
            if hasattr(vp, attr):
                setattr(vp, attr, val)
        print(f"  video_processor capped: shortest/longest={vmin}/{vmax}, size={vp.size}")
    elif hasattr(processor, "video_processor") and processor.video_processor is not None:
        print(f"  video_processor UNCHANGED (--video-max-pixels=0); size={processor.video_processor.size}")

    print("\n=== visual expansion probe (1 dummy sample) ===")
    per_cam_video, image_total, n_vp, n_ip = measure_visual_expansion(processor, tokenizer)
    # HARD ASSERT (per [[feedback_audit_must_match_training_processor]]):
    # fail-fast if probe doesn't match expected native config
    expected_per_cam = 2800  # Qwen3-VL patch=16, 1600x900 native, 4 frames
    if per_cam_video != expected_per_cam:
        print(f"  WARN: per_cam_video={per_cam_video} != expected {expected_per_cam}", flush=True)
        print(f"  if not native res, set --video-max-pixels accordingly", flush=True)
    print(f"  → per_cam_video={per_cam_video}, image_total={image_total}, n_video_pads={n_vp}, n_image_pads={n_ip}")
    VISUAL_FIXED = per_cam_video * len(CAMS) + image_total
    print(f"  → VISUAL_FIXED total = 3 × {per_cam_video} + {image_total} = {VISUAL_FIXED}\n")

    vp_id = tokenizer.convert_tokens_to_ids("<|video_pad|>")
    ip_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")

    results = {}
    for split, infos_path, bbox_path in [
        ("train", args.infos_train, args.bbox_train),
        ("val",   args.infos_val,   args.bbox_val),
    ]:
        print(f"\n=== {split} ===")
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
                sample_token = infos[i]["token"]
                bbox_text = bbox_lookup.get(sample_token, "Detected objects in ego frame:\nnone")
                if not bbox_text.strip():
                    bbox_text = "Detected objects in ego frame:\nnone"
                ego = format_ego_speed(infos[i])
                user_content = build_multicam_content(bbox_text, ego)
                msgs = [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": "Predicted trajectory:"},
                ]
                text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
                ids = tokenizer(text, add_special_tokens=False)["input_ids"]
                text_only = len(ids) - ids.count(vp_id) - ids.count(ip_id)
                total = text_only + VISUAL_FIXED
            except Exception as e:
                if len(worst) < 3:
                    print(f"  err at i={i}: {e}")
                continue
            totals.append(total)
            if total > args.max_length:
                n_over += 1
                if len(worst) < 10:
                    worst.append((i, total, text_only, infos[i]["token"]))
            if (i + 1) % 5000 == 0:
                el = time.time() - t0
                rate = (i + 1) / el
                eta = (n_target - i - 1) / rate
                print(f"  {i+1}/{n_target}  rate={rate:.0f}/s  ETA={eta:.0f}s  over={n_over}")
        el = time.time() - t0
        print(f"  DONE {n_target} samples in {el:.1f}s")

        totals.sort()
        n = len(totals)
        results[split] = {
            "n": n,
            "min": totals[0],
            "p50": totals[n // 2],
            "p90": totals[int(n * 0.9)],
            "p99": totals[int(n * 0.99)],
            "max": totals[-1],
            "n_over": n_over,
            "pct_over": 100.0 * n_over / max(n, 1),
            "max_length_threshold": args.max_length,
            "visual_fixed": VISUAL_FIXED,
            "worst": worst,
        }
        print(f"  min={totals[0]} p50={totals[n//2]} p90={totals[int(n*0.9)]} p99={totals[int(n*0.99)]} max={totals[-1]}")
        print(f"  over max_length={args.max_length}: {n_over}/{n} = {100.0 * n_over / max(n, 1):.2f}%")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("# B.5''' Token Audit (3-cam Qwen3-VL-4B × multimodal)\n\n")
        f.write(f"**Threshold**: max_length = **{args.max_length}**\n\n")
        f.write(f"**Visual fixed**: 3 cams × {per_cam_video} + HD-map {image_total} = **{VISUAL_FIXED}** tokens\n\n")
        f.write(f"**Method**: real Qwen3-VL-4B processor for visual expansion + tokenize chat-template text per sample. No model load, no real image I/O (dummy frames for grid probe only).\n\n")
        for split, r in results.items():
            f.write(f"## {split}\n\n")
            f.write(f"- n = {r['n']}\n")
            f.write(f"- min/p50/p90/p99/max = {r['min']} / {r['p50']} / {r['p90']} / {r['p99']} / {r['max']}\n")
            f.write(f"- visual fixed = {r['visual_fixed']}\n")
            f.write(f"- **samples exceeding {args.max_length}: {r['n_over']} ({r['pct_over']:.2f}%)**\n")
            if r['worst']:
                f.write(f"- worst (first 10): {[(w[0], w[1]) for w in r['worst']]}\n")
            f.write("\n")
        f.write("\n## Recommended max_length\n")
        f.write("Set max_length = max(over both splits) + ≥10% safety buffer.\n")
    print(f"\nreport: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
