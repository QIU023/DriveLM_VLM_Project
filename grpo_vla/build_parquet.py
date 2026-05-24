#!/usr/bin/env python3
"""Convert VeRLNuScenesDataset -> veRL parquet files (FAST, parallel, JPEG).

Bottleneck fixes vs v1:
  - PNG (150ms/img @ 1280x720) -> JPEG q75 @ 448px max edge (~5ms/img)
  - Single-thread -> multiprocessing.Pool(N_WORKERS) over sample indices
  - 24K rows full set -> subsample N for sensible epoch count vs total_training_steps

veRL RLHFDataset row schema (matches what the trainer expects):
    prompt           : str
    images           : list[bytes]   (JPEG, resized)
    extra_info       : json str      (reward-side: gt_waypoints, bbox_3d_list, ego)
    ground_truth     : json str      (mirror)
    data_source      : str

Usage:
    /usr/bin/python3 build_parquet.py --config configs/grpo_b5prime_3cam.yaml \\
        --split train --max-samples 12000 --workers 8 --max-edge 448 --jpeg-q 75
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Tuple

GRPO_DIR = Path("/workspace/DriveLM_VLM_Project/grpo_vla")
DATA_DIR = GRPO_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _jpeg_bytes(img, max_edge: int = 448, q: int = 75) -> bytes:
    """Resize keeping aspect ratio (max edge=max_edge) + JPEG encode."""
    img = img.convert("RGB")
    w, h = img.size
    scale = max_edge / max(w, h) if max(w, h) > max_edge else 1.0
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=q, optimize=False)
    return buf.getvalue()


# Globals for worker (set in initializer)
_DS = None
_MAX_EDGE = 448
_JPEG_Q = 75


def _worker_init(cfg_path: str, split: str, max_edge: int, jpeg_q: int):
    global _DS, _MAX_EDGE, _JPEG_Q
    import yaml
    sys.path.insert(0, str(GRPO_DIR))
    sys.path.insert(0, str(GRPO_DIR.parent))
    sys.path.insert(0, str(GRPO_DIR.parent / "scripts"))
    try:
        from grpo_vla.dataset_adapter import build_verl_nuscenes_dataset  # type: ignore
    except Exception:
        from dataset_adapter import build_verl_nuscenes_dataset  # type: ignore
    from transformers import AutoProcessor
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f) or {}
    dcfg = cfg.get("data", cfg)
    proc_path = cfg.get("actor_rollout_ref", {}).get("model", {}).get("path")
    proc = AutoProcessor.from_pretrained(proc_path, trust_remote_code=True)
    _DS = build_verl_nuscenes_dataset(dcfg, proc, split=split)
    _MAX_EDGE = max_edge
    _JPEG_Q = jpeg_q


def _process_one(i: int):
    """Build one row in veRL's expected schema:
        prompt: [{"role": "user", "content": "<image>...<image> {text}"}]
        images: [{"bytes": jpeg_bytes}, ...]   # one per <image> marker
        extra_info: dict (not json str)
        reward_model: {"style": "rule", "ground_truth": gt_waypoints}
        data_source: str

    Flatten 3 cam × 4 frame videos to 12 individual <image> markers + 1
    HD-map = 13 images total. This loses video-pad temporal merging from
    SFT but works with veRL's <image>/<video> regex split path (line 320
    of rl_dataset.py). Model adapts during RL since KL anchor keeps it
    close to SFT behavior on the new visual format.
    """
    try:
        s = _DS[i]
    except Exception as e:
        return ("ERR", i, str(e)[:200])
    mm = s.get("multi_modal_data", {}) or {}
    images_payload: List[dict] = []
    image_marker_blocks: List[str] = []

    # Flatten 3 cam clips → 12 frames as images, with per-frame label
    cam_labels = ["FRONT", "FRONT_LEFT", "FRONT_RIGHT"]
    video_clips = mm.get("video") or []
    for c_idx, clip in enumerate(video_clips):
        label = cam_labels[c_idx] if c_idx < len(cam_labels) else f"CAM_{c_idx}"
        for f_idx, frame in enumerate(clip):
            images_payload.append({"bytes": _jpeg_bytes(frame, _MAX_EDGE, _JPEG_Q)})
            image_marker_blocks.append(f"Camera {label} t-{len(clip)-1-f_idx}: <image>")

    # HD-map image
    for img in (mm.get("image") or []):
        images_payload.append({"bytes": _jpeg_bytes(img, _MAX_EDGE, _JPEG_Q)})
        image_marker_blocks.append("HD-map BEV: <image>")

    # Build the user content: image block headers + bbox + ego + planning prompt
    ei = s.get("extra_info", {}) or {}
    bbox_text = ei.get("bbox_text", "") or ""
    ego_state = ei.get("ego_state", {}) or {}
    ego_speed_mps = float(ego_state.get("speed_mps", 0.0))
    planning_prompt = ei.get("planning_prompt") or (
        "Given the above observations, predict the next 6 ego waypoints "
        "as <traj_start>{12 bin tokens}<traj_end>."
    )
    content = "\n".join(image_marker_blocks)
    content += f"\n\nDetected objects in ego frame:\n{bbox_text}\n"
    content += f"\nEgo speed at current frame: {ego_speed_mps:.2f} m/s\n"
    content += f"\n{planning_prompt}"

    gt_wp = s.get("ground_truth", [])
    if hasattr(gt_wp, "tolist"):
        gt_wp = gt_wp.tolist()

    return ("OK", i, {
        "prompt": [{"role": "user", "content": content}],
        "images": images_payload,
        "extra_info": dict(ei),
        "reward_model": {"style": "rule", "ground_truth": gt_wp},
        "data_source": "nusc_planning",
    })


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", choices=["train", "val"], required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-samples", type=int, default=None,
                    help="cap N samples (random subsample with fixed seed); default by split")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-edge", type=int, default=448, help="resize images to max edge")
    ap.add_argument("--jpeg-q", type=int, default=75)
    ap.add_argument("--chunk-size", type=int, default=512, help="rows per parquet write")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.max_samples is None:
        args.max_samples = 12000 if args.split == "train" else 500

    out_path = Path(args.out) if args.out else (
        DATA_DIR / f"nusc_planning_{args.split}.parquet"
    )
    if out_path.exists() and not args.force:
        print(f"[build_parquet] {out_path} exists; skip (use --force to rebuild)")
        return 0

    # Build full dataset (parent) just to get length + pick indices
    sys.path.insert(0, str(GRPO_DIR))
    sys.path.insert(0, str(GRPO_DIR.parent))
    sys.path.insert(0, str(GRPO_DIR.parent / "scripts"))
    try:
        from grpo_vla.dataset_adapter import build_verl_nuscenes_dataset  # type: ignore
    except Exception:
        from dataset_adapter import build_verl_nuscenes_dataset  # type: ignore
    import yaml
    from transformers import AutoProcessor
    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}
    dcfg = cfg.get("data", cfg)
    proc_path = cfg.get("actor_rollout_ref", {}).get("model", {}).get("path")
    proc = AutoProcessor.from_pretrained(proc_path, trust_remote_code=True)
    ds = build_verl_nuscenes_dataset(dcfg, proc, split=args.split)
    N_FULL = len(ds)

    import random
    rng = random.Random(args.seed)
    if args.max_samples >= N_FULL:
        idxs = list(range(N_FULL))
    else:
        idxs = rng.sample(range(N_FULL), args.max_samples)
        idxs.sort()
    n_total = len(idxs)
    print(f"[build_parquet] split={args.split} N_full={N_FULL} -> sampling {n_total} (seed={args.seed})")
    print(f"[build_parquet] workers={args.workers} max_edge={args.max_edge} jpeg_q={args.jpeg_q}")

    import pyarrow as pa
    import pyarrow.parquet as pq

    writer = None
    t0 = time.time()
    n_ok = 0; n_err = 0
    chunk_rows: List[Dict[str, Any]] = []

    def _flush():
        nonlocal writer
        if not chunk_rows:
            return
        import pandas as pd
        df = pd.DataFrame(chunk_rows)
        tbl = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(str(out_path), tbl.schema, compression="zstd")
        writer.write_table(tbl)
        chunk_rows.clear()

    with Pool(
        processes=args.workers,
        initializer=_worker_init,
        initargs=(args.config, args.split, args.max_edge, args.jpeg_q),
    ) as pool:
        for i_done, (status, idx, payload) in enumerate(
            pool.imap_unordered(_process_one, idxs, chunksize=8), start=1
        ):
            if status == "OK":
                chunk_rows.append(payload)
                n_ok += 1
            else:
                n_err += 1
                if n_err <= 5:
                    print(f"  ERR i={idx}: {payload}", file=sys.stderr)
            if len(chunk_rows) >= args.chunk_size:
                _flush()
                elapsed = time.time() - t0
                rate = i_done / max(elapsed, 1e-3)
                eta_min = (n_total - i_done) / max(rate, 1e-3) / 60
                print(f"  {i_done}/{n_total}  ok={n_ok} err={n_err}  "
                      f"{rate:.2f} rows/s  ETA {eta_min:.1f} min")

    _flush()
    if writer is not None:
        writer.close()

    size_mb = out_path.stat().st_size / 1e6 if out_path.exists() else 0
    elapsed = time.time() - t0
    print(f"[build_parquet] DONE {out_path} "
          f"({size_mb:.0f} MB, {n_ok} ok, {n_err} err, {elapsed:.0f}s = {elapsed/60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
