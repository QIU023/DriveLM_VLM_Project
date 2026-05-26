"""Build a Lance-columnar lakehouse dataset for the nuScenes multimodal VLA (P0).

Converts a SUBSET of nuScenes multimodal planning samples into a Lance dataset
at ``data_infra/lance/nusc_mm.lance``. One row per sample:

    sample_token   (str)   nuScenes keyframe token
    scene_token    (str)   parent scene token
    timestamp      (int64) lidar canonical timestamp (microseconds)
    scenario       (str)   ego-motion bucket (reuses planning_eval.classify_scenario)
    hdmap_png      (binary)224x224 HD-map BEV PNG bytes (or black substitute)
    bbox_text      (str)   3D bbox + ego text (from bbox_egostate_{split}.jsonl)
    ego_speed      (float) CAN-bus ego speed scalar (m/s) at current frame
    traj_gt        (list<float>) future xy waypoints, flattened [x0,y0,x1,y1,...]
    camera_paths   (list<str>)   ABSOLUTE paths to the 4 past CAM_FRONT frames
                                 (paths NOT pixels — keeps the Lance file small;
                                 vision tokens are cached separately by
                                 cache_vision_tokens.py)

Why store paths not pixels: the camera JPEGs already live on disk; duplicating
them as binary blobs would balloon the Lance file with no access-pattern win.
The whole point of the lakehouse is the *derived* artifact — the cached vision
tokens (int8) — which IS stored columnar and is what the train-time path reads.

This module deliberately does NOT instantiate the Qwen processor or any model:
building the dataset is a pure CPU/IO operation (read infos pkl, read PNG bytes,
compute waypoints + scenario). The vision-token cache is a separate GPU pass.

Usage:
    export HF_HOME=/workspace/.hf_home; unset HF_HUB_OFFLINE
    /usr/bin/python3 data_infra/lance/build_lance_dataset.py --n 3000
    /usr/bin/python3 data_infra/lance/build_lance_dataset.py --n 8   # smoke
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np
import pyarrow as pa
import lance

# Reuse the repo's scenario classifier + geometry from scripts/.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SCRIPTS = os.path.join(_REPO, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from planning_eval import classify_scenario  # noqa: E402
from planning_dataset import quat_to_R, CAN_BUS_SPEED_IDX  # noqa: E402

NUM_PAST_FRAMES = 4
NUM_FUTURE_WP = 6
HDMAP_SIDE_PX = 224

DEFAULT_INFOS = os.path.join(_REPO, "data/uniad_infos/nuscenes_infos_temporal_train.pkl")
DEFAULT_NUSC_ROOT = os.path.join(_REPO, "data/nuscenes")
DEFAULT_HDMAP_DIR = os.path.join(_REPO, "data/preproc/hdmap_bev")
DEFAULT_BBOX_JSONL = os.path.join(_REPO, "data/preproc/bbox_egostate_{split}.jsonl")
DEFAULT_OUT = os.path.join(_REPO, "data_infra/lance/nusc_mm.lance")


def _black_png_bytes() -> bytes:
    """Black 224x224 RGB PNG (missing-HD-map substitute; mirrors the dataset)."""
    from PIL import Image
    import io

    buf = io.BytesIO()
    Image.new("RGB", (HDMAP_SIDE_PX, HDMAP_SIDE_PX), color=(0, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


def _image_path(info: dict, nusc_root: str, cam: str = "CAM_FRONT") -> str:
    rel = info["cams"][cam]["data_path"]
    if rel.startswith("./"):
        rel = rel[2:]
    if rel.startswith("data/nuscenes/"):
        rel = rel[len("data/nuscenes/"):]
    return os.path.join(nusc_root, rel)


def _ego_speed(info: dict) -> float:
    cb = info.get("can_bus")
    if cb is None:
        return 0.0
    try:
        return max(0.0, float(cb[CAN_BUS_SPEED_IDX]))
    except (IndexError, TypeError, ValueError):
        return 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000, help="number of train samples")
    ap.add_argument("--infos", default=DEFAULT_INFOS)
    ap.add_argument("--nusc-root", default=DEFAULT_NUSC_ROOT)
    ap.add_argument("--hdmap-dir", default=DEFAULT_HDMAP_DIR)
    ap.add_argument("--bbox-jsonl", default=DEFAULT_BBOX_JSONL)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    print(f"[build] loading infos: {args.infos}")
    with open(args.infos, "rb") as f:
        blob = pickle.load(f)
    infos = blob["infos"] if isinstance(blob, dict) and "infos" in blob else blob
    tok2idx = {info["token"]: i for i, info in enumerate(infos)}
    print(f"[build] {len(infos)} infos total")

    # Load bbox jsonl.
    bbox_path = args.bbox_jsonl.replace("{split}", args.split)
    print(f"[build] loading bbox: {bbox_path}")
    bbox_map: dict[str, str] = {}
    import json

    with open(bbox_path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            t = row.get("sample_token")
            if t is not None:
                bbox_map[t] = row.get("bbox_text", "")

    hdmap_split_dir = os.path.join(args.hdmap_dir, args.split)
    black_png = _black_png_bytes()

    def has_full_future(base_idx: int) -> bool:
        cur = infos[base_idx]
        for _ in range(NUM_FUTURE_WP):
            nxt = cur.get("next")
            if not nxt or nxt not in tok2idx:
                return False
            cur = infos[tok2idx[nxt]]
        return True

    def walk_history(base_idx: int) -> list:
        out = [infos[base_idx]]
        cur = infos[base_idx]
        for _ in range(NUM_PAST_FRAMES - 1):
            prev = cur.get("prev")
            if prev and prev in tok2idx:
                cur = infos[tok2idx[prev]]
            out.append(cur)
        return out[::-1]

    def compute_waypoints(base_idx: int):
        cur = infos[base_idx]
        R_cur_T = quat_to_R(cur["ego2global_rotation"]).T
        p_cur = np.asarray(cur["ego2global_translation"], dtype=np.float64)
        wp = np.zeros((NUM_FUTURE_WP, 2), dtype=np.float32)
        mask = np.zeros((NUM_FUTURE_WP,), dtype=np.float32)
        c = cur
        for i in range(NUM_FUTURE_WP):
            nxt = c.get("next")
            if not nxt or nxt not in tok2idx:
                break
            c = infos[tok2idx[nxt]]
            local = R_cur_T @ (np.asarray(c["ego2global_translation"], dtype=np.float64) - p_cur)
            wp[i, 0] = local[0]
            wp[i, 1] = local[1]
            mask[i] = 1.0
        return wp, mask

    # Select the first N indices with a full future chain (matches the dataset's
    # require_full_future filter, so the subset == the trainer's first N samples).
    keep = [i for i in range(len(infos)) if has_full_future(i)]
    keep = keep[: args.n]
    print(f"[build] selected {len(keep)} samples (full-future) for n={args.n}")

    sample_tokens, scene_tokens, timestamps = [], [], []
    scenarios, hdmap_pngs, bbox_texts = [], [], []
    ego_speeds, traj_gts, camera_paths_col = [], [], []

    n_missing_hdmap = 0
    n_missing_cam = 0
    for base_idx in keep:
        info = infos[base_idx]
        tok = info["token"]

        wp, mask = compute_waypoints(base_idx)
        scenario = classify_scenario(wp, mask)

        # HD map PNG bytes (or black substitute on cache miss).
        hp = os.path.join(hdmap_split_dir, f"{tok}.png")
        if os.path.isfile(hp):
            with open(hp, "rb") as fh:
                png = fh.read()
        else:
            png = black_png
            n_missing_hdmap += 1

        hist = walk_history(base_idx)
        cam_paths = [_image_path(h, args.nusc_root) for h in hist]
        for p in cam_paths:
            if not os.path.isfile(p):
                n_missing_cam += 1

        sample_tokens.append(tok)
        scene_tokens.append(info.get("scene_token", ""))
        timestamps.append(int(info.get("timestamp", 0)))
        scenarios.append(scenario)
        hdmap_pngs.append(png)
        bbox_texts.append(bbox_map.get(tok, ""))
        ego_speeds.append(_ego_speed(info))
        traj_gts.append(wp.reshape(-1).astype(np.float32).tolist())
        camera_paths_col.append(cam_paths)

    print(f"[build] missing HD maps (black substitute): {n_missing_hdmap}")
    print(f"[build] missing camera files: {n_missing_cam}")

    table = pa.table(
        {
            "sample_token": pa.array(sample_tokens, pa.string()),
            "scene_token": pa.array(scene_tokens, pa.string()),
            "timestamp": pa.array(timestamps, pa.int64()),
            "scenario": pa.array(scenarios, pa.string()),
            "hdmap_png": pa.array(hdmap_pngs, pa.binary()),
            "bbox_text": pa.array(bbox_texts, pa.string()),
            "ego_speed": pa.array(ego_speeds, pa.float32()),
            "traj_gt": pa.array(traj_gts, pa.list_(pa.float32())),
            "camera_paths": pa.array(camera_paths_col, pa.list_(pa.string())),
        }
    )

    if os.path.exists(args.out):
        import shutil

        shutil.rmtree(args.out)
    lance.write_dataset(table, args.out)

    # Verify: reopen, count, print one row + scenario histogram.
    ds = lance.dataset(args.out)
    n_rows = ds.count_rows()
    on_disk = _dir_bytes(args.out)
    print("\n[verify] ---------------------------------------------------------")
    print(f"[verify] Lance dataset at {args.out}")
    print(f"[verify] rows = {n_rows}  on-disk = {on_disk/1e6:.1f} MB")
    row0 = ds.take([0]).to_pylist()[0]
    print(f"[verify] row[0].sample_token = {row0['sample_token']}")
    print(f"[verify] row[0].scenario     = {row0['scenario']}")
    print(f"[verify] row[0].ego_speed    = {row0['ego_speed']:.2f} m/s")
    print(f"[verify] row[0].timestamp    = {row0['timestamp']}")
    print(f"[verify] row[0].traj_gt[:4]  = {row0['traj_gt'][:4]}")
    print(f"[verify] row[0].hdmap_png    = {len(row0['hdmap_png'])} bytes")
    print(f"[verify] row[0].camera_paths = {len(row0['camera_paths'])} paths")
    print(f"[verify]   first cam path    = {row0['camera_paths'][0]}")
    print(f"[verify]   cam path exists?  = {os.path.isfile(row0['camera_paths'][0])}")
    print(f"[verify] row[0].bbox_text[:80] = {row0['bbox_text'][:80]!r}")
    # Scenario histogram.
    scen_col = ds.to_table(columns=["scenario"]).column("scenario").to_pylist()
    from collections import Counter

    print(f"[verify] scenario histogram = {dict(Counter(scen_col))}")
    print("[verify] ---------------------------------------------------------")
    return 0


def _dir_bytes(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for fn in files:
            total += os.path.getsize(os.path.join(root, fn))
    return total


if __name__ == "__main__":
    raise SystemExit(main())
