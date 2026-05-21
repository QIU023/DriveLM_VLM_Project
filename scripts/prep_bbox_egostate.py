"""Serialize per-keyframe 3D bbox + ego state into compact text prompts for the
multi-modal VLA prefix (phase13).

Reads a UniAD-style temporal infos pkl (e.g. data/uniad_infos/nuscenes_infos_temporal_train.pkl)
and writes a JSONL where each line is:

    {"sample_token": "...", "bbox_text": "...", "egostate_text": "..."}

Design notes (see docs/multimodal/bbox_egostate_README.md):
  * gt_boxes in the UniAD pkl are already expressed in the LIDAR frame, which
    in nuScenes coincides with the ego frame up to a fixed [+0.94, 0, +1.84]
    translation and a near-identity rotation. For planner prompting we treat
    LIDAR-frame xy directly as ego-frame xy (the planner's own targets live in
    the same frame), so no extra transform is applied.
  * gt_boxes columns are [cx, cy, cz, w, l, h, yaw] (UniAD/BEVFormer convention).
    `yaw` is the heading in radians around +z.
  * gt_velocity is (vx, vy) in the global frame in the raw nuScenes pkl, but in
    our infos it has already been rotated into ego frame by the UniAD pipeline
    (matches what UniAD's planning head consumes). We include velocity in the
    text only when speed >= 0.3 m/s to avoid noise.
  * can_bus layout (see scripts/planning_dataset.py header comment):
        [13] -> ego speed magnitude (m/s)
        [12] -> ego yaw rate (rad/s, rotation_rate.z)
  * "Previous trajectory" is computed by walking the `prev` token chain
    backwards 4 steps and projecting each past ego2global position into the
    current ego frame.

CPU-only, no GPU side effects.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np


# Keep classes the planner actually needs to reason about. Skip street furniture
# (barriers, cones, debris, bicycle_racks) and the rare stroller class.
KEEP_CLASSES = {
    "car",
    "truck",
    "bus",
    "pedestrian",
    "motorcycle",
    "bicycle",
    "construction_vehicle",
    "trailer",
}

TOP_K = 10
PAST_STEPS = 4
SPEED_TEXT_THRESHOLD_MPS = 0.3  # below this we omit per-object velocity
CAN_BUS_SPEED_IDX = 13
CAN_BUS_YAWRATE_IDX = 12


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def quat_wxyz_to_rotmat(q: List[float]) -> np.ndarray:
    """Convert a (w, x, y, z) quaternion to a 3x3 rotation matrix.

    nuScenes/UniAD store rotations as (w, x, y, z); confirmed against the pkl
    field shapes (length 4, magnitude 1).
    """
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float64,
    )


def project_world_to_ego(
    p_world: np.ndarray, ego_t: np.ndarray, ego_R: np.ndarray
) -> np.ndarray:
    """Project a world-frame xyz point into the ego frame defined by (t, R).

    Ego pose convention in the pkl: a point in ego frame `p_e` maps to world
    via `p_w = R @ p_e + t`, so the inverse is `p_e = R^T @ (p_w - t)`.
    """
    return ego_R.T @ (p_world - ego_t)


# ---------------------------------------------------------------------------
# Per-sample serializers
# ---------------------------------------------------------------------------


def serialize_bboxes(info: dict) -> str:
    """Render the top-K nearest objects (whitelisted classes) as text."""
    boxes = info.get("gt_boxes")
    names = info.get("gt_names")
    if boxes is None or names is None or len(boxes) == 0:
        return "Detected objects in ego frame: none."

    velocities = info.get("gt_velocity")
    valid = info.get("valid_flag")

    n = len(boxes)
    rows: List[Tuple[float, str]] = []  # (distance, formatted line)
    for i in range(n):
        if valid is not None and not bool(valid[i]):
            continue
        cls = str(names[i])
        if cls not in KEEP_CLASSES:
            continue
        cx, cy, cz, w, l, h, yaw = (float(v) for v in boxes[i, :7])
        dist = (cx * cx + cy * cy) ** 0.5
        line = (
            f"- {cls} at ({cx:.1f}, {cy:.1f}, {cz:.1f}) m, "
            f"size {l:.1f}x{w:.1f}x{h:.1f}, yaw {yaw:.2f} rad"
        )
        if velocities is not None and i < len(velocities):
            vx, vy = float(velocities[i, 0]), float(velocities[i, 1])
            spd = (vx * vx + vy * vy) ** 0.5
            if spd >= SPEED_TEXT_THRESHOLD_MPS and np.isfinite(spd):
                line += f", vel ({vx:.1f}, {vy:.1f}) m/s"
        rows.append((dist, line))

    if not rows:
        return "Detected objects in ego frame: none."

    rows.sort(key=lambda r: r[0])
    kept = rows[:TOP_K]
    body = "\n".join(line for _, line in kept)
    return f"Detected objects in ego frame:\n{body}"


def serialize_egostate(info: dict, past_traj: List[Tuple[float, float]]) -> str:
    """Render ego speed, yaw rate, and a 4-step relative past trajectory.

    past_traj: list of (dx, dy) in current ego frame for the last PAST_STEPS
    keyframes (most-recent-prev first ... oldest last). Empty if not available.
    """
    cb = info.get("can_bus")
    speed: Optional[float] = None
    yaw_rate: Optional[float] = None
    if cb is not None and len(cb) > max(CAN_BUS_SPEED_IDX, CAN_BUS_YAWRATE_IDX):
        try:
            speed = max(0.0, float(cb[CAN_BUS_SPEED_IDX]))
        except (TypeError, ValueError):
            speed = None
        try:
            yaw_rate = float(cb[CAN_BUS_YAWRATE_IDX])
        except (TypeError, ValueError):
            yaw_rate = None

    parts: List[str] = []
    if speed is not None and np.isfinite(speed):
        parts.append(f"speed {speed:.2f} m/s")
    else:
        parts.append("speed unknown")
    if yaw_rate is not None and np.isfinite(yaw_rate):
        parts.append(f"yaw rate {yaw_rate:.2f} rad/s")
    if past_traj:
        # Render oldest -> most-recent so the reader sees temporal order.
        ordered = list(reversed(past_traj))
        traj_str = ", ".join(f"({dx:.1f}, {dy:.1f})" for dx, dy in ordered)
        parts.append(
            f"prev trajectory (Δx, Δy) over last {len(ordered)} keyframes: {traj_str}"
        )
    else:
        parts.append("prev trajectory unavailable")
    return "Ego state: " + ", ".join(parts) + "."


# ---------------------------------------------------------------------------
# Past-trajectory walking
# ---------------------------------------------------------------------------


def build_token_index(infos: List[dict]) -> Dict[str, int]:
    return {s["token"]: i for i, s in enumerate(infos)}


def past_trajectory_in_ego(
    info: dict,
    infos: List[dict],
    tok2idx: Dict[str, int],
    n_steps: int = PAST_STEPS,
) -> List[Tuple[float, float]]:
    """Walk back up to `n_steps` `prev` links and project each ego2global
    position into the *current* ego frame. Returns most-recent-first."""
    try:
        ego_t = np.array(info["ego2global_translation"], dtype=np.float64)
        ego_R = quat_wxyz_to_rotmat(info["ego2global_rotation"])
    except (KeyError, TypeError, ValueError):
        return []

    out: List[Tuple[float, float]] = []
    cur = info
    for _ in range(n_steps):
        prev_tok = cur.get("prev", "")
        if not prev_tok or prev_tok not in tok2idx:
            break
        cur = infos[tok2idx[prev_tok]]
        try:
            p_world = np.array(cur["ego2global_translation"], dtype=np.float64)
        except (KeyError, TypeError, ValueError):
            break
        p_ego = project_world_to_ego(p_world, ego_t, ego_R)
        out.append((float(p_ego[0]), float(p_ego[1])))
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def process_pkl(infos_path: str, output_path: str) -> Tuple[int, List[dict]]:
    if not os.path.exists(infos_path):
        raise FileNotFoundError(infos_path)
    with open(infos_path, "rb") as f:
        blob = pickle.load(f)
    if not isinstance(blob, dict) or "infos" not in blob:
        raise RuntimeError(
            f"Unexpected pkl schema in {infos_path}: top-level keys "
            f"{list(blob.keys()) if isinstance(blob, dict) else type(blob)}"
        )
    infos = blob["infos"]
    tok2idx = build_token_index(infos)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    written = 0
    samples_for_audit: List[dict] = []
    t0 = time.time()
    with open(output_path, "w") as fout:
        for i, info in enumerate(infos):
            past = past_trajectory_in_ego(info, infos, tok2idx, PAST_STEPS)
            bbox_text = serialize_bboxes(info)
            ego_text = serialize_egostate(info, past)
            rec = {
                "sample_token": info.get("token", ""),
                "bbox_text": bbox_text,
                "egostate_text": ego_text,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
            samples_for_audit.append(rec)
            if (i + 1) % 5000 == 0:
                print(
                    f"  [{infos_path}] {i + 1}/{len(infos)} "
                    f"({(i + 1) / (time.time() - t0):.0f}/s)",
                    flush=True,
                )
    print(
        f"  wrote {written} lines to {output_path} in {time.time() - t0:.1f}s",
        flush=True,
    )
    return written, samples_for_audit


def cli() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--infos-pkl",
        action="append",
        required=True,
        help="UniAD-style temporal infos pkl. Pass multiple times to process "
        "train + val in one invocation.",
    )
    ap.add_argument(
        "--output",
        action="append",
        required=True,
        help="JSONL output path. Must be passed the same number of times as "
        "--infos-pkl, in matching order.",
    )
    ap.add_argument(
        "--smoke-only",
        action="store_true",
        help="Skip writing the full JSONL; only run the 3-sample smoke + "
        "round-trip check on the first --infos-pkl.",
    )
    ap.add_argument(
        "--token-audit",
        action="store_true",
        help="After writing, sample 100 lines from the first output and report "
        "Qwen2.5-VL token length stats.",
    )
    ap.add_argument(
        "--model-path",
        default="/workspace/models/Qwen2.5-VL-3B-Instruct",
        help="HF model dir for the tokenizer used in --token-audit.",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if len(args.infos_pkl) != len(args.output):
        print(
            f"ERROR: got {len(args.infos_pkl)} --infos-pkl but {len(args.output)} --output",
            file=sys.stderr,
        )
        return 2
    random.seed(args.seed)

    # ---- smoke: 3 real samples from first pkl
    print("=" * 72)
    print(f"SMOKE: loading {args.infos_pkl[0]} for 3-sample serialization check")
    print("=" * 72)
    with open(args.infos_pkl[0], "rb") as f:
        blob = pickle.load(f)
    infos = blob["infos"]
    tok2idx = build_token_index(infos)
    # Pick samples that are guaranteed to have a non-trivial prev chain
    # (frame_idx > 4) and >=1 whitelisted object — so the smoke prints
    # interesting output, not three "none." rows.
    chosen: List[int] = []
    for i in range(len(infos)):
        s = infos[i]
        if s.get("frame_idx", 0) < 4:
            continue
        names = s.get("gt_names")
        if names is None or not any(str(n) in KEEP_CLASSES for n in names):
            continue
        chosen.append(i)
        if len(chosen) == 3:
            break

    smoke_records: List[dict] = []
    for idx in chosen:
        info = infos[idx]
        past = past_trajectory_in_ego(info, infos, tok2idx, PAST_STEPS)
        rec = {
            "sample_token": info.get("token", ""),
            "bbox_text": serialize_bboxes(info),
            "egostate_text": serialize_egostate(info, past),
        }
        smoke_records.append(rec)
        print(f"\n--- sample idx={idx} token={rec['sample_token']} ---")
        print(rec["bbox_text"])
        print(rec["egostate_text"])

    # Round-trip
    print("\n--- round-trip check ---")
    rt_path = "/tmp/_bbox_egostate_roundtrip.jsonl"
    with open(rt_path, "w") as f:
        for r in smoke_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    reloaded = [json.loads(line) for line in open(rt_path)]
    assert reloaded == smoke_records, "round-trip mismatch!"
    print(f"round-trip OK ({len(reloaded)} records reread from {rt_path})")
    os.unlink(rt_path)

    if args.smoke_only:
        print("\n--smoke-only set; not writing full JSONL.")
        return 0

    # ---- full preprocess
    audit_samples: List[dict] = []
    for pkl_path, out_path in zip(args.infos_pkl, args.output):
        print("\n" + "=" * 72)
        print(f"PROCESS: {pkl_path} -> {out_path}")
        print("=" * 72)
        n, samples = process_pkl(pkl_path, out_path)
        if not audit_samples:
            audit_samples = samples

    # ---- token audit
    if args.token_audit:
        print("\n" + "=" * 72)
        print("TOKEN AUDIT")
        print("=" * 72)
        if len(audit_samples) < 100:
            print(
                f"WARNING: only {len(audit_samples)} samples available, using all"
            )
            picks = audit_samples
        else:
            picks = random.sample(audit_samples, 100)
        try:
            from transformers import AutoTokenizer
        except ImportError as e:
            print(f"transformers not importable, skipping token audit: {e}")
            return 0
        tok = AutoTokenizer.from_pretrained(args.model_path)
        lens: List[int] = []
        for rec in picks:
            text = rec["bbox_text"] + "\n" + rec["egostate_text"]
            ids = tok.encode(text, add_special_tokens=False)
            lens.append(len(ids))
        lens_arr = np.array(lens)
        print(f"sampled N={len(lens)} records")
        print(f"  min   : {int(lens_arr.min())}")
        print(f"  median: {int(np.median(lens_arr))}")
        print(f"  p90   : {int(np.percentile(lens_arr, 90))}")
        print(f"  p99   : {int(np.percentile(lens_arr, 99))}")
        print(f"  max   : {int(lens_arr.max())}")
        if int(lens_arr.max()) > 500:
            print("  FLAG: max > 500 tokens — would balloon LM context")
        else:
            print("  OK: max <= 500 tokens")

    return 0


if __name__ == "__main__":
    sys.exit(cli())
