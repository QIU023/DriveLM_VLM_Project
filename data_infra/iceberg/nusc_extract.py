"""nuScenes VLA metadata extraction for the Iceberg lakehouse.

Reads per-keyframe `infos` (nuscenes_infos_temporal_{split}.pkl) + the
bbox/ego JSONL, and derives the row schema for the `keyframes` Iceberg table:

    sample_token, scene_token, timestamp, split, scenario,
    ego_speed, n_objects, has_hdmap

`scenario` is reproduced *byte-for-byte* from scripts/planning_eval.py's
`classify_scenario` (same thresholds, same ego-frame future-waypoint
construction from scripts/planning_dataset.py `_compute_waypoints` /
`_walk_future`). This is what makes the lineage credible: the Iceberg
partition key matches the scenario buckets reported in
docs/eval_results/*.json.

CPU-only, no torch import (we replicate the small numpy bits directly).
"""

from __future__ import annotations

import json
import math
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants copied verbatim from scripts/planning_{eval,dataset}.py
# ---------------------------------------------------------------------------
HZ = 2.0
DT = 1.0 / HZ                       # 0.5 s
NUM_FUTURE_WP = 6                   # 6 waypoints over 3 s @ 2 Hz
CAN_BUS_SPEED_IDX = 13              # scalar ego speed magnitude (m/s)

# classify_scenario thresholds (planning_eval.py lines 548-552)
STRAIGHT_HEADING_THRESH_RAD = math.radians(5.0)
TURNING_HEADING_THRESH_RAD = math.radians(15.0)
LANE_CHANGE_LATERAL_THRESH_M = 1.5
BRAKING_DECEL_THRESH_M_S = 2.0
STATIONARY_SPEED_THRESH_M_S = 1.0


def quat_to_R(q_wxyz) -> np.ndarray:
    """Hamilton (w,x,y,z) -> 3x3 rotation matrix (nuScenes ego2global)."""
    w, x, y, z = (float(q_wxyz[0]), float(q_wxyz[1]),
                  float(q_wxyz[2]), float(q_wxyz[3]))
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    return np.array(
        [
            [1.0 - s * (y * y + z * z), s * (x * y - z * w),       s * (x * z + y * w)],
            [s * (x * y + z * w),       1.0 - s * (x * x + z * z), s * (y * z - x * w)],
            [s * (x * z - y * w),       s * (y * z + x * w),       1.0 - s * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _wrap_to_pi(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def _step_headings(wp: np.ndarray) -> np.ndarray:
    T = wp.shape[0]
    if T == 0:
        return np.zeros((0,), dtype=np.float64)
    prev = np.zeros((T, 2), dtype=np.float64)
    prev[1:] = wp[:-1]
    delta = wp.astype(np.float64) - prev
    return np.arctan2(delta[:, 1], delta[:, 0])


def _step_distances(wp: np.ndarray) -> np.ndarray:
    T = wp.shape[0]
    if T == 0:
        return np.zeros((0,), dtype=np.float64)
    prev = np.zeros((T, 2), dtype=np.float64)
    prev[1:] = wp[:-1]
    delta = wp.astype(np.float64) - prev
    return np.sqrt((delta ** 2).sum(axis=-1))


def classify_scenario(gt_wp: np.ndarray, valid: np.ndarray) -> str:
    """Reproduction of planning_eval.classify_scenario (first-match priority)."""
    T = int(gt_wp.shape[0])
    if T < 2 or valid.sum() < 2:
        return "stationary"
    last_valid_idx = int(np.where(valid > 0.5)[0].max())
    if last_valid_idx < 1:
        return "stationary"

    step_d = _step_distances(gt_wp[: last_valid_idx + 1])
    total_d = float(step_d.sum())
    elapsed_s = float(last_valid_idx + 1) * DT
    avg_speed = total_d / max(elapsed_s, 1e-6)

    headings = _step_headings(gt_wp[: last_valid_idx + 1])
    heading_change = _wrap_to_pi(float(headings[-1] - headings[0]))
    abs_heading_change = abs(heading_change)

    lateral_disp = abs(float(gt_wp[last_valid_idx, 1]))

    v_start = float(step_d[0]) / DT
    v_end = float(step_d[-1]) / DT
    speed_drop = v_start - v_end

    if avg_speed <= STATIONARY_SPEED_THRESH_M_S:
        return "stationary"
    if abs_heading_change >= TURNING_HEADING_THRESH_RAD:
        return "turning"
    if lateral_disp > LANE_CHANGE_LATERAL_THRESH_M:
        return "lane_change"
    if speed_drop > BRAKING_DECEL_THRESH_M_S:
        return "braking"
    if abs_heading_change < STRAIGHT_HEADING_THRESH_RAD:
        return "straight"
    return "cruising"


# ---------------------------------------------------------------------------
# Source loading
# ---------------------------------------------------------------------------
def load_infos(pkl_path: str) -> Tuple[List[dict], Dict[str, int]]:
    with open(pkl_path, "rb") as f:
        blob = pickle.load(f)
    infos = blob["infos"] if isinstance(blob, dict) and "infos" in blob else blob
    tok2idx = {info["token"]: i for i, info in enumerate(infos)}
    return infos, tok2idx


def load_bbox_egostate(jsonl_path: str) -> Dict[str, dict]:
    """token -> {bbox_text, egostate_text}."""
    out: Dict[str, dict] = {}
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[r["sample_token"]] = r
    return out


def _walk_future(infos, tok2idx, base_idx, num_future=NUM_FUTURE_WP) -> List[dict]:
    cur = infos[base_idx]
    out: List[dict] = []
    for _ in range(num_future):
        nxt = cur.get("next")
        if not nxt or nxt not in tok2idx:
            break
        cur = infos[tok2idx[nxt]]
        out.append(cur)
    return out


def _compute_waypoints(infos, tok2idx, base_idx,
                       num_future=NUM_FUTURE_WP) -> Tuple[np.ndarray, np.ndarray]:
    cur = infos[base_idx]
    R_cur = quat_to_R(cur["ego2global_rotation"])
    p_cur = np.asarray(cur["ego2global_translation"], dtype=np.float64)
    R_cur_T = R_cur.T
    future = _walk_future(infos, tok2idx, base_idx, num_future)
    wp = np.zeros((num_future, 2), dtype=np.float32)
    mask = np.zeros((num_future,), dtype=np.float32)
    for i, info in enumerate(future):
        p_f = np.asarray(info["ego2global_translation"], dtype=np.float64)
        local = R_cur_T @ (p_f - p_cur)
        wp[i, 0] = local[0]
        wp[i, 1] = local[1]
        mask[i] = 1.0
    return wp, mask


def _ego_speed(info: dict) -> float:
    cb = info.get("can_bus")
    if cb is None:
        return 0.0
    try:
        return max(0.0, float(cb[CAN_BUS_SPEED_IDX]))
    except (IndexError, TypeError, ValueError):
        return 0.0


def _count_objects(bbox_text: str) -> int:
    if not bbox_text:
        return 0
    return sum(1 for ln in bbox_text.splitlines() if ln.lstrip().startswith("- "))


def build_rows(pkl_path: str, jsonl_path: str, split: str,
               hdmap_dir: str = "", limit: int = 0,
               full_future_only: bool = True) -> List[dict]:
    """Yield keyframes-table rows for one split.

    `limit>0` truncates (smoke). `has_hdmap` = whether a BEV png exists.
    Rows are returned only for tokens present in BOTH infos and bbox jsonl
    (so every row is a real, trainable VLA planning sample).

    `full_future_only=True` keeps only keyframes with all NUM_FUTURE_WP future
    waypoints valid — this exactly reproduces the eval-scored set
    (val: 5119 rows, matching docs/eval_results n_scored=5119 and its
    scenario_counts). Set False to keep scene-tail frames too (val: 6019).
    """
    infos, tok2idx = load_infos(pkl_path)
    bbox = load_bbox_egostate(jsonl_path)

    hdmap_tokens = set()
    if hdmap_dir and os.path.isdir(hdmap_dir):
        for fn in os.listdir(hdmap_dir):
            if fn.endswith(".png"):
                hdmap_tokens.add(fn[:-4])

    rows: List[dict] = []
    for idx, info in enumerate(infos):
        tok = info["token"]
        if tok not in bbox:
            continue
        wp, mask = _compute_waypoints(infos, tok2idx, idx)
        if full_future_only and mask.sum() < NUM_FUTURE_WP:
            continue
        scenario = classify_scenario(wp, mask)
        rows.append({
            "sample_token": tok,
            "scene_token": info["scene_token"],
            "timestamp": int(info["timestamp"]),
            "split": split,
            "scenario": scenario,
            "ego_speed": float(_ego_speed(info)),
            "n_objects": int(_count_objects(bbox[tok].get("bbox_text", ""))),
            "has_hdmap": bool(tok in hdmap_tokens),
        })
        if limit and len(rows) >= limit:
            break
    return rows
