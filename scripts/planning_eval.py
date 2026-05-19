"""Offline planning evaluation for the nuScenes Phase-B VLA.

Loads any HF checkpoint produced by `train_lora.py --config configs/nuscenes_planning_*.yaml`,
iterates the val infos, runs greedy `model.generate()` to produce trajectory
tokens, decodes to (Δx, Δy) waypoints, then computes:

  * L2 at 1 s (idx 1), 2 s (idx 3), 3 s (idx 5)
  * L2 average — under BOTH VAD's TemAvg protocol AND UniAD's NoAvg protocol
      - TemAvg (VAD): mean L2 over ALL future timesteps that fall within each
        cumulative horizon. For a 1 s horizon we average errors at t in {0.5, 1.0} s.
      - NoAvg  (UniAD): point-wise L2 at exactly t = 1/2/3 s.
  * Collision rate at 1 s, 2 s, 3 s — port of UniAD's footprint-overlap check:
    construct the ego footprint bbox at each predicted future waypoint, and
    check overlap against every annotated agent at that timestamp.

Output: a JSON file matching the AutoVLA/UniAD/VAD table format so it's drop-in
for paper comparison.

Notes:
  - This script runs on a SINGLE GPU. FSDP-sharded ckpts produced by `train_lora.py`
    with `train_mode: full_sft` are written as plain HF dirs (state_dict gathered
    on rank 0), so `AutoModelForImageTextToText.from_pretrained(ckpt)` just works.
  - Collisions: ground-truth agent boxes come from each future frame's
    `gt_boxes` (in current-frame ego coordinates per UniAD's transform).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from planning_dataset import (  # noqa: E402
    DEFAULT_PLANNING_CAMS,
    PROMPT_TEXT,
    PlanningDataset,
    _build_user_content_multicam,
    _format_ego_speed_preamble,
    quat_to_R,
)
from trajectory_tokenizer import (  # noqa: E402
    TrajectoryTokenizer,
    TrajectoryTokenizerConfig,
)


HZ = 2.0
DT = 1.0 / HZ              # 0.5 s
HORIZONS = (1.0, 2.0, 3.0)
HORIZON_IDX = (1, 3, 5)    # 0-indexed waypoint at each horizon (t = 1/2/3 s)
# Ego footprint — matches UniAD/VAD/ST-P3 exactly (Renault Zoe: 4.084 m length,
# 1.85 m width). Both papers use these values in their official planning-metric
# code (see UniAD planning_head_plugin/metric_stp3.py and VAD planner/metric_stp3.py).
EGO_LENGTH_M = 4.084
EGO_WIDTH_M = 1.85
EGO_HALF_LEN_M = EGO_LENGTH_M * 0.5
EGO_HALF_WID_M = EGO_WIDTH_M * 0.5
# nuScenes ego pose is reported at the rear-axle (lidar-top mount point), so
# the box CENTRE is +0.5 m forward of the pose origin along ego +x. Both UniAD
# and VAD shift the box by +0.5 m forward to match this (`[-H/2 + 0.5, ...]`
# in their code). We replicate that shift here.
EGO_BOX_FWD_OFFSET_M = 0.5


# ============================================================================
# Helpers: decoding the model output
# ============================================================================

def _find_traj_block(token_ids: List[int], traj_start_id: int, traj_end_id: int) -> List[int]:
    """Extract the bin tokens between <traj_start> and <traj_end>."""
    try:
        i0 = token_ids.index(traj_start_id)
    except ValueError:
        return []
    try:
        i1 = token_ids.index(traj_end_id, i0 + 1)
    except ValueError:
        i1 = len(token_ids)
    return token_ids[i0:i1 + 1]


def decode_waypoints(generated_ids: List[int], traj_tok: TrajectoryTokenizer,
                     num_waypoints: int) -> np.ndarray:
    """Return (num_waypoints, 2) of decoded (dx, dy) in metres. Pads zeros if
    generation produced fewer."""
    block = _find_traj_block(
        generated_ids, traj_tok.cfg.traj_start_id, traj_tok.cfg.traj_end_id
    )
    if block:
        wp = traj_tok.decode(block)
    else:
        # No boundary tokens at all -> fall back to "raw" decode over the whole
        # generation, which the tokenizer accepts.
        wp = traj_tok.decode(generated_ids)
    out = np.zeros((num_waypoints, 2), dtype=np.float32)
    n = min(num_waypoints, wp.shape[0])
    if n > 0:
        out[:n] = wp[:n]
    return out


# ============================================================================
# Helpers: agent boxes in current ego frame
# ============================================================================

def _gt_boxes_in_ego(future_info: dict, cur_info: dict
                     ) -> List[Tuple[float, float, float, float, float]]:
    """Return list of (x, y, length, width, yaw_rad) for each annotated agent in
    `future_info`, transformed from the future LiDAR frame back into the
    *current* ego frame.

    UniAD's nuscenes_converter stores `gt_boxes` as Nx7 = (x, y, z, w, l, h, yaw)
    in the **LiDAR** frame at that keyframe (NOT the ego frame — verified on
    nuscenes_infos_temporal_val.pkl 2026-05-19). The LiDAR sensor is mounted
    rotated ~90 deg about z relative to ego, so we MUST compose lidar2ego on
    both sides of the round-trip:

        p_global       = R_e2g_f @ (R_le_f @ p_lidar + t_le_f) + t_e2g_f
        p_in_cur_ego   = R_e2g_c.T @ (p_global - t_e2g_c)

    Yaw transforms via the same chain — the box yaw is measured in the future
    LiDAR frame, so we add the (future-LiDAR -> global) - (current-ego ->
    global) heading delta (this lands the yaw in the current-ego frame).

    Without the lidar2ego step, every agent box was rotated ~90 deg around the
    ego origin in our frame, which placed agents directly on the predicted
    trajectory and inflated the collision rate ~25x relative to the paper
    baselines (~24% -> <1%).
    """
    boxes_lidar = np.asarray(future_info["gt_boxes"], dtype=np.float64)
    if boxes_lidar.size == 0:
        return []
    # Future frame: lidar->ego, ego->global.
    R_le_f = quat_to_R(future_info["lidar2ego_rotation"])
    t_le_f = np.asarray(future_info["lidar2ego_translation"], dtype=np.float64)
    R_e2g_f = quat_to_R(future_info["ego2global_rotation"])
    t_e2g_f = np.asarray(future_info["ego2global_translation"], dtype=np.float64)
    # Current frame: ego->global. (lidar->ego at current isn't needed because
    # we transform agent positions all the way back into the current ego
    # frame; only the future-frame lidar->ego matters for the box's intrinsic
    # yaw, see yaw_offset below.)
    R_e2g_c = quat_to_R(cur_info["ego2global_rotation"])
    t_e2g_c = np.asarray(cur_info["ego2global_translation"], dtype=np.float64)

    # Heading of each pose (yaw about z) — use atan2 of the first column.
    def _heading(R: np.ndarray) -> float:
        return math.atan2(R[1, 0], R[0, 0])

    # Box yaw is in the *future-LiDAR* frame. The compound rotation
    # future-LiDAR -> current-ego is R_e2g_c.T @ R_e2g_f @ R_le_f; for
    # rotations about z the heading of that compound equals
    #   heading(R_e2g_f @ R_le_f) - heading(R_e2g_c).
    # Add that to each box's yaw_box to get its yaw in the current-ego frame.
    yaw_offset = _heading(R_e2g_f @ R_le_f) - _heading(R_e2g_c)

    out: List[Tuple[float, float, float, float, float]] = []
    for b in boxes_lidar:
        x_lidar, y_lidar, z_lidar = float(b[0]), float(b[1]), float(b[2])
        # nuScenes box dims order: (w, l, h). w = width (lateral),
        # l = length (along-yaw).
        w_box, l_box = float(b[3]), float(b[4])
        yaw_box = float(b[6])
        p_lidar = np.array([x_lidar, y_lidar, z_lidar], dtype=np.float64)
        # lidar @ future -> ego @ future -> global -> ego @ current.
        p_ego_fut = R_le_f @ p_lidar + t_le_f
        p_global = R_e2g_f @ p_ego_fut + t_e2g_f
        p_cur_ego = R_e2g_c.T @ (p_global - t_e2g_c)
        out.append((float(p_cur_ego[0]), float(p_cur_ego[1]),
                    l_box, w_box, yaw_box + yaw_offset))
    return out


def _box_corners(cx: float, cy: float, length: float, width: float, yaw: float
                 ) -> np.ndarray:
    hl, hw = length * 0.5, width * 0.5
    c, s = math.cos(yaw), math.sin(yaw)
    pts = np.array([[+hl, +hw], [+hl, -hw], [-hl, -hw], [-hl, +hw]], dtype=np.float64)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    return (pts @ R.T) + np.array([cx, cy])


def _sat_overlap(corners_a: np.ndarray, corners_b: np.ndarray) -> bool:
    """Separating-axis theorem for 2 convex quads. Returns True on overlap."""
    for poly in (corners_a, corners_b):
        for i in range(poly.shape[0]):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % poly.shape[0]]
            ax, ay = -(y2 - y1), x2 - x1
            n = math.hypot(ax, ay)
            if n < 1e-9:
                continue
            ax, ay = ax / n, ay / n
            pa = corners_a @ np.array([ax, ay])
            pb = corners_b @ np.array([ax, ay])
            if pa.max() < pb.min() - 1e-6 or pb.max() < pa.min() - 1e-6:
                return False
    return True


def collision_at_step(ego_xy: np.ndarray, ego_yaw: float,
                      gt_boxes_cur_ego: List[Tuple[float, float, float, float, float]],
                      ) -> bool:
    """Returns True if the ego footprint at (ego_xy, ego_yaw) overlaps any
    annotated agent.

    The footprint centre is shifted +EGO_BOX_FWD_OFFSET_M forward of the pose
    origin to match the UniAD/VAD convention (nuScenes ego pose is at the
    rear-axle/lidar mount, not the box centre)."""
    cx = float(ego_xy[0]) + EGO_BOX_FWD_OFFSET_M * math.cos(ego_yaw)
    cy = float(ego_xy[1]) + EGO_BOX_FWD_OFFSET_M * math.sin(ego_yaw)
    ego_corners = _box_corners(
        cx, cy, EGO_LENGTH_M, EGO_WIDTH_M, ego_yaw,
    )
    for (ax, ay, l, w, yaw) in gt_boxes_cur_ego:
        agent_corners = _box_corners(ax, ay, l, w, yaw)
        if _sat_overlap(ego_corners, agent_corners):
            return True
    return False


def _yaw_from_traj(wp: np.ndarray, step_idx: int) -> float:
    """Estimate ego heading at waypoint step_idx by finite difference."""
    if step_idx == 0:
        dx = wp[0, 0]
        dy = wp[0, 1]
    else:
        dx = wp[step_idx, 0] - wp[step_idx - 1, 0]
        dy = wp[step_idx, 1] - wp[step_idx - 1, 1]
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return 0.0
    return math.atan2(dy, dx)


# ============================================================================
# L2 protocols
# ============================================================================

def l2_temavg(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> Dict[str, float]:
    """VAD-style TemAvg: average error over all timesteps within horizon.
    For 1 s horizon we average idx 0..1; for 2 s -> 0..3; for 3 s -> 0..5."""
    out: Dict[str, float] = {}
    for horizon_idx, horizon_s in zip(HORIZON_IDX, HORIZONS):
        sl = slice(0, horizon_idx + 1)
        diff = pred[sl] - gt[sl]
        l2 = np.sqrt((diff ** 2).sum(axis=-1))
        m = valid[sl]
        if m.sum() < 1e-6:
            out[f"L2_{int(horizon_s)}s"] = float("nan")
        else:
            out[f"L2_{int(horizon_s)}s"] = float((l2 * m).sum() / m.sum())
    vals = [out[k] for k in ["L2_1s", "L2_2s", "L2_3s"] if not math.isnan(out[k])]
    out["L2_avg"] = float(np.mean(vals)) if vals else float("nan")
    return out


def l2_noavg(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> Dict[str, float]:
    """UniAD-style NoAvg: point-wise L2 at exactly t=1/2/3 s."""
    out: Dict[str, float] = {}
    for horizon_idx, horizon_s in zip(HORIZON_IDX, HORIZONS):
        if valid[horizon_idx] < 1e-6:
            out[f"L2_{int(horizon_s)}s"] = float("nan")
            continue
        diff = pred[horizon_idx] - gt[horizon_idx]
        out[f"L2_{int(horizon_s)}s"] = float(math.hypot(*diff))
    vals = [out[k] for k in ["L2_1s", "L2_2s", "L2_3s"] if not math.isnan(out[k])]
    out["L2_avg"] = float(np.mean(vals)) if vals else float("nan")
    return out


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="HF model checkpoint dir")
    p.add_argument("--infos-val", required=True, help="path to nuscenes_infos_temporal_val.pkl")
    p.add_argument("--nusc-root", default=os.path.join(_BASE_DIR, "data", "nuscenes"))
    p.add_argument("--max-samples", type=int, default=None,
                   help="Cap eval to N samples (default: all val)")
    p.add_argument("--output", default=None, help="Output JSON path (defaults to <ckpt>/eval_results.json)")
    p.add_argument("--num-past-frames", type=int, default=4)
    p.add_argument("--num-future-waypoints", type=int, default=6)
    p.add_argument("--video-fps", type=float, default=2.0)
    p.add_argument(
        "--planning-cams",
        default="CAM_FRONT",
        help="Comma-separated cam list, e.g. 'CAM_FRONT,CAM_FRONT_LEFT,CAM_FRONT_RIGHT' "
             "(AutoVLA 3-cam). Must match training config.",
    )
    p.add_argument("--max-new-tokens", type=int, default=20,
                   help="Greedy generate budget; 1 start + 12 bins + 1 end is enough.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="bfloat16")
    args = p.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    print(f"[planning_eval] loading model from {args.ckpt}")
    model = AutoModelForImageTextToText.from_pretrained(
        args.ckpt, torch_dtype=dtype, attn_implementation="sdpa",
    ).to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.ckpt)

    traj_cfg = TrajectoryTokenizerConfig(num_waypoints=args.num_future_waypoints)
    traj_tok = TrajectoryTokenizer(traj_cfg)

    planning_cams = [c.strip() for c in args.planning_cams.split(",") if c.strip()]
    # Multi-cam expands visual tokens ~Nx; raise the eval max_length to match
    # the 3-cam training config (8192). Single-cam keeps 4096 for back-compat.
    eval_max_length = 4096 if len(planning_cams) == 1 else 8192
    ds = PlanningDataset(
        infos_path=args.infos_val,
        nusc_root=args.nusc_root,
        processor=processor,
        max_length=eval_max_length,
        num_past_frames=args.num_past_frames,
        num_future_waypoints=args.num_future_waypoints,
        video_fps=args.video_fps,
        vla_loss_mode="answer_and_traj",
        max_samples=args.max_samples,
        require_full_future=True,  # only score samples with full 3 s of future
        planning_cams=planning_cams,
        require_all_cams=True,
    )

    # Build the *generation prompt* (NO appended action tokens). We rebuild it
    # here rather than using ds.__getitem__ because we want generation, not
    # teacher-forcing input. The user text is per-sample because we prepend the
    # ego-speed preamble (read from info["can_bus"][13]) — must match training.

    n_total = len(ds)
    print(f"[planning_eval] val samples: {n_total}")
    if n_total == 0:
        raise RuntimeError("Empty val set after require_full_future filter.")

    # Per-sample stats
    temavg_acc: Dict[str, List[float]] = {k: [] for k in ["L2_1s", "L2_2s", "L2_3s", "L2_avg"]}
    noavg_acc: Dict[str, List[float]] = {k: [] for k in ["L2_1s", "L2_2s", "L2_3s", "L2_avg"]}
    coll: Dict[str, List[int]] = {k: [] for k in ["collision_1s", "collision_2s", "collision_3s", "collision_avg"]}

    t0 = time.time()
    with torch.inference_mode():
        for i in range(n_total):
            sample = ds[i]
            gt_wp = sample["_meta_waypoints"].cpu().numpy()  # (6, 2)
            valid = sample["_meta_valid_mask"].cpu().numpy()
            token = sample["_meta_token"]

            # Build *generation* prompt (chat template with add_generation_prompt=True)
            base_idx = ds._keep[i]
            info = ds.infos[base_idx]
            hist = ds._walk_history(base_idx)
            # Single- or multi-cam clip loading; mirrors training dataset.
            if len(planning_cams) == 1:
                clips = [ds._load_frames(hist, planning_cams[0])]
            else:
                clips = ds._load_frames_multicam(hist)
            user_content = _build_user_content_multicam(info, planning_cams)
            sys_user_messages = [{"role": "user", "content": user_content}]
            text = processor.apply_chat_template(
                sys_user_messages, tokenize=False, add_generation_prompt=True
            )
            from transformers.video_utils import VideoMetadata
            md = [
                VideoMetadata(
                    total_num_frames=len(clip),
                    fps=args.video_fps,
                    frames_indices=list(range(len(clip))),
                    height=clip[0].height,
                    width=clip[0].width,
                )
                for clip in clips
            ]
            inputs = processor(
                text=[text], videos=clips, video_metadata=md, return_tensors="pt"
            ).to(device)
            # Cast video pixels to model dtype
            if "pixel_values_videos" in inputs:
                inputs["pixel_values_videos"] = inputs["pixel_values_videos"].to(dtype)

            gen = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=processor.tokenizer.pad_token_id or 0,
            )
            new_ids = gen[0, inputs["input_ids"].shape[1]:].tolist()
            pred_wp = decode_waypoints(new_ids, traj_tok, args.num_future_waypoints)

            # L2 protocols
            t = l2_temavg(pred_wp, gt_wp, valid)
            for k in temavg_acc:
                if not math.isnan(t[k]):
                    temavg_acc[k].append(t[k])
            n = l2_noavg(pred_wp, gt_wp, valid)
            for k in noavg_acc:
                if not math.isnan(n[k]):
                    noavg_acc[k].append(n[k])

            # Collision check: ego footprint at each horizon step.
            # _gt_boxes_in_ego reads `cur_info` for both ego2global AND
            # lidar2ego transforms (the LiDAR mount is rotated ~90 deg from
            # the ego frame on nuScenes, so we cannot skip lidar2ego).
            future_infos = ds._walk_future(base_idx)
            collisions_per_horizon: List[int] = []
            for h_idx, h_s in zip(HORIZON_IDX, HORIZONS):
                if h_idx >= len(future_infos) or valid[h_idx] < 1e-6:
                    collisions_per_horizon.append(0)
                    continue
                gt_boxes = _gt_boxes_in_ego(future_infos[h_idx], info)
                yaw_pred = _yaw_from_traj(pred_wp, h_idx)
                hit = collision_at_step(pred_wp[h_idx], yaw_pred, gt_boxes)
                collisions_per_horizon.append(int(hit))
            coll["collision_1s"].append(collisions_per_horizon[0])
            coll["collision_2s"].append(collisions_per_horizon[1])
            coll["collision_3s"].append(collisions_per_horizon[2])
            # Avg per-sample collision (any horizon hit -> count it as 1)
            coll["collision_avg"].append(int(any(collisions_per_horizon)))

            if (i + 1) % 50 == 0:
                rate = (i + 1) / max(time.time() - t0, 1e-6)
                eta = (n_total - i - 1) / max(rate, 1e-6)
                print(f"  [{i + 1}/{n_total}] {rate:.2f} sample/s | ETA {eta / 60:.1f} min")

    # Aggregate
    def _mean(xs: List[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    results = {
        "ckpt": os.path.abspath(args.ckpt),
        "infos_val": os.path.abspath(args.infos_val),
        "n_samples": n_total,
        "horizon_s": list(HORIZONS),
        "TemAvg": {k: _mean(v) for k, v in temavg_acc.items()},
        "NoAvg": {k: _mean(v) for k, v in noavg_acc.items()},
        "collision_rate": {
            "collision_1s": _mean([float(x) for x in coll["collision_1s"]]),
            "collision_2s": _mean([float(x) for x in coll["collision_2s"]]),
            "collision_3s": _mean([float(x) for x in coll["collision_3s"]]),
            "collision_avg": _mean([float(x) for x in coll["collision_avg"]]),
        },
        # Flat shortcut keys matching the table format requested in the spec.
        "L2_1s": _mean(temavg_acc["L2_1s"]),
        "L2_2s": _mean(temavg_acc["L2_2s"]),
        "L2_3s": _mean(temavg_acc["L2_3s"]),
        "L2_avg": _mean(temavg_acc["L2_avg"]),
        "collision_1s": _mean([float(x) for x in coll["collision_1s"]]),
        "collision_2s": _mean([float(x) for x in coll["collision_2s"]]),
        "collision_3s": _mean([float(x) for x in coll["collision_3s"]]),
        "collision_avg": _mean([float(x) for x in coll["collision_avg"]]),
        "protocol_l2": "TemAvg (VAD) shown in flat L2_*; full both protocols inside this JSON",
        "ego_footprint_m": {
            "length": EGO_LENGTH_M,
            "width": EGO_WIDTH_M,
            "half_length": EGO_HALF_LEN_M,
            "half_width": EGO_HALF_WID_M,
            "fwd_offset_from_pose": EGO_BOX_FWD_OFFSET_M,
        },
    }

    out_path = args.output or os.path.join(args.ckpt, "eval_results.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[planning_eval] wrote {out_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
