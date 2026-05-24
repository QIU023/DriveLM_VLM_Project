"""Scan val set, score each sample on speed / turn / complexity, pick top-N
samples per category for demo. Lightweight — no model loading.

Categories:
  HIGH_SPEED:  speed > 10 m/s (~36 km/h)
  TURN:        max |lateral drift| > 3 m AND speed > 5 m/s
  COMPLEX:     >= 12 detected objects within 50m
  REVERSE:     final dx < -1.0 m  (rare; pulling out / parking)
  STATIC:      final |dx,dy| < 0.5 m (the baseline we already have)

Output: deploy/multimodal_trt/_val_scenarios.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_BASE = _HERE.parent.parent
sys.path.insert(0, str(_BASE / "scripts"))


def main():
    from transformers import AutoProcessor
    from multimodal_planning_dataset import MultiModalPlanningDataset

    ckpt = str(_BASE / "checkpoints_qwen25/nusc_planning_b5pp_1cam_qwen3vl_multimodal/final")
    print(f"[scan] loading processor")
    proc = AutoProcessor.from_pretrained(ckpt)

    print(f"[scan] building dataset (no model)")
    ds = MultiModalPlanningDataset(
        infos_path=str(_BASE / "data/uniad_infos/nuscenes_infos_temporal_val.pkl"),
        nusc_root=str(_BASE / "data/nuscenes"),
        processor=proc, max_length=12288,
        num_past_frames=4, num_future_waypoints=6, video_fps=2.0,
        vla_loss_mode="answer_and_traj",
        require_full_future=True,
        planning_cams=["CAM_FRONT"], require_all_cams=True,
        hdmap_dir=str(_BASE / "data/preproc/hdmap_bev"),
        bbox_jsonl=str(_BASE / "data/preproc/bbox_egostate_val.jsonl"),
        split="val", modality_dropout_p=0.0,
    )
    N = len(ds._keep)
    print(f"[scan] {N} val samples")

    scores = []
    for i in range(N):
        base_idx = ds._keep[i]
        info = ds.infos[base_idx]
        sample_token = info["token"]
        wp, valid = ds._compute_waypoints(base_idx)
        if wp is None or len(wp) < 6:
            continue
        # speed: distance from origin to last waypoint / 3s horizon
        dist_final = float(np.linalg.norm(wp[-1]))
        speed_mps = dist_final / 3.0
        # max lateral drift (dy)
        max_dy = float(np.max(np.abs(wp[:, 1])))
        # max forward
        max_dx_abs = float(np.max(np.abs(wp[:, 0])))
        # bbox count
        bbox_text = ds._lookup_bbox(sample_token) or ""
        n_objects = sum(1 for ln in bbox_text.split("\n") if ln.strip().startswith("-"))
        scores.append(dict(
            idx=i, sample_token=sample_token,
            speed_mps=speed_mps, max_dx_abs=max_dx_abs, max_dy=max_dy,
            n_objects=n_objects, wp_final=wp[-1].tolist(),
        ))

    print(f"[scan] scored {len(scores)} samples")

    # Categorize
    high_speed = sorted([s for s in scores if s["speed_mps"] > 10], key=lambda x: -x["speed_mps"])
    turn = sorted([s for s in scores if s["max_dy"] > 3.0 and s["speed_mps"] > 5],
                  key=lambda x: -x["max_dy"])
    complex_traffic = sorted([s for s in scores if s["n_objects"] >= 12], key=lambda x: -x["n_objects"])
    reverse = sorted([s for s in scores if s["wp_final"][0] < -1.0], key=lambda x: x["wp_final"][0])
    static_baseline = sorted([s for s in scores if s["speed_mps"] < 0.3], key=lambda x: x["speed_mps"])

    out = {
        "n_total": N,
        "n_scored": len(scores),
        "categories": {
            "high_speed_top5": high_speed[:5],
            "turn_top5": turn[:5],
            "complex_traffic_top5": complex_traffic[:5],
            "reverse_top3": reverse[:3],
            "static_top3": static_baseline[:3],
        },
    }
    out_path = _HERE / "_val_scenarios.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n[scan] === TOP scenarios ===")
    for cat, lst in out["categories"].items():
        print(f"\n{cat}:")
        for s in lst:
            print(f"  idx={s['idx']:5d}  speed={s['speed_mps']:5.2f} m/s  max_dy={s['max_dy']:5.2f} m  "
                  f"max_dx={s['max_dx_abs']:5.2f} m  n_obj={s['n_objects']:3d}  wp_final={s['wp_final']}")
    print(f"\n[scan] saved → {out_path}")


if __name__ == "__main__":
    main()
