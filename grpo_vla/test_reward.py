"""Unit-test the GRPO reward function on 10 real nuScenes val samples.

We deliberately bypass the heavy Qwen processor + VeRLNuScenesDataset
because the reward function takes only (token_ids, gt_wp, bboxes,
ego_state) — none of which need the VLM. Loading the val infos pkl +
bbox_egostate_val.jsonl is enough.

For each of 10 samples we fabricate 3 candidate responses:
  - perfect    : tokenize the GT waypoints   -> token_ids
  - static     : tokenize all-zeros waypoints -> token_ids
  - wrong_dir  : tokenize negated GT waypoints -> token_ids

The reward sanity gate is:
    r_total[perfect] > r_total[static] > r_total[wrong_dir]

for every one of the 10 samples. If any sample violates the ordering
we print which dim broke the sign and exit nonzero.

Run with: /usr/bin/python3 grpo_vla/test_reward.py
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from typing import Dict, List, Tuple

import numpy as np

# Make grpo_vla/ importable when this script is invoked directly.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# Also expose scripts/ for the trajectory tokenizer.
_REPO_ROOT = os.path.dirname(_THIS_DIR)
_SCRIPTS_DIR = os.path.join(_REPO_ROOT, "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from reward import (  # noqa: E402
    compute_reward,
    parse_bbox_text,
    MALFORMED_REWARD,
)
from trajectory_tokenizer import (  # noqa: E402
    TrajectoryTokenizer,
    TrajectoryTokenizerConfig,
)
from planning_dataset import CAN_BUS_SPEED_IDX  # noqa: E402


VAL_INFOS_PATH = os.path.join(
    _REPO_ROOT, "data/uniad_infos/nuscenes_infos_temporal_val.pkl"
)
VAL_BBOX_JSONL = os.path.join(
    _REPO_ROOT, "data/preproc/bbox_egostate_val.jsonl"
)
N_SAMPLES = 10
NUM_FUTURE = 6


def load_bbox_index(path: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            tok = row.get("sample_token")
            if tok is not None:
                out[tok] = row.get("bbox_text", "")
    return out


def compute_waypoints(infos: List[dict], tok2idx: Dict[str, int],
                      base_idx: int, num_future: int) -> Tuple[np.ndarray, np.ndarray]:
    """Replicates PlanningDataset._compute_waypoints (current-ego-frame xy)."""
    # We reuse the project's util via a local import to stay aligned with
    # whatever convention scripts/planning_dataset uses. quat_to_R lives in
    # scripts/planning_dataset.
    from planning_dataset import quat_to_R  # noqa: E402

    cur = infos[base_idx]
    R_cur = quat_to_R(cur["ego2global_rotation"])
    p_cur = np.asarray(cur["ego2global_translation"], dtype=np.float64)
    R_cur_T = R_cur.T

    wp = np.zeros((num_future, 2), dtype=np.float32)
    mask = np.zeros((num_future,), dtype=np.float32)
    cur_walker = cur
    for t in range(num_future):
        nxt = cur_walker.get("next")
        if not nxt or nxt not in tok2idx:
            break
        cur_walker = infos[tok2idx[nxt]]
        p_f = np.asarray(cur_walker["ego2global_translation"], dtype=np.float64)
        local = R_cur_T @ (p_f - p_cur)
        wp[t, 0] = local[0]
        wp[t, 1] = local[1]
        mask[t] = 1.0
    return wp, mask


def fabricate_candidates(traj_tok: TrajectoryTokenizer,
                         gt_wp: np.ndarray) -> Dict[str, List[int]]:
    """Build the 3 candidate response token id lists."""
    perfect_ids = traj_tok.encode(gt_wp, with_boundaries=True)
    static_ids = traj_tok.encode(
        np.zeros_like(gt_wp), with_boundaries=True
    )
    # Negate GT waypoints. Clamp dx into the tokenizer's asymmetric range
    # [-2, +40] m via the per-dim clip inside encode; -gt_wp will saturate
    # forward motion (dx > 2 m) at the -2 m forward floor, which is exactly
    # the "complete wrong-direction" signal we want.
    wrong_dir_ids = traj_tok.encode(-gt_wp, with_boundaries=True)
    return {
        "perfect": perfect_ids,
        "static": static_ids,
        "wrong_dir": wrong_dir_ids,
    }


def fmt_row(name: str, r: dict) -> str:
    return (
        f"  {name:10s}  total={r['r_total']:+8.4f}  "
        f"l2={r['r_l2']:+7.4f}  smooth={r['r_smooth']:+7.4f}  "
        f"coll={r['r_collision']:+5.2f}(n={r['n_collisions']})  "
        f"intent={r['r_intent']:+6.3f}  speed={r['r_speed']:+7.4f}  "
        f"malformed={int(r['malformed'])}"
    )


def main() -> int:
    if not os.path.exists(VAL_INFOS_PATH):
        print(f"FATAL: val infos pkl not found at {VAL_INFOS_PATH}", file=sys.stderr)
        return 2
    if not os.path.exists(VAL_BBOX_JSONL):
        print(f"FATAL: val bbox jsonl not found at {VAL_BBOX_JSONL}", file=sys.stderr)
        return 2

    print(f"[load] {VAL_INFOS_PATH}")
    with open(VAL_INFOS_PATH, "rb") as f:
        blob = pickle.load(f)
    infos = blob["infos"] if isinstance(blob, dict) else blob
    tok2idx = {info["token"]: i for i, info in enumerate(infos)}

    print(f"[load] {VAL_BBOX_JSONL}")
    bbox_index = load_bbox_index(VAL_BBOX_JSONL)

    # Pick the first N val samples that have a full future horizon.
    candidates: List[int] = []
    for i in range(len(infos)):
        cur = infos[i]
        ok = True
        walker = cur
        for _ in range(NUM_FUTURE):
            nxt = walker.get("next")
            if not nxt or nxt not in tok2idx:
                ok = False
                break
            walker = infos[tok2idx[nxt]]
        if ok:
            candidates.append(i)
        if len(candidates) >= N_SAMPLES:
            break
    if len(candidates) < N_SAMPLES:
        print(f"FATAL: only {len(candidates)} val samples have full future "
              f"horizon (need {N_SAMPLES}).", file=sys.stderr)
        return 2
    print(f"[select] {len(candidates)} val samples with full {NUM_FUTURE}-step future")

    traj_tok = TrajectoryTokenizer(TrajectoryTokenizerConfig())

    n_pass = 0
    n_fail = 0
    detail_printed = False
    failures: List[str] = []
    for n, base_idx in enumerate(candidates):
        info = infos[base_idx]
        sample_token = info["token"]
        gt_wp, valid_mask = compute_waypoints(infos, tok2idx, base_idx, NUM_FUTURE)
        if (valid_mask < 1.0).any():
            # Shouldn't happen — we pre-filtered. But be defensive.
            print(f"  [skip] {sample_token}: partial future (mask={valid_mask})")
            continue
        bbox_text = bbox_index.get(sample_token, "")
        bbox_3d_list = parse_bbox_text(bbox_text)
        try:
            speed = max(0.0, float(info["can_bus"][CAN_BUS_SPEED_IDX]))
        except (KeyError, TypeError, ValueError, IndexError):
            speed = 0.0
        ego_state = {"speed_mps": speed}

        cands = fabricate_candidates(traj_tok, gt_wp)
        results = {
            name: compute_reward(ids, gt_wp, bbox_3d_list, ego_state)
            for name, ids in cands.items()
        }

        rt_perfect = results["perfect"]["r_total"]
        rt_static = results["static"]["r_total"]
        rt_wrong = results["wrong_dir"]["r_total"]
        ok = (rt_perfect > rt_static) and (rt_static > rt_wrong)

        # Print one full breakdown table for inspection (the first sample).
        if not detail_printed:
            print()
            print(f"=== sample #{n} token={sample_token} ===")
            print(f"  gt_wp           = {np.round(gt_wp, 2).tolist()}")
            print(f"  ego_speed_mps   = {speed:.3f}")
            print(f"  n_bboxes_parsed = {len(bbox_3d_list)}")
            print()
            for name, r in results.items():
                print(fmt_row(name, r))
            print()
            detail_printed = True

        if ok:
            n_pass += 1
        else:
            n_fail += 1
            # Diagnose which dim is mis-signed.
            details: List[str] = []
            for dim in ("r_l2", "r_smooth", "r_collision", "r_intent", "r_speed"):
                p = results["perfect"][dim]
                s = results["static"][dim]
                w = results["wrong_dir"][dim]
                if not (p >= s >= w):
                    details.append(f"{dim}: perfect={p:+.4f} static={s:+.4f} "
                                   f"wrong={w:+.4f} (expected perfect>=static>=wrong)")
            failures.append(
                f"sample#{n} token={sample_token}  totals: "
                f"perfect={rt_perfect:+.4f} static={rt_static:+.4f} "
                f"wrong_dir={rt_wrong:+.4f}\n     "
                + ("\n     ".join(details) if details else "(no single dim out of order)")
            )

    print()
    print("=" * 80)
    print(f"SUMMARY: {n_pass}/{N_SAMPLES} PASS, {n_fail}/{N_SAMPLES} FAIL")
    if n_fail:
        print()
        print("FAILURE DETAILS:")
        for f in failures:
            print("  - " + f)
        print()
        print("RESULT: FAIL  (reward sanity gate violated -- do NOT ship)")
        return 1
    print("RESULT: PASS  (perfect > static > wrong_dir on all 10 samples)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
