"""CPU-only smoke test for sub-scenario classifier + behavior metrics.

Builds 5 hand-crafted (gt, pred) waypoint pairs covering the buckets we care
about and asserts:

  * classify_scenario picks the expected bucket
  * behavior_metrics returns finite (non-NaN) values for non-degenerate cases
  * when pred == gt: heading_error < 0.01 rad, speed_error < 0.01 m/s, and
    progress_ratio ~ 1.0 (sanity check)

Run:
  /usr/bin/python3 scripts/_smoke_planning_metrics.py

Exit code 0 on PASS, 1 on any FAIL.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from planning_eval import (  # noqa: E402
    DT,
    HZ,
    behavior_metrics,
    classify_scenario,
)


def _full_valid(n: int) -> np.ndarray:
    return np.ones((n,), dtype=np.float32)


def _straight_traj(v_mps: float, n: int = 6) -> np.ndarray:
    """Pure-forward at constant velocity v in ego frame."""
    wp = np.zeros((n, 2), dtype=np.float32)
    for i in range(n):
        wp[i, 0] = v_mps * (i + 1) * DT
    return wp


def _turning_traj(v_mps: float, radius_m: float, n: int = 6) -> np.ndarray:
    """Constant-radius left turn. Heading change per step = (v*DT)/radius."""
    wp = np.zeros((n, 2), dtype=np.float32)
    omega = v_mps / radius_m  # rad/s
    # Integrate position along an arc centered at (0, +radius).
    for i in range(n):
        theta = omega * (i + 1) * DT
        wp[i, 0] = radius_m * math.sin(theta)
        wp[i, 1] = radius_m * (1.0 - math.cos(theta))
    return wp


def _lane_change_traj(forward_v_mps: float, lateral_m: float,
                      n: int = 6) -> np.ndarray:
    """Forward + smooth lateral offset, ending with heading ~= 0 at horizon."""
    wp = np.zeros((n, 2), dtype=np.float32)
    for i in range(n):
        s = (i + 1) / float(n)
        wp[i, 0] = forward_v_mps * (i + 1) * DT
        # Smooth s-curve via half-cosine: lateral grows then plateaus.
        wp[i, 1] = lateral_m * 0.5 * (1.0 - math.cos(math.pi * s))
    return wp


def _braking_traj(v0_mps: float, decel_mps2: float, n: int = 6) -> np.ndarray:
    """Forward deceleration: v(t) = max(0, v0 - decel * t)."""
    wp = np.zeros((n, 2), dtype=np.float32)
    x = 0.0
    v = v0_mps
    for i in range(n):
        v = max(0.0, v - decel_mps2 * DT)
        # Trapezoidal step (using mean of v_prev and v_now) — close enough.
        v_step = max(0.0, v + decel_mps2 * DT * 0.5)
        x += v_step * DT
        wp[i, 0] = x
    return wp


def _stationary_traj(n: int = 6) -> np.ndarray:
    return np.zeros((n, 2), dtype=np.float32)


def _check(label: str, cond: bool, detail: str) -> bool:
    flag = "PASS" if cond else "FAIL"
    print(f"  [{flag}] {label}: {detail}")
    return cond


def main() -> int:
    print(f"# CPU-only smoke. HZ={HZ}, DT={DT}\n")

    cases = [
        # (name, gt_wp, expected_bucket)
        ("straight",   _straight_traj(8.0),                  "straight"),
        ("turning",    _turning_traj(5.0, 12.0),             "turning"),
        ("lane_change", _lane_change_traj(6.0, 3.0),         "lane_change"),
        ("braking",    _braking_traj(8.0, 1.5),              "braking"),
        ("stationary", _stationary_traj(),                   "stationary"),
    ]

    all_ok = True

    # ---- Classifier checks ----
    print("## Sub-scenario classifier")
    for name, gt, expected in cases:
        got = classify_scenario(gt, _full_valid(gt.shape[0]))
        ok = (got == expected)
        all_ok &= _check(f"classify[{name}]", ok, f"expected={expected!r}, got={got!r}")

    # ---- Behavior metrics (pred = gt) — should be ~zero errors ----
    print("\n## Behavior metrics, pred == gt (parity check)")
    for name, gt, _ in cases:
        valid = _full_valid(gt.shape[0])
        beh = behavior_metrics(gt.copy(), gt.copy(), valid)
        for k, v in beh.items():
            if isinstance(v, float) and math.isnan(v):
                # Only legal for stationary (progress_ratio: gt total dist ~=0)
                # and stationary-ish lateral_accel for straight (kappa=0).
                if name == "stationary":
                    continue
                # progress_ratio NaN for stationary only is OK; anything else NaN
                # in a non-stationary scenario is a bug.
                all_ok &= _check(
                    f"behavior[{name}][{k}] non-NaN", False,
                    f"got NaN for non-stationary scenario {name}",
                )
        # Headline parity checks (skip stationary which has trivial 0 path):
        if name != "stationary":
            all_ok &= _check(
                f"behavior[{name}].heading_error_rad < 0.01",
                beh["heading_error_rad"] < 0.01,
                f"got {beh['heading_error_rad']:.5f}",
            )
            all_ok &= _check(
                f"behavior[{name}].speed_error_m_s < 0.01",
                beh["speed_error_m_s"] < 0.01,
                f"got {beh['speed_error_m_s']:.5f}",
            )
            pr = beh["progress_ratio"]
            all_ok &= _check(
                f"behavior[{name}].progress_ratio ~= 1.0",
                abs(pr - 1.0) < 0.01,
                f"got {pr:.4f}",
            )
            # hard_brake_rate should be 0 for parity, EXCEPT braking with the
            # configured decel of 1.5 m/s^2 < threshold(3.0) — so 0 expected.
            all_ok &= _check(
                f"behavior[{name}].hard_brake == 0",
                beh["hard_brake"] == 0,
                f"got {beh['hard_brake']}",
            )

    # ---- Behavior metrics, perturbed pred — should be finite, non-zero ----
    print("\n## Behavior metrics, perturbed pred (sanity, no NaN)")
    for name, gt, _ in cases:
        # Perturb pred by small offset in x — speed_error / heading_error nonzero
        pred = gt.copy()
        pred[:, 0] += 0.5  # constant +0.5m offset (changes step distances)
        beh = behavior_metrics(pred, gt, _full_valid(gt.shape[0]))
        # All metrics finite or NaN-but-legal
        for k, v in beh.items():
            if isinstance(v, float):
                ok = math.isfinite(v) or (name == "stationary" and k in ("progress_ratio", "heading_error_rad"))
                all_ok &= _check(
                    f"behavior[{name}][{k}] finite-or-stationary-NaN",
                    ok,
                    f"got {v}",
                )

    # ---- Synthetic hard-brake case ----
    print("\n## Hard brake detection (predicted decel > 3 m/s^2)")
    gt = _straight_traj(8.0)
    pred = _braking_traj(8.0, 4.0)  # decel of 4 m/s^2 > threshold 3
    beh = behavior_metrics(pred, gt, _full_valid(gt.shape[0]))
    all_ok &= _check(
        "hard_brake fires when decel > 3 m/s^2",
        beh["hard_brake"] == 1,
        f"got hard_brake={beh['hard_brake']} (max decel ~ 4 m/s^2 expected)",
    )

    # ---- Sample JSON dump (one scenario) ----
    print("\n## Sample behavior JSON (case=lane_change, pred~=gt)")
    gt = _lane_change_traj(6.0, 3.0)
    beh = behavior_metrics(gt.copy(), gt.copy(), _full_valid(gt.shape[0]))
    print(json.dumps({
        "scenario": classify_scenario(gt, _full_valid(gt.shape[0])),
        "behavior": beh,
    }, indent=2))

    print("\n" + ("ALL PASS" if all_ok else "SOME FAIL"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
