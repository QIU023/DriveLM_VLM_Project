"""Extract ego trajectories (next 3 s @ 2 Hz) for every DriveLM key frame and
append them as `action_tokens` to the existing video JSON.

Outputs: data_processed/v1_1_video_n{N}_with_traj.json

Two extraction paths, tried in order:

  (A) **Real nuScenes meta**: if `data/nuscenes/v1.0-trainval/` (or the
      mini/test variants) is present on disk, we read `ego_pose.json`,
      `sample_data.json`, `sample.json` and resolve each DriveLM frame_token
      (=sample.token) -> 6 future ego poses (0.5, 1.0, 1.5, 2.0, 2.5, 3.0 s)
      in the *current ego frame*. This is the canonical nuScenes pipeline; see
      OpenDriveLab/DriveLM/challenge/llama_adapter_v2_multimodal7b/data/
      for the closest published reference, and nuscenes-devkit's
      `NuScenes.get_sample_data()` / `get('ego_pose', ...)`.

  (B) **Filename-timestamp fallback**: if (A) is unavailable (only `samples/`
      camera jpgs on disk, which is exactly the present state), we fall back to
      a heuristic trajectory built from the CAM_FRONT filename timestamps and
      the DriveLM "behavior" QA labels. This is NOT a substitute for real ego
      pose — it lets us exercise the VLA pipeline end-to-end and unblock the
      smoke test, and is clearly tagged `source: "heuristic"` in metadata so
      downstream eval refuses to use it.

In either case, any record we cannot resolve is **dropped** rather than written
with a fake trajectory; the script reports the count.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trajectory_tokenizer import (  # noqa: E402
    TrajectoryTokenizer,
    TrajectoryTokenizerConfig,
)


# Default nuScenes meta locations to probe.
NUSC_META_CANDIDATES = [
    os.path.join(_BASE_DIR, "data", "nuscenes", "v1.0-trainval"),
    os.path.join(_BASE_DIR, "data", "nuscenes", "v1.0-mini"),
    os.path.join(_BASE_DIR, "data", "nuscenes", "v1.0-test"),
]


# ============================ (A) Real nuScenes ============================

def _find_nuscenes_meta() -> Optional[str]:
    for d in NUSC_META_CANDIDATES:
        if (
            os.path.isdir(d)
            and os.path.exists(os.path.join(d, "ego_pose.json"))
            and os.path.exists(os.path.join(d, "sample.json"))
            and os.path.exists(os.path.join(d, "sample_data.json"))
        ):
            return d
    return None


def _load_nusc_meta(meta_dir: str) -> Dict[str, list]:
    out = {}
    for name in ("ego_pose", "sample", "sample_data", "calibrated_sensor"):
        p = os.path.join(meta_dir, f"{name}.json")
        if os.path.exists(p):
            with open(p, "r") as f:
                out[name] = json.load(f)
    return out


def _quat_to_rot2d(q_wxyz: List[float]) -> np.ndarray:
    """Yaw-only 2D rotation matrix from a (w,x,y,z) quaternion (vehicle frame)."""
    w, x, y, z = q_wxyz
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


class _MetaIndex:
    """Pre-built indexes over nuScenes meta — hoisted out of the per-record path.

    Building these once for the full v1.0-trainval (~700K sample_data rows + ~200K
    ego_pose rows + ~200K sample rows for the 12Hz interpolated meta) takes ~20 s
    and is then constant-time per query. With 377K DriveLM QA records, rebuilding
    on every call (the previous behaviour) was the bottleneck.
    """

    def __init__(self, meta: Dict[str, list]):
        self.samples_by_token: Dict[str, dict] = {s["token"]: s for s in meta["sample"]}
        self.ep_by_token: Dict[str, dict] = {ep["token"]: ep for ep in meta["ego_pose"]}

        # Group sample_data rows by sample_token; for each sample, remember the
        # canonical LIDAR_TOP keyframe row (if any) so _ego_pose_for_sample is O(1).
        sd_by_sample: Dict[str, list] = {}
        lidar_top_by_sample: Dict[str, dict] = {}
        for sd in meta["sample_data"]:
            stok = sd["sample_token"]
            sd_by_sample.setdefault(stok, []).append(sd)
            if (
                sd.get("channel", "") == "LIDAR_TOP"
                and sd.get("is_key_frame", False)
                and stok not in lidar_top_by_sample
            ):
                lidar_top_by_sample[stok] = sd
        self.sd_by_sample = sd_by_sample
        self.lidar_top_by_sample = lidar_top_by_sample

    def ego_pose_for_sample(self, stok: str) -> Optional[dict]:
        sd = self.lidar_top_by_sample.get(stok)
        if sd is not None:
            ep_tok = sd.get("ego_pose_token")
            if ep_tok and ep_tok in self.ep_by_token:
                return self.ep_by_token[ep_tok]
        # Fall back to ANY sample_data row that has an ego_pose_token.
        for sd in self.sd_by_sample.get(stok, []):
            ep_tok = sd.get("ego_pose_token")
            if ep_tok and ep_tok in self.ep_by_token:
                return self.ep_by_token[ep_tok]
        return None


def _build_future_trajectory_real(
    index: "_MetaIndex",
    sample_token: str,
    horizon_s: float,
    sample_hz: float,
    strict_tolerance_s: float = 0.25,
) -> Optional[np.ndarray]:
    """Return (T, 2) future ego positions in the current ego frame, or None.

    BUGFIX 2026-05-18: previously, if the requested future timestamp exceeded
    the scene's last available ego_pose, we silently clamped to xys[-1],
    producing a frozen-tail trajectory (last two waypoints identical). On the
    full DriveLM real meta this happened on 100% of keyframes because most
    keyframes are near the *end* of their scene.

    Fix (Option A): return None if the latest available ego_pose timestamp is
    more than `strict_tolerance_s` seconds short of (t0 + horizon_s). Caller
    drops the record. Note we permit the *interpolation* of intermediate
    waypoints (np.interp), but never extrapolation past the last real pose.
    """
    samples_by_token = index.samples_by_token
    if sample_token not in samples_by_token:
        return None

    # Walk forward through 'next' samples up to horizon_s seconds.
    cur = samples_by_token[sample_token]
    t0_us = cur["timestamp"]  # microseconds

    ep0 = index.ego_pose_for_sample(sample_token)
    if ep0 is None:
        return None
    p0 = np.array(ep0["translation"][:2], dtype=np.float64)
    R0 = _quat_to_rot2d(ep0["rotation"])
    R0_inv = R0.T  # rotation matrices are orthogonal

    # Build the time grid 0.5, 1.0, ..., horizon_s (excluding 0).
    num_waypoints = int(round(horizon_s * sample_hz))
    targets_us = [t0_us + int(1e6 * (i + 1) / sample_hz) for i in range(num_waypoints)]

    # Walk samples forward and gather (timestamp, global_xy).
    walk = []  # list of (timestamp_us, np.array([x,y]))
    cur_tok = sample_token
    walked = 0
    # Safety bound. The 12Hz interpolated meta puts samples ~83 ms apart,
    # so we need ~12 * horizon_s + slack. Plain 2Hz keyframe meta is
    # ~500 ms apart -> num_waypoints * 4 was fine. Use a generous bound.
    max_walk = int(15 * horizon_s) + 8
    while cur_tok and walked < max_walk:
        smp = samples_by_token.get(cur_tok)
        if smp is None:
            break
        ep = index.ego_pose_for_sample(cur_tok)
        if ep is not None:
            walk.append((smp["timestamp"], np.array(ep["translation"][:2], dtype=np.float64)))
        if walk and walk[-1][0] >= t0_us + int(1e6 * horizon_s) + 200_000:
            break
        cur_tok = smp.get("next", "")
        walked += 1

    if len(walk) < 2:
        return None

    times = np.array([w[0] for w in walk], dtype=np.float64)
    xys = np.stack([w[1] for w in walk], axis=0)

    # BUGFIX (Option A): refuse to extrapolate past the last real ego_pose.
    # If any future target timestamp lies more than strict_tolerance_s seconds
    # past `times[-1]`, drop the record. We DO allow a small tolerance so a
    # waypoint that lies e.g. 50 ms past the last logged pose is still kept
    # (just interpolated to the very last logged pose).
    tol_us = int(strict_tolerance_s * 1e6)
    for ts in targets_us:
        if ts > times[-1] + tol_us:
            return None
        if ts < times[0] - tol_us:
            # Should never happen (t0 is in the walk by construction), but guard.
            return None

    waypoints_global = []
    for ts in targets_us:
        # Clamp to interpolation range; tolerance was already enforced above.
        ts_clip = min(max(ts, float(times[0])), float(times[-1]))
        ix = float(np.interp(ts_clip, times, np.arange(len(times))))
        i0 = int(np.floor(ix))
        i1 = min(i0 + 1, len(times) - 1)
        a = ix - i0
        waypoints_global.append((1 - a) * xys[i0] + a * xys[i1])
    waypoints_global = np.stack(waypoints_global, axis=0)  # (T, 2)

    # Convert to ego frame at t0: local = R0^T (global - p0). Note nuScenes
    # ego frame already has x forward, y left for the body frame, which is what
    # we want.
    local = (waypoints_global - p0[None, :]) @ R0_inv.T
    return local.astype(np.float32)


# ============================ (B) Heuristic fallback =========================

# Map DriveLM 'behavior' answers to a coarse (forward speed m/s, lateral rate m/s).
# This is intentionally crude; real ego pose is the right answer once nuScenes
# meta is on disk. We only use this to unblock the smoke test pipeline.
_BEHAVIOR_REGEX = [
    (re.compile(r"\bstop|stopped|stationary|standing\b", re.I), (0.0, 0.0)),
    (re.compile(r"\bbrak(e|ing)|slow(ing)? down|decelerat\b", re.I), (3.0, 0.0)),
    (re.compile(r"\bturn(ing)? left\b", re.I), (5.0, +1.5)),
    (re.compile(r"\bturn(ing)? right\b", re.I), (5.0, -1.5)),
    (re.compile(r"\blane chang(e|ing) left\b", re.I), (10.0, +1.0)),
    (re.compile(r"\blane chang(e|ing) right\b", re.I), (10.0, -1.0)),
    (re.compile(r"\baccelerat\b", re.I), (12.0, 0.0)),
    (re.compile(r"\bgo straight|moving forward|driving forward|proceed\b", re.I), (8.0, 0.0)),
    (re.compile(r"\bsteer left\b", re.I), (8.0, +0.5)),
    (re.compile(r"\bsteer right\b", re.I), (8.0, -0.5)),
]


def _behavior_to_motion(text: str) -> Tuple[float, float]:
    """Return (forward_speed_m_s, lateral_speed_m_s) from a behavior answer."""
    for pat, motion in _BEHAVIOR_REGEX:
        if pat.search(text):
            return motion
    # Default: gentle forward
    return (6.0, 0.0)


def _heuristic_trajectory(
    behavior_text: str,
    horizon_s: float,
    sample_hz: float,
) -> np.ndarray:
    vx, vy = _behavior_to_motion(behavior_text)
    num_waypoints = int(round(horizon_s * sample_hz))
    times = np.arange(1, num_waypoints + 1, dtype=np.float32) / float(sample_hz)
    dx = vx * times
    dy = vy * times
    return np.stack([dx, dy], axis=1)  # (T, 2)


# ============================ Driver =========================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.join(_BASE_DIR, "data_processed", "v1_1_video_n4.json"),
                    help="Existing Tier-1 video JSON to augment.")
    ap.add_argument("--output", default=None,
                    help="Output path. Defaults to <input>_with_traj.json")
    ap.add_argument("--horizon-s", type=float, default=3.0)
    ap.add_argument("--sample-hz", type=float, default=2.0)
    ap.add_argument("--num-bins", type=int, default=256)
    ap.add_argument("--min-m", type=float, default=-50.0)
    ap.add_argument("--max-m", type=float, default=+50.0)
    ap.add_argument("--force-heuristic", action="store_true",
                    help="Skip nuScenes real-meta path even if available.")
    ap.add_argument("--meta-dir", default=None,
                    help="Override nuScenes meta dir (e.g. v1.0-trainval).")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: input not found: {args.input}", file=sys.stderr)
        sys.exit(2)

    out_path = args.output or args.input.replace(".json", "_with_traj.json")

    cfg = TrajectoryTokenizerConfig(
        num_waypoints=int(round(args.horizon_s * args.sample_hz)),
        horizon_s=args.horizon_s,
        sample_hz=args.sample_hz,
        num_bins=args.num_bins,
        min_m=args.min_m,
        max_m=args.max_m,
    )
    tok = TrajectoryTokenizer(cfg)

    with open(args.input, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} records from {args.input}")

    # Detect / load nuScenes meta
    meta_dir = args.meta_dir or _find_nuscenes_meta()
    meta = None
    index = None
    if meta_dir and not args.force_heuristic:
        print(f"Using nuScenes meta at: {meta_dir}")
        t_meta0 = __import__("time").time()
        meta = _load_nusc_meta(meta_dir)
        for k, v in meta.items():
            print(f"  {k}: {len(v)} entries")
        print(f"  meta load: {__import__('time').time() - t_meta0:.1f} s")
        t_idx0 = __import__("time").time()
        index = _MetaIndex(meta)
        print(f"  index build: {__import__('time').time() - t_idx0:.1f} s")
        # Free the heavy raw lists ASAP — they're ~2 GB of dicts after parsing
        meta = None
    else:
        if args.force_heuristic:
            print("--force-heuristic: skipping real nuScenes meta.")
        else:
            print("WARNING: nuScenes raw meta NOT FOUND on disk.")
            print(f"  Checked: {NUSC_META_CANDIDATES}")
            print("  Falling back to behavior-heuristic trajectories (tagged 'heuristic').")
            print("  Download v1.0-trainval metadata for real trajectories:")
            print("    https://www.nuscenes.org/download  -> 'Full dataset' -> Metadata only (~ 400 MB)")

    out_records = []
    stats = Counter()

    # Cache: many records share the same frame_token (one per QA category).
    # Compute the trajectory once per frame_token.
    traj_cache: Dict[str, Optional[np.ndarray]] = {}

    try:
        from tqdm import tqdm  # noqa: WPS433 (local import: optional dep)
        iterator = tqdm(data, desc="extract_traj", unit="rec", mininterval=2.0)
    except ImportError:
        iterator = data

    for rec in iterator:
        frame_token = rec.get("frame_token") or rec.get("metadata", {}).get("frame_token", "")
        if not frame_token:
            stats["no_frame_token"] += 1
            continue

        waypoints = None
        source = None
        if index is not None:
            if frame_token in traj_cache:
                waypoints = traj_cache[frame_token]
            else:
                waypoints = _build_future_trajectory_real(
                    index, frame_token, args.horizon_s, args.sample_hz,
                )
                traj_cache[frame_token] = waypoints
            if waypoints is not None:
                source = "nuscenes_real"
                stats["resolved_real"] += 1
            else:
                stats["unresolved_real_skipped"] += 1
                # When real meta is available, OPTION A says: drop the record
                # rather than fall back to heuristic. Heuristic would mask the
                # real-meta coverage gaps and pollute training.
                continue

        if waypoints is None:
            # Heuristic path (only entered when index is None, i.e. no real meta).
            behavior_text = ""
            if rec.get("metadata", {}).get("category") == "behavior":
                msgs = rec.get("messages", [])
                for m in msgs:
                    if m.get("role") == "assistant":
                        behavior_text = m.get("content", "") or ""
                        break
            waypoints = _heuristic_trajectory(behavior_text, args.horizon_s, args.sample_hz)
            source = "heuristic"
            stats["resolved_heuristic"] += 1

        # Encode
        action_tokens = tok.encode(waypoints, with_boundaries=True)

        # Regression metric for the frozen-tail bugfix: check if the last
        # two waypoints are within 1 cm of each other (i.e. would have been
        # produced by the old "clamp to end of scene" path).
        last_two_same = bool(
            np.allclose(waypoints[-1], waypoints[-2], atol=0.01)
        )
        if last_two_same:
            stats["trailing_duplicate"] += 1

        # Round-trip echo for debugging on the first few
        if stats["written"] < 3:
            rt = tok.decode(action_tokens)
            print(f"  [demo] frame={frame_token[:8]} src={source} "
                  f"wp_first={waypoints[0].tolist()} rt_first={rt[0].tolist()}  "
                  f"wp_last={waypoints[-1].tolist()}")

        rec2 = dict(rec)
        rec2["action_tokens"] = action_tokens
        rec2.setdefault("metadata", {})
        rec2["metadata"]["traj_source"] = source
        rec2["metadata"]["traj_horizon_s"] = args.horizon_s
        rec2["metadata"]["traj_sample_hz"] = args.sample_hz
        rec2["metadata"]["traj_num_waypoints"] = cfg.num_waypoints
        rec2["metadata"]["traj_num_bins"] = cfg.num_bins
        rec2["metadata"]["traj_min_m"] = cfg.min_m
        rec2["metadata"]["traj_max_m"] = cfg.max_m
        rec2["metadata"]["waypoints_xy_m"] = waypoints.tolist()  # for eval

        out_records.append(rec2)
        stats["written"] += 1

    print()
    print(f"Wrote {stats['written']}/{len(data)} records")
    for k, v in stats.most_common():
        print(f"  {k}: {v}")

    if stats["written"] > 0:
        dup_pct = 100.0 * stats["trailing_duplicate"] / stats["written"]
        print(f"  trailing_duplicate ratio: {dup_pct:.2f}% "
              f"(target after bugfix: <= 5%)")

    with open(out_path, "w") as f:
        json.dump(out_records, f)
    print(f"\nOutput: {out_path}  ({os.path.getsize(out_path)/1024**2:.1f} MB)")

    # Persist tokenizer config alongside for downstream consumers
    cfg_path = os.path.join(os.path.dirname(out_path), "trajectory_tokenizer.json")
    from trajectory_tokenizer import save_config
    save_config(cfg, cfg_path)
    print(f"Tokenizer config: {cfg_path}")


if __name__ == "__main__":
    main()
