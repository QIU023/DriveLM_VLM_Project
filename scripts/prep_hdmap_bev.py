#!/usr/bin/env python3
"""Pre-render HD-map -> 224x224 BEV PNG per nuScenes keyframe.

Output convention: ``{output_dir}/{sample_token}.png`` (RGB, 224x224, vehicle
points up). Cache is consumed by the multimodal AD-VLA dataloader.

This is a CPU-only preprocessor. Multiprocessed (one ``NuScenesMap`` instance
per worker, persisted across samples within the worker for amortised JSON
load).

Layer / colour code (RGB, 0-255):
    drivable_area         (100, 100, 100)   gray background
    lane                  ( 70, 200, 100)   light green
    road_divider          (255, 255, 255)   white
    lane_divider          (255, 240,  80)   yellow
    ped_crossing          (220,  60,  60)   red
    stop_line             (200,   0,   0)   deep red
    walkway               ( 60, 120,  60)   muted green
    carpark_area          (140, 100,  60)   brown
    road_segment          ( 80,  80, 100)   blue-ish gray (below drivable)

Layer order (drawn back-to-front): road_segment, drivable_area, walkway,
carpark_area, lane, ped_crossing, stop_line, road_divider, lane_divider.
Traffic-light layer is dropped (point-only; <0.05 mean pix per sample, not
worth the API cost).

Map data requirement: nuScenes Map Expansion v1.3 unpacked into
``{maps_root}/expansion/{location}.json`` for the four
nuScenes locations (singapore-onenorth, singapore-hollandvillage,
singapore-queenstown, boston-seaport).

Run a smoke first (single sample) before the full sweep:
    python3 scripts/prep_hdmap_bev.py \\
        --infos-pkl data/uniad_infos/nuscenes_infos_temporal_train.pkl \\
        --maps-root data/nuscenes \\
        --output-dir data/preproc/hdmap_bev \\
        --max-samples 1
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from pyquaternion import Quaternion


# nuScenes Map Expansion locations (fixed by the dataset).
NUSCENES_LOCATIONS = (
    "singapore-onenorth",
    "singapore-hollandvillage",
    "singapore-queenstown",
    "boston-seaport",
)

# Layer order (back -> front). get_map_mask returns layers in the SAME order
# we request; we composite from layer[0] up to layer[-1].
LAYER_ORDER = (
    "road_segment",
    "drivable_area",
    "walkway",
    "carpark_area",
    "lane",
    "ped_crossing",
    "stop_line",
    "road_divider",
    "lane_divider",
)

LAYER_COLOURS: Dict[str, Tuple[int, int, int]] = {
    "road_segment": (80, 80, 100),
    "drivable_area": (100, 100, 100),
    "walkway": (60, 120, 60),
    "carpark_area": (140, 100, 60),
    "lane": (70, 200, 100),
    "ped_crossing": (220, 60, 60),
    "stop_line": (200, 0, 0),
    "road_divider": (255, 255, 255),
    "lane_divider": (255, 240, 80),
}


def quat_to_yaw(rot_quat: List[float]) -> float:
    """Quaternion -> yaw (rotation about z) in degrees."""
    q = Quaternion(rot_quat)
    # pyquaternion -> yaw via rotation of a forward vector
    fwd = q.rotate(np.array([1.0, 0.0, 0.0]))
    yaw_rad = float(np.arctan2(fwd[1], fwd[0]))
    return float(np.degrees(yaw_rad))


def load_scene_location_map(maps_meta_root: Path) -> Dict[str, str]:
    """Build ``scene_token -> location`` from nuScenes ``v1.0-trainval`` JSONs.

    Reads ``scene.json`` and ``log.json`` once.
    """
    sj = json.loads((maps_meta_root / "scene.json").read_text())
    lj = json.loads((maps_meta_root / "log.json").read_text())
    log_to_loc = {l["token"]: l["location"] for l in lj}
    scene_to_loc = {s["token"]: log_to_loc[s["log_token"]] for s in sj}
    return scene_to_loc


def composite_layers(masks: np.ndarray, layer_names: List[str]) -> np.ndarray:
    """Composite per-layer binary masks (C, H, W) into an RGB image (H, W, 3).

    Layers are drawn back-to-front in the order they appear in ``layer_names``.
    """
    if masks.ndim != 3:
        raise ValueError(f"Expected (C,H,W) masks, got shape {masks.shape}")
    C, H, W = masks.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for i, name in enumerate(layer_names):
        col = LAYER_COLOURS.get(name, (255, 255, 255))
        m = masks[i].astype(bool)
        rgb[m] = col
    return rgb


def render_one(
    nmap,
    ego_xy: Tuple[float, float],
    ego_yaw_deg: float,
    patch_range_m: float,
    canvas_size: int,
) -> np.ndarray:
    """Render a single BEV crop with the vehicle pointing up.

    ``get_map_mask`` takes ``patch_box = [x, y, height, width]`` in map coords
    and a ``patch_angle`` (degrees). We rotate the patch so that the ego's
    heading aligns with the canvas's +Y (up); equivalent to ``patch_angle =
    yaw - 90`` so heading -> up after image_y-flip.

    Returns RGB ``(canvas_size, canvas_size, 3)`` uint8 with vehicle at centre,
    facing up.
    """
    x, y = ego_xy
    side_m = float(patch_range_m) * 2.0  # full edge length
    patch_box = (float(x), float(y), side_m, side_m)
    # patch_angle rotation convention: see nuScenes devkit's get_map_mask.
    # 0 deg = north-aligned patch. To make ego heading point up in the output
    # image, we rotate the patch by (yaw - 90) deg.
    patch_angle = float(ego_yaw_deg) - 90.0

    masks = nmap.get_map_mask(
        patch_box=patch_box,
        patch_angle=patch_angle,
        layer_names=list(LAYER_ORDER),
        canvas_size=(canvas_size, canvas_size),
    )
    # get_map_mask returns image-row-major: row 0 at top. We want vehicle to
    # point UP in the output PNG, so flip vertically (devkit convention).
    masks = np.flip(masks, axis=1).copy()
    rgb = composite_layers(masks, list(LAYER_ORDER))
    return rgb


# ---- worker harness --------------------------------------------------------

_WORKER_STATE: Dict[str, object] = {}


def _worker_init(maps_root: str) -> None:
    from nuscenes.map_expansion.map_api import NuScenesMap  # noqa: WPS433

    _WORKER_STATE["maps_root"] = maps_root
    _WORKER_STATE["nmap_cache"] = {}
    _WORKER_STATE["NuScenesMap_cls"] = NuScenesMap


def _get_nmap(location: str):
    cache = _WORKER_STATE["nmap_cache"]
    if location not in cache:
        cls = _WORKER_STATE["NuScenesMap_cls"]
        cache[location] = cls(
            dataroot=_WORKER_STATE["maps_root"],
            map_name=location,
        )
    return cache[location]


def _render_and_save(task: dict) -> Tuple[str, str]:
    """task = {token, location, ego_xy, ego_yaw_deg, out_path,
                patch_range_m, canvas_size}."""
    out_path = task["out_path"]
    if os.path.exists(out_path):
        return (task["token"], "cached")
    try:
        nmap = _get_nmap(task["location"])
        rgb = render_one(
            nmap=nmap,
            ego_xy=task["ego_xy"],
            ego_yaw_deg=task["ego_yaw_deg"],
            patch_range_m=task["patch_range_m"],
            canvas_size=task["canvas_size"],
        )
        # Atomic write: tmp -> rename.
        tmp_path = out_path + ".tmp"
        Image.fromarray(rgb, mode="RGB").save(tmp_path, format="PNG", optimize=False)
        os.replace(tmp_path, out_path)
        return (task["token"], "ok")
    except Exception as exc:  # noqa: BLE001
        return (task["token"], f"err:{type(exc).__name__}:{exc}")


# ---- main ------------------------------------------------------------------


def build_tasks(
    infos_pkl: Path,
    scene_to_loc: Dict[str, str],
    output_dir: Path,
    patch_range_m: float,
    canvas_size: int,
    max_samples: Optional[int],
) -> List[dict]:
    with open(infos_pkl, "rb") as f:
        data = pickle.load(f)
    infos = data["infos"]
    tasks: List[dict] = []
    skipped_loc = 0
    for inf in infos:
        token = inf["token"]
        scene_token = inf.get("scene_token")
        loc = scene_to_loc.get(scene_token)
        if loc is None:
            skipped_loc += 1
            continue
        ego_t = inf.get("ego2global_translation")
        ego_r = inf.get("ego2global_rotation")
        if ego_t is None or ego_r is None:
            skipped_loc += 1
            continue
        yaw_deg = quat_to_yaw(ego_r)
        tasks.append({
            "token": token,
            "location": loc,
            "ego_xy": (float(ego_t[0]), float(ego_t[1])),
            "ego_yaw_deg": yaw_deg,
            "out_path": str(output_dir / f"{token}.png"),
            "patch_range_m": float(patch_range_m),
            "canvas_size": int(canvas_size),
        })
        if max_samples is not None and len(tasks) >= max_samples:
            break
    if skipped_loc:
        print(f"WARN: skipped {skipped_loc} infos without scene/log/loc/ego mapping")
    return tasks


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--infos-pkl", type=Path, required=True,
                    help="UniAD-style temporal infos pkl (e.g. nuscenes_infos_temporal_train.pkl)")
    ap.add_argument("--maps-root", type=Path, required=True,
                    help="Root directory containing maps/expansion/*.json and v1.0-trainval/*.json")
    ap.add_argument("--meta-root", type=Path, default=None,
                    help="Directory with scene.json / log.json. Defaults to <maps-root>/v1.0-trainval")
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="Where to dump per-sample PNGs.")
    ap.add_argument("--size", type=int, default=224,
                    help="Output canvas size (square). Default 224.")
    ap.add_argument("--range", dest="range_m", type=float, default=50.0,
                    help="Half-edge of the BEV crop in metres. 50 -> 100m x 100m. Default 50.")
    ap.add_argument("--num-workers", type=int, default=8,
                    help="Worker processes. Default 8.")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="Cap on samples (debug/smoke).")
    ap.add_argument("--progress-every", type=int, default=500,
                    help="Print progress every N samples.")
    return ap.parse_args()


def precheck_maps(maps_root: Path) -> Optional[str]:
    """Verify that the expansion JSONs we need exist. Returns error string or None."""
    exp_dir = maps_root / "maps" / "expansion"
    if not exp_dir.exists():
        return f"Map expansion dir not found: {exp_dir}"
    missing = []
    for loc in NUSCENES_LOCATIONS:
        if not (exp_dir / f"{loc}.json").exists():
            missing.append(loc)
    if missing:
        return f"Missing expansion JSON(s): {missing} under {exp_dir}"
    return None


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    meta_root = args.meta_root or (args.maps_root / "v1.0-trainval")
    if not meta_root.exists():
        print(f"ERROR: meta dir not found: {meta_root}", file=sys.stderr)
        return 2

    err = precheck_maps(args.maps_root)
    if err:
        print(f"ERROR: {err}", file=sys.stderr)
        print(
            "Hint: download nuScenes Map Expansion v1.3 and unpack into "
            f"{args.maps_root}/maps/expansion/",
            file=sys.stderr,
        )
        return 3

    scene_to_loc = load_scene_location_map(meta_root)
    print(f"scene_to_loc: {len(scene_to_loc)} scenes mapped to "
          f"{len(set(scene_to_loc.values()))} locations")

    tasks = build_tasks(
        infos_pkl=args.infos_pkl,
        scene_to_loc=scene_to_loc,
        output_dir=args.output_dir,
        patch_range_m=args.range_m,
        canvas_size=args.size,
        max_samples=args.max_samples,
    )
    print(f"Planned {len(tasks)} renders -> {args.output_dir}")
    if not tasks:
        return 0

    n_workers = max(1, int(args.num_workers))
    t0 = time.time()
    n_ok = 0
    n_cached = 0
    n_err = 0
    sample_errors: List[str] = []

    if n_workers == 1:
        _worker_init(str(args.maps_root))
        for i, task in enumerate(tasks, 1):
            _, status = _render_and_save(task)
            if status == "ok":
                n_ok += 1
            elif status == "cached":
                n_cached += 1
            else:
                n_err += 1
                if len(sample_errors) < 5:
                    sample_errors.append(status)
            if i % args.progress_every == 0:
                rate = i / max(time.time() - t0, 1e-6)
                print(f"  [{i}/{len(tasks)}] ok={n_ok} cached={n_cached} err={n_err} "
                      f"({rate:.1f} samp/s)")
    else:
        ctx = mp.get_context("spawn")  # avoid forking torch state from parent
        with ctx.Pool(n_workers, initializer=_worker_init,
                      initargs=(str(args.maps_root),)) as pool:
            for i, (_, status) in enumerate(
                pool.imap_unordered(_render_and_save, tasks, chunksize=8),
                start=1,
            ):
                if status == "ok":
                    n_ok += 1
                elif status == "cached":
                    n_cached += 1
                else:
                    n_err += 1
                    if len(sample_errors) < 5:
                        sample_errors.append(status)
                if i % args.progress_every == 0:
                    rate = i / max(time.time() - t0, 1e-6)
                    print(f"  [{i}/{len(tasks)}] ok={n_ok} cached={n_cached} "
                          f"err={n_err} ({rate:.1f} samp/s)")

    elapsed = time.time() - t0
    print(
        f"DONE: ok={n_ok} cached={n_cached} err={n_err} elapsed={elapsed:.1f}s "
        f"({len(tasks)/max(elapsed,1e-6):.1f} samp/s)"
    )
    if sample_errors:
        print("Sample errors (first 5):")
        for e in sample_errors:
            print(f"  - {e}")
        return 4 if n_err > 0 else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
