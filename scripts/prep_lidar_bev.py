"""Precompute frozen LiDAR BEV feature cache for the full-modal AD-VLA.

Why this exists
---------------
The 3B Qwen2.5-VL VLA is trained on RGB-only today. To extend to a full-modal
stack (DriveMLM / OmniDrive style), we want a per-keyframe LiDAR BEV feature
that can be concatenated with the visual-token block via a learnable
projector. The encoder side is **frozen** — we cache once, then the VLA only
trains the small projector that maps (C, H, W) BEV -> 64 LM tokens.

Encoder choice (see Step 1 investigation in the task brief):
- `pointpillars` / `centerpoint` rely on mmdetection3d + spconv. As of
  2026-05-21 on this RTX 5090 (Blackwell sm_120) + torch 2.11 + cu130 box,
  there is NO prebuilt mmcv / spconv wheel and source builds fail. We
  surface that here as a hard error rather than monkey-patching.
- `occupancy` (this script's only working mode today) is a hand-rolled
  multi-channel BEV occupancy + intensity + height summary. NOT a pretrained
  detector — it's a simplified BEV that the downstream projector still has to
  learn to read. This is honest about the trade-off: ~zero install risk, no
  learnable weights to track, frozen by construction. If/when we move to a
  Blackwell-compatible mmdet3d wheel, swap in PointPillars by adding an
  `--encoder pointpillars` branch.

Output layout
-------------
For each keyframe with token `t`, we write
    <output_dir>/<t>.npz
containing a single key `bev` with shape (C, H, W), dtype float16.

Default config (--encoder occupancy):
    range  x ∈ [-50, 50] m,  y ∈ [-50, 50] m,  z ∈ [-5, 3] m
    grid   H = W = 128  (0.78 m / cell)
    C = 8 channels:
      0..4  : occupancy per z-slice in {[-5,-3), [-3,-1), [-1,1), [1,2), [2,3)}
      5     : max intensity in cell (normalized to [0,1])
      6     : max height in cell (normalized to [0,1])
      7     : log(1 + point density in cell), normalized to [0,1]
    size per sample: 8 * 128 * 128 * 2 B = 256 KB
    full trainval (≈29K samples) ≈ 7.3 GB compressed .npz (a bit less due to
    zlib on the sparse occupancy planes).

CLI
---
    python scripts/prep_lidar_bev.py \
        --infos-pkl data/nuscenes/_hf_meta/nuscenes_mmdet3d-12Hz/nuscenes_interp_12Hz_infos_train.pkl \
        --lidar-root data/nuscenes \
        --output-dir data_processed/lidar_bev_occ_v1/train \
        --encoder occupancy \
        --device cpu \
        [--limit N]  [--workers K]  [--skip-existing]

The script never touches GPU memory unless `--device cuda` is passed AND a
GPU is free. Default device is cpu so we can run during A.3 v2 training
without risk.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_X_RANGE: Tuple[float, float] = (-50.0, 50.0)
DEFAULT_Y_RANGE: Tuple[float, float] = (-50.0, 50.0)
# 5 z-slices spanning -5..3 m above ground plane. Boundaries chosen so the
# road surface (~0) sits inside slice 2, and roof-level returns (~2-3 m on a
# nuScenes vehicle frame) live in slice 4.
Z_SLICE_EDGES: Tuple[float, ...] = (-5.0, -3.0, -1.0, 1.0, 2.0, 3.0)
DEFAULT_GRID: int = 128
INTENSITY_MAX: float = 255.0
HEIGHT_NORM_RANGE: Tuple[float, float] = (-5.0, 3.0)  # (min, max) for height channel
DENSITY_LOG_CAP: float = 5.0  # log1p(150) ≈ 5.0 — caps high-density cells
N_CHANNELS: int = len(Z_SLICE_EDGES) - 1 + 3  # 5 occupancy + intensity + height + density = 8


# ---------------------------------------------------------------------------
# Core BEV encoding
# ---------------------------------------------------------------------------
def load_pcd_bin(path: str) -> np.ndarray:
    """Load a nuScenes LIDAR_TOP .pcd.bin file into an (N, 5) array of
    (x, y, z, intensity, ring)."""
    pc = np.fromfile(path, dtype=np.float32)
    if pc.size % 5 != 0:
        raise ValueError(f"{path}: size {pc.size} not divisible by 5")
    return pc.reshape(-1, 5)


def encode_occupancy_bev(
    points: np.ndarray,
    grid: int = DEFAULT_GRID,
    x_range: Tuple[float, float] = DEFAULT_X_RANGE,
    y_range: Tuple[float, float] = DEFAULT_Y_RANGE,
    z_edges: Tuple[float, ...] = Z_SLICE_EDGES,
) -> np.ndarray:
    """Vectorized hand-rolled BEV occupancy encoder.

    Args:
        points: (N, >=4) array; columns 0..3 = x, y, z, intensity.
        grid: spatial grid size H = W.
        x_range, y_range: BEV crop bounds (m).
        z_edges: 1-D iterable of z bin edges; produces len(z_edges)-1 slices.

    Returns:
        (C, H, W) float16 BEV feature with channels documented at the top of
        this file. Always returns the same shape regardless of point count.
    """
    n_z = len(z_edges) - 1
    H = W = grid
    out = np.zeros((N_CHANNELS, H, W), dtype=np.float32)

    if points.shape[0] == 0:
        return out.astype(np.float16)

    x, y, z, intensity = points[:, 0], points[:, 1], points[:, 2], points[:, 3]

    # Crop to BEV window
    in_xy = (
        (x >= x_range[0]) & (x < x_range[1]) & (y >= y_range[0]) & (y < y_range[1])
    )
    in_z = (z >= z_edges[0]) & (z < z_edges[-1])
    keep = in_xy & in_z
    if not keep.any():
        return out.astype(np.float16)

    x, y, z, intensity = x[keep], y[keep], z[keep], intensity[keep]

    # Discretize to grid indices. nuScenes convention: x forward, y left. We
    # follow common BEV image convention: image rows = -y axis (so "up" in
    # image = forward), image cols = +x axis. But to keep things simple and
    # consistent, we'll map (x, y) -> (col, row) directly without sign flips.
    # The downstream projector learns whatever orientation we pick.
    x_res = (x_range[1] - x_range[0]) / W
    y_res = (y_range[1] - y_range[0]) / H
    col = np.clip(((x - x_range[0]) / x_res).astype(np.int64), 0, W - 1)
    row = np.clip(((y - y_range[0]) / y_res).astype(np.int64), 0, H - 1)

    # z-slice index per point
    z_idx = np.clip(np.searchsorted(np.asarray(z_edges), z, side="right") - 1, 0, n_z - 1)

    # Channels 0..n_z-1 : occupancy per z-slice (binary, max over points)
    # We use np.maximum.at to mark any point in cell as 1.0. For speed on
    # large N we flatten and use bincount with a per-channel offset.
    flat_xy = row * W + col            # 0..H*W-1
    # Occupancy: count points per (slice, cell) > 0
    occ_flat = np.zeros((n_z, H * W), dtype=np.uint32)
    np.add.at(occ_flat, (z_idx, flat_xy), 1)
    out[:n_z] = (occ_flat > 0).astype(np.float32).reshape(n_z, H, W)

    # Channel n_z : max intensity per cell (normalized to [0,1])
    intensity_norm = np.clip(intensity / INTENSITY_MAX, 0.0, 1.0)
    inten_flat = np.full(H * W, -1.0, dtype=np.float32)
    np.maximum.at(inten_flat, flat_xy, intensity_norm)
    inten_flat = np.where(inten_flat < 0, 0.0, inten_flat)
    out[n_z] = inten_flat.reshape(H, W)

    # Channel n_z+1 : max height per cell (normalized to [0,1] over height range)
    h_min, h_max = HEIGHT_NORM_RANGE
    z_norm = np.clip((z - h_min) / (h_max - h_min), 0.0, 1.0)
    h_flat = np.full(H * W, -1.0, dtype=np.float32)
    np.maximum.at(h_flat, flat_xy, z_norm)
    h_flat = np.where(h_flat < 0, 0.0, h_flat)
    out[n_z + 1] = h_flat.reshape(H, W)

    # Channel n_z+2 : log(1 + density) normalized to [0,1]
    dens_flat = np.zeros(H * W, dtype=np.float32)
    np.add.at(dens_flat, flat_xy, 1.0)
    dens_log = np.log1p(dens_flat) / DENSITY_LOG_CAP
    out[n_z + 2] = np.clip(dens_log, 0.0, 1.0).reshape(H, W)

    return out.astype(np.float16)


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
def resolve_lidar_path(info: Dict, lidar_root: Path) -> Path:
    """info['lidar_path'] is stored as a relative path like
    '../data/nuscenes/samples/LIDAR_TOP/<file>.pcd.bin' (relative to the
    mmdet3d-12Hz pkl). We strip leading '../' and re-root under lidar_root.
    """
    raw = info["lidar_path"]
    # Strip a few patterns of ../ prefix
    rel = raw
    while rel.startswith("../"):
        rel = rel[3:]
    # Strip leading 'data/nuscenes/' if present, so we can re-root cleanly
    for prefix in ("data/nuscenes/", "./data/nuscenes/"):
        if rel.startswith(prefix):
            rel = rel[len(prefix) :]
            break
    return lidar_root / rel


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def _process_one(
    args_tuple: Tuple[Dict, str, str, bool, int],
) -> Tuple[str, str]:
    """Worker for ProcessPoolExecutor. Returns (token, status_string)."""
    info, lidar_root, out_dir, skip_existing, grid = args_tuple
    token = info["token"]
    out_path = Path(out_dir) / f"{token}.npz"
    if skip_existing and out_path.exists():
        return token, "skip"
    try:
        path = resolve_lidar_path(info, Path(lidar_root))
        if not path.exists():
            return token, f"missing:{path}"
        points = load_pcd_bin(str(path))
        bev = encode_occupancy_bev(points, grid=grid)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, bev=bev)
        return token, "ok"
    except Exception as e:  # pragma: no cover — defensive
        return token, f"err:{type(e).__name__}:{e}"


# ---------------------------------------------------------------------------
# CLI driver
# ---------------------------------------------------------------------------
def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Precompute frozen LiDAR BEV occupancy cache (per-keyframe)."
    )
    ap.add_argument("--infos-pkl", required=True, type=Path)
    ap.add_argument("--lidar-root", required=True, type=Path,
                    help="Root such that <root>/samples/LIDAR_TOP/*.pcd.bin exists")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--encoder", default="occupancy",
                    choices=["occupancy", "pointpillars", "centerpoint"])
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--grid", type=int, default=DEFAULT_GRID)
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, only process the first N samples (smoke).")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--keyframes-only", action="store_true",
                    help="Only process info['is_key_frame']==True samples "
                         "(default: process all entries in the pkl).")
    ap.add_argument("--log-every", type=int, default=500)
    args = ap.parse_args(argv)

    # Hard fail with a useful error if user asks for a non-installable encoder
    if args.encoder in ("pointpillars", "centerpoint"):
        sys.stderr.write(
            f"[FATAL] --encoder {args.encoder} requires mmdetection3d + spconv,\n"
            "which currently have NO prebuilt wheel for this box's stack\n"
            "(torch 2.11 + cu130 + sm_120 / Blackwell RTX 5090). Source builds\n"
            "of mmcv 2.x fail with missing pkg_resources / kernel-compile errors\n"
            "against cu13. See docs/multimodal/lidar_bev_README.md.\n"
            "Use --encoder occupancy as the documented fallback.\n"
        )
        return 2

    if args.device == "cuda":
        sys.stderr.write(
            "[WARN] --device cuda is currently a no-op: the occupancy encoder\n"
            "is pure numpy and runs faster on CPU due to vectorized scatter ops.\n"
            "Keeping device=cpu internally.\n"
        )

    # Load infos
    if not args.infos_pkl.exists():
        sys.stderr.write(f"[FATAL] {args.infos_pkl} not found\n")
        return 3
    with args.infos_pkl.open("rb") as f:
        pkl = pickle.load(f)
    if not isinstance(pkl, dict) or "infos" not in pkl:
        sys.stderr.write(f"[FATAL] {args.infos_pkl}: unexpected schema\n")
        return 3
    infos: List[Dict] = pkl["infos"]
    if args.keyframes_only:
        infos = [i for i in infos if i.get("is_key_frame", False)]
    if args.limit > 0:
        infos = infos[: args.limit]
    print(f"[prep_lidar_bev] {len(infos)} samples to process "
          f"(keyframes_only={args.keyframes_only})")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    work = [
        (info, str(args.lidar_root), str(args.output_dir), args.skip_existing, args.grid)
        for info in infos
    ]

    counts = {"ok": 0, "skip": 0, "missing": 0, "err": 0}
    t0 = time.time()

    # ProcessPool keeps each worker's numpy in a separate address space, so
    # the parent process can stream tqdm-like progress without GIL pressure.
    if args.workers <= 1:
        for w in work:
            tok, status = _process_one(w)
            _tally(counts, status)
            if (counts["ok"] + counts["skip"]) % args.log_every == 0:
                _log_progress(counts, len(work), t0)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(_process_one, w) for w in work]
            for i, fut in enumerate(as_completed(futures), 1):
                _tok, status = fut.result()
                _tally(counts, status)
                if i % args.log_every == 0:
                    _log_progress(counts, len(work), t0)

    _log_progress(counts, len(work), t0, final=True)
    return 0 if counts["err"] == 0 and counts["missing"] == 0 else 1


def _tally(counts: Dict[str, int], status: str) -> None:
    if status == "ok":
        counts["ok"] += 1
    elif status == "skip":
        counts["skip"] += 1
    elif status.startswith("missing"):
        counts["missing"] += 1
    elif status.startswith("err"):
        counts["err"] += 1


def _log_progress(counts: Dict[str, int], total: int, t0: float,
                  final: bool = False) -> None:
    done = sum(counts.values())
    elapsed = time.time() - t0
    rate = done / max(elapsed, 1e-6)
    tag = "[DONE]" if final else "[..]"
    print(
        f"{tag} {done}/{total}  ok={counts['ok']} skip={counts['skip']} "
        f"missing={counts['missing']} err={counts['err']}  "
        f"{rate:.1f} samp/s  elapsed={elapsed:.1f}s",
        flush=True,
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
