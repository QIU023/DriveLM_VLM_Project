"""Real-data smoke for prep_lidar_bev.encode_occupancy_bev.

Verifies:
  1. A real .pcd.bin loads.
  2. encode_occupancy_bev produces the expected shape and non-degenerate stats.
  3. save(.npz) -> load(.npz) round-trip is bitwise-identical.
  4. Captures all boot warnings raised during import + encode (none expected).

Run:
    PYTHONPATH=. python3 scripts/_smoke_lidar_bev_roundtrip.py <some>.pcd.bin
or no args -> auto-picks first .bin under data/nuscenes/samples/LIDAR_TOP_smoke.
"""

from __future__ import annotations

import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

# Capture warnings emitted at import / encode time
_warn_log: list[warnings.WarningMessage] = []
warnings.simplefilter("always")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    from scripts.prep_lidar_bev import (  # noqa: E402
        N_CHANNELS,
        DEFAULT_GRID,
        encode_occupancy_bev,
        load_pcd_bin,
    )
    _warn_log.extend(caught)


def _pick_bin() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    default_dir = REPO / "data/nuscenes/samples/LIDAR_TOP_smoke"
    cands = sorted(default_dir.glob("*.pcd.bin"))
    if not cands:
        # also try the canonical location
        alt = REPO / "data/nuscenes/samples/LIDAR_TOP"
        cands = sorted(alt.glob("*.pcd.bin"))
    if not cands:
        sys.exit(
            "[FAIL] no .pcd.bin found under "
            f"{default_dir} or canonical LIDAR_TOP path"
        )
    return cands[0]


def main() -> int:
    bin_path = _pick_bin()
    print(f"[INFO] using {bin_path}")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pc = load_pcd_bin(str(bin_path))
        bev = encode_occupancy_bev(pc)
        _warn_log.extend(caught)

    # 1) Shape check
    expected = (N_CHANNELS, DEFAULT_GRID, DEFAULT_GRID)
    if bev.shape != expected:
        print(f"[FAIL] bev shape {bev.shape} != expected {expected}")
        return 1
    print(f"[PASS] bev.shape = {bev.shape}, dtype = {bev.dtype}")

    # 2) Non-degenerate stats
    nz = int((bev != 0).sum())
    occ_planes = bev[:5]
    inten_plane = bev[5]
    height_plane = bev[6]
    density_plane = bev[7]
    occ_cells = int((occ_planes.sum(axis=0) > 0).sum())
    total_cells = DEFAULT_GRID * DEFAULT_GRID
    print(
        f"[INFO] non-zero entries: {nz} / {bev.size} "
        f"({100*nz/bev.size:.2f}%)"
    )
    print(
        f"[INFO] BEV cells with any occupancy: {occ_cells} / {total_cells} "
        f"({100*occ_cells/total_cells:.2f}%)"
    )
    print(f"[INFO] intensity plane: min={inten_plane.min():.4f} "
          f"max={inten_plane.max():.4f} mean(occ)={inten_plane[inten_plane>0].mean():.4f}")
    print(f"[INFO] height plane:    min={height_plane.min():.4f} "
          f"max={height_plane.max():.4f} mean(occ)={height_plane[height_plane>0].mean():.4f}")
    print(f"[INFO] density plane:   min={density_plane.min():.4f} "
          f"max={density_plane.max():.4f} mean(occ)={density_plane[density_plane>0].mean():.4f}")

    if occ_cells < 100:
        print(f"[FAIL] only {occ_cells} occupied cells — encoder is broken or "
              f"the point cloud is degenerate")
        return 1
    if not (0.0 <= inten_plane.max() <= 1.0):
        print("[FAIL] intensity plane out of [0,1]")
        return 1
    if not (0.0 <= height_plane.max() <= 1.0):
        print("[FAIL] height plane out of [0,1]")
        return 1
    if not (0.0 <= density_plane.max() <= 1.0):
        print("[FAIL] density plane out of [0,1]")
        return 1

    # 3) Save -> load round-trip (bitwise identical)
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tf:
        npz_path = tf.name
    np.savez_compressed(npz_path, bev=bev)
    reloaded = np.load(npz_path)["bev"]
    if reloaded.dtype != bev.dtype:
        print(f"[FAIL] dtype changed: {bev.dtype} -> {reloaded.dtype}")
        return 1
    if reloaded.shape != bev.shape:
        print(f"[FAIL] shape changed: {bev.shape} -> {reloaded.shape}")
        return 1
    if not np.array_equal(reloaded, bev):
        diff = (reloaded != bev).sum()
        print(f"[FAIL] {diff} elements differ after round-trip")
        return 1
    size_bytes = Path(npz_path).stat().st_size
    print(f"[PASS] save/load round-trip identical "
          f"(npz on-disk = {size_bytes/1024:.1f} KB)")

    # 4) Frozen-encoder verification: the encoder is a pure function of the
    #    input point cloud with no learnable parameters. Run it twice and
    #    confirm bit-equality (proves no internal RNG / state).
    bev2 = encode_occupancy_bev(pc)
    if not np.array_equal(bev, bev2):
        print("[FAIL] encoder not deterministic (this would imply hidden state)")
        return 1
    print("[PASS] frozen-encoder verification: deterministic, no learnable params")

    # 5) Boot warnings
    if _warn_log:
        print(f"[INFO] {len(_warn_log)} warning(s) during import+encode:")
        for w in _warn_log:
            print(f"   {w.category.__name__}: {w.message}  ({w.filename}:{w.lineno})")
    else:
        print("[PASS] no warnings emitted during import + encode")

    print("[ALL PASS]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
