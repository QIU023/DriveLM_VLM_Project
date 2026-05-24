#!/usr/bin/env python3
"""Convert grpo_vla.dataset_adapter.VeRLNuScenesDataset -> veRL parquet files.

veRL's default `RLHFDataset` reads parquet rows with columns:
    prompt           : str               (chat-template text)
    images           : list[bytes]       (PNG-encoded image bytes per sample)
    extra_info       : dict (json-safe)  (reward-side info)
    ground_truth     : any  (json-safe)  (mirror; passed to reward fn)
    data_source      : str               (constant tag for our task)

This builder iterates Agent C's VeRLNuScenesDataset and writes:
    /workspace/.../grpo_vla/data/nusc_planning_train.parquet
    /workspace/.../grpo_vla/data/nusc_planning_val.parquet

Memory: each row encodes 4-7 images (3 cams x N frames + HD-map BEV) as PNG
bytes ~80-150 KB each -> ~0.5-1 MB / row. 24K train rows ~= 18 GB on disk.
We stream in chunks of 256 rows to avoid OOM.

Usage:
    /usr/bin/python3 build_parquet.py --config configs/grpo_b5prime_3cam.yaml \
        --split train --limit -1 --chunk-size 256

The launcher calls this for `train` (if file missing) and `val` (if missing
or stale). On a re-run with the file present, this script no-ops.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

GRPO_DIR = Path("/workspace/DriveLM_VLM_Project/grpo_vla")
DATA_DIR = GRPO_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _png_bytes(img) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def _flatten_images(mm: Dict[str, Any]) -> List[bytes]:
    out: List[bytes] = []
    for clip in (mm.get("video") or []):
        for img in clip:
            out.append(_png_bytes(img))
    for img in (mm.get("image") or []):
        out.append(_png_bytes(img))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", choices=["train", "val"], required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--chunk-size", type=int, default=256)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_path = Path(args.out) if args.out else (
        DATA_DIR / f"nusc_planning_{args.split}.parquet"
    )
    if out_path.exists() and not args.force:
        print(f"[build_parquet] {out_path} exists; skip (use --force to rebuild)")
        return 0

    sys.path.insert(0, str(GRPO_DIR))
    sys.path.insert(0, str(GRPO_DIR.parent))
    try:
        from grpo_vla.dataset_adapter import build_verl_nuscenes_dataset  # type: ignore
    except Exception:
        from dataset_adapter import build_verl_nuscenes_dataset  # type: ignore

    import yaml
    import pandas as pd

    try:
        from transformers import AutoProcessor
    except Exception:
        print("ERROR: transformers not available")
        return 1

    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}
    dcfg = cfg.get("data", cfg)
    proc_path = (
        cfg.get("actor_rollout_ref", {}).get("model", {}).get("path")
        or cfg.get("model_path")
    )
    if not proc_path:
        print("ERROR: actor_rollout_ref.model.path missing from config")
        return 1
    proc = AutoProcessor.from_pretrained(proc_path, trust_remote_code=True)
    ds = build_verl_nuscenes_dataset(dcfg, proc, split=args.split)

    n_total = len(ds) if args.limit < 0 else min(len(ds), args.limit)
    print(f"[build_parquet] split={args.split} n_total={n_total} -> {out_path}")

    # Stream in chunks to avoid OOM; pandas append-via-pyarrow.
    import pyarrow as pa
    import pyarrow.parquet as pq

    writer = None
    chunk_rows: List[Dict[str, Any]] = []
    t0 = time.time()

    def _flush(rows: List[Dict[str, Any]]):
        nonlocal writer
        if not rows:
            return
        df = pd.DataFrame(rows)
        tbl = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(str(out_path), tbl.schema, compression="zstd")
        writer.write_table(tbl)

    for i in range(n_total):
        try:
            s = ds[i]
        except Exception as e:
            print(f"  skip i={i}: {e}", file=sys.stderr)
            continue
        row = {
            "prompt": s["prompt"],
            "images": _flatten_images(s.get("multi_modal_data", {}) or {}),
            "extra_info": json.dumps(s.get("extra_info", {}), default=str),
            "ground_truth": json.dumps(s.get("ground_truth", []), default=str),
            "data_source": "nusc_planning",
        }
        chunk_rows.append(row)
        if len(chunk_rows) >= args.chunk_size:
            _flush(chunk_rows)
            chunk_rows.clear()
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-3)
            eta_min = (n_total - i - 1) / max(rate, 1e-3) / 60
            print(f"  i={i+1}/{n_total}  {rate:.1f} rows/s  ETA {eta_min:.1f} min")

    _flush(chunk_rows)
    if writer is not None:
        writer.close()

    size_mb = out_path.stat().st_size / 1e6 if out_path.exists() else 0
    print(f"[build_parquet] DONE {out_path} ({size_mb:.0f} MB, {n_total} rows, "
          f"{time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
