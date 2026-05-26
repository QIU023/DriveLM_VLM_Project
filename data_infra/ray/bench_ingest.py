#!/usr/bin/env python3
"""Benchmark: legacy multiprocessing.Pool ingestion (A) vs Ray Data DAG (B).

Runs BOTH paths on the SAME N-sample subset (same split, same seed so the
random sub-sample picks the same nuScenes samples) and reports wall-clock +
rows/sec.

  A = grpo_vla/build_parquet.py    (multiprocessing.Pool(workers))
  B = data_infra/ray/ray_ingest.py (ray.data lazy streaming DAG)

Both are launched as SUBPROCESSES (clean process per run; no shared interpreter
state, no Ray<->multiprocessing fork interference) and timed end-to-end
(process start -> exit, the wall-clock a user actually waits).

Usage:
    /usr/bin/python3 bench_ingest.py --n 2000 --workers 8
    /usr/bin/python3 bench_ingest.py --n 16 --workers 8   # smoke
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "/workspace/DriveLM_VLM_Project"
PY = "/usr/bin/python3"
GRPO_CFG = os.path.join(REPO, "grpo_vla", "configs", "grpo_b5prime_3cam.yaml")


def _run(cmd, cwd, env, log_path):
    t0 = time.time()
    with open(log_path, "w") as lf:
        p = subprocess.run(cmd, cwd=cwd, env=env, stdout=lf,
                           stderr=subprocess.STDOUT)
    return time.time() - t0, p.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--split", default="val")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-edge", type=int, default=448)
    ap.add_argument("--jpeg-q", type=int, default=75)
    args = ap.parse_args()

    env = dict(os.environ)
    env["HF_HOME"] = "/workspace/.hf_home"
    env.pop("HF_HUB_OFFLINE", None)

    results = {}

    # ---- A: legacy multiprocessing.Pool ----
    a_out = os.path.join(HERE, "bench_legacy.parquet")
    if os.path.exists(a_out):
        os.remove(a_out)
    a_cmd = [
        PY, os.path.join(REPO, "grpo_vla", "build_parquet.py"),
        "--config", GRPO_CFG, "--split", args.split,
        "--out", a_out, "--max-samples", str(args.n),
        "--workers", str(args.workers),
        "--max-edge", str(args.max_edge), "--jpeg-q", str(args.jpeg_q),
        "--seed", str(args.seed), "--force",
    ]
    print(f"[bench] A (multiprocessing.Pool, workers={args.workers}) ...")
    a_wall, a_rc = _run(a_cmd, REPO, env, os.path.join(HERE, "bench_A.log"))
    a_rows = _count_parquet(a_out)
    results["A"] = (a_wall, a_rows, a_rc)
    print(f"[bench] A done rc={a_rc} wall={a_wall:.1f}s rows={a_rows}")

    # ---- B: Ray Data DAG ----
    b_out = os.path.join(HERE, "bench_ray.parquet")
    b_cmd = [
        PY, os.path.join(HERE, "ray_ingest.py"),
        "--split", args.split, "--n", str(args.n),
        "--out", b_out, "--format", "parquet",
        "--seed", str(args.seed), "--num-cpus", str(args.workers),
        "--concurrency", str(args.workers),
        "--max-edge", str(args.max_edge), "--jpeg-q", str(args.jpeg_q),
    ]
    print(f"[bench] B (ray.data DAG, concurrency={args.workers}) ...")
    b_wall, b_rc = _run(b_cmd, HERE, env, os.path.join(HERE, "bench_B.log"))
    b_rows = _count_parquet(b_out)
    results["B"] = (b_wall, b_rows, b_rc)
    print(f"[bench] B done rc={b_rc} wall={b_wall:.1f}s rows={b_rows}")

    # ---- report ----
    def rate(w, r):
        return r / w if w > 0 else 0.0

    aw, ar, _ = results["A"]
    bw, br, _ = results["B"]
    speedup = (rate(bw, br) / rate(aw, ar)) if rate(aw, ar) > 0 else float("nan")

    print("\n" + "=" * 64)
    print(f"BENCH  split={args.split}  n={args.n}  workers={args.workers}  "
          f"seed={args.seed}")
    print("=" * 64)
    print(f"{'path':<34}{'wall(s)':>10}{'rows':>8}{'rows/s':>12}")
    print("-" * 64)
    print(f"{'A multiprocessing.Pool':<34}{aw:>10.1f}{ar:>8}{rate(aw, ar):>12.2f}")
    print(f"{'B ray.data streaming DAG':<34}{bw:>10.1f}{br:>8}{rate(bw, br):>12.2f}")
    print("-" * 64)
    print(f"B/A throughput speedup: {speedup:.2f}x")
    print("=" * 64)
    return 0


def _count_parquet(path):
    if not os.path.exists(path):
        return 0
    try:
        import pyarrow.parquet as pq
        if os.path.isdir(path):
            import glob
            files = glob.glob(os.path.join(path, "*.parquet"))
            return sum(pq.ParquetFile(f).metadata.num_rows for f in files)
        return pq.ParquetFile(path).metadata.num_rows
    except Exception as e:  # pragma: no cover
        print(f"[bench] count error {path}: {e}", file=sys.stderr)
        return 0


if __name__ == "__main__":
    sys.exit(main())
