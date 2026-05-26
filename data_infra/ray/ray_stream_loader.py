#!/usr/bin/env python3
"""DP-rank-sharded streaming prefetch over the ingested dataset.

Demonstrates how N trainer ranks (data-parallel) each pull a DISJOINT,
BALANCED stream from one logical Ray Dataset using
``Dataset.streaming_split(n=WORLD, equal=True)``.

This is the loader side of the pipeline: in real training each rank would call
``shard.iter_torch_batches(...)`` in its own process. Here we run all WORLD
shard iterators in the driver (CPU-only) and count rows + collect sample_tokens
per shard to PROVE:

  1. disjoint  : the per-shard token sets do not overlap (intersection == 0)
  2. balanced  : with equal=True every shard yields the same row count
  3. complete  : union of shards == every row (no drops, no dupes)

Usage:
    /usr/bin/python3 ray_stream_loader.py --path out.parquet --world 4
    /usr/bin/python3 ray_stream_loader.py --path out_smoke.parquet --world 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import ray


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="out.parquet",
                    help="ingested dataset (parquet dir/file or .lance)")
    ap.add_argument("--world", type=int, default=4, help="number of DP ranks")
    ap.add_argument("--num-cpus", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    path = args.path if os.path.isabs(args.path) else os.path.join(here, args.path)
    if not os.path.exists(path):
        print(f"[stream_loader] ERROR: dataset not found: {path}", file=sys.stderr)
        return 1

    ray.init(num_cpus=args.num_cpus, ignore_reinit_error=True,
             include_dashboard=False, log_to_driver=False)
    try:
        if path.endswith(".lance"):
            ds = ray.data.read_lance(path)
        else:
            ds = ray.data.read_parquet(path)
        total = ds.count()
        print(f"[stream_loader] dataset rows={total} world={args.world}")

        # The core API: split the dataset into WORLD disjoint, equal streams.
        # equal=True trims the tail so every shard has exactly floor(N/WORLD)
        # rows -> perfectly balanced (drops up to WORLD-1 rows, standard DDP
        # behavior to keep ranks in lockstep).
        shards = ds.streaming_split(n=args.world, equal=True)

        # Each rank consumes its shard as a stream of batches. IMPORTANT:
        # streaming_split routes blocks to shards round-robin and expects ALL
        # shards to be consumed CONCURRENTLY (exactly as N independent trainer
        # ranks would). Draining shard 0 fully before shard 1 deadlocks, so we
        # iterate every shard in its own thread — one thread == one DP rank.
        import threading

        per_shard_tokens = [None] * args.world
        per_shard_counts = [0] * args.world
        per_shard_first = [None] * args.world

        def _consume(rank, shard):
            toks = []
            for batch in shard.iter_batches(
                batch_size=args.batch_size, batch_format="pandas"
            ):
                toks.extend(batch["sample_token"].tolist())
            per_shard_tokens[rank] = set(toks)
            per_shard_counts[rank] = len(toks)
            per_shard_first[rank] = toks[0] if toks else "-"

        threads = [
            threading.Thread(target=_consume, args=(r, s))
            for r, s in enumerate(shards)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for rank in range(args.world):
            print(f"  rank {rank}: rows={per_shard_counts[rank]}  "
                  f"first_token={per_shard_first[rank]}")

        # ---- disjointness + balance proof ----
        union = set()
        overlap_total = 0
        for r in range(args.world):
            overlap_total += len(union & per_shard_tokens[r])
            union |= per_shard_tokens[r]
        # pairwise overlap explicit
        pairwise = 0
        for a in range(args.world):
            for b in range(a + 1, args.world):
                pairwise += len(per_shard_tokens[a] & per_shard_tokens[b])
        # dupes within union vs sum
        sum_counts = sum(per_shard_counts)
        dupes = sum_counts - len(union)

        print("\n[stream_loader] === SHARDING PROOF ===")
        print(f"  per-shard counts        : {per_shard_counts}")
        print(f"  balanced (all equal?)   : {len(set(per_shard_counts)) == 1}")
        print(f"  pairwise token overlap  : {pairwise}  (disjoint iff 0)")
        print(f"  total rows consumed     : {sum_counts}")
        print(f"  unique tokens (union)   : {len(union)}")
        print(f"  duplicate tokens        : {dupes}  (no-dupe iff 0)")
        print(f"  dataset total           : {total}")
        print(f"  dropped by equal=True   : {total - sum_counts} "
              f"(expected < world={args.world})")

        ok = (
            pairwise == 0
            and dupes == 0
            and len(set(per_shard_counts)) == 1
            and (total - sum_counts) < args.world
        )
        print(f"\n[stream_loader] DISJOINT+BALANCED+COMPLETE: {ok}")
        return 0 if ok else 2
    finally:
        ray.shutdown()


if __name__ == "__main__":
    sys.exit(main())
