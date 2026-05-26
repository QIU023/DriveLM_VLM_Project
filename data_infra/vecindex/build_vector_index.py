#!/usr/bin/env /usr/bin/python3
"""P2: Lance/LanceDB vector index over clip embeddings for rare-scenario
retrieval + dedup — data-centric-AI evidence (semantic indexing, vector DB).

Reuses P0's cached ViT tokens (no GPU): dequantize the int8 vis_tokens
(700x2560) and mean-pool -> a 2560-d clip embedding per keyframe. Build a
LanceDB table with an IVF_PQ vector index. Evaluate:
  (1) label-consistency recall@k vs the 6 scenario labels (does the embedding
      capture driving semantics? recall@k >> base-rate => yes => mineable),
  (2) rare-scenario mining precision (query a rare class, measure purity of
      top-k), (3) near-duplicate dedup (cosine > thr within a scene).
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np
import lance
import lancedb

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "lance" / "nusc_mm.lance"
RARE = ("cruising", "braking", "lane_change")   # the low-frequency classes


def load_embeddings():
    ds = lance.dataset(str(SRC))
    embs, meta = [], []
    # Stream in small batches: each int8 blob is 1.79MB, so >~1150 rows/batch
    # overflows Arrow's 2GB int32 offset for the binary column.
    for batch in ds.to_batches(columns=["sample_token", "scenario", "scene_token",
                                        "vis_tokens_int8", "vis_tokens_scale",
                                        "vis_tokens_shape"], batch_size=256):
        for r in batch.to_pylist():
            shp = tuple(r["vis_tokens_shape"])           # (700, 2560)
            q = np.frombuffer(r["vis_tokens_int8"], dtype=np.int8).reshape(shp)
            deq = q.astype(np.float32) * float(r["vis_tokens_scale"])
            e = deq.mean(axis=0)                         # (2560,)
            e = e / (np.linalg.norm(e) + 1e-8)          # L2 normalize -> cosine via dot
            embs.append(e)
            meta.append({"sample_token": r["sample_token"], "scenario": r["scenario"],
                         "scene_token": r["scene_token"]})
    return np.stack(embs).astype(np.float32), meta


def recall_at_k(embs, labels, k=10):
    """Leave-one-out label-consistency recall@k (brute-force cosine, 3000 tiny)."""
    sims = embs @ embs.T
    np.fill_diagonal(sims, -1.0)
    nn = np.argsort(-sims, axis=1)[:, :k]
    lab = np.array(labels)
    same = (lab[nn] == lab[:, None]).mean(axis=1)        # per-query frac same-label
    return same, lab


def main():
    t0 = time.time()
    embs, meta = load_embeddings()
    labels = [m["scenario"] for m in meta]
    n, d = embs.shape
    print(f"[p2] {n} clip embeddings, dim={d}, loaded in {time.time()-t0:.1f}s")

    # ---- LanceDB table + IVF_PQ vector index ----
    db = lancedb.connect(str(HERE / "vecdb"))
    if "clips" in db.table_names():
        db.drop_table("clips")
    data = [{"vector": embs[i], "sample_token": meta[i]["sample_token"],
             "scenario": labels[i], "scene_token": meta[i]["scene_token"]}
            for i in range(n)]
    tbl = db.create_table("clips", data=data)
    try:
        tbl.create_index(num_partitions=16, num_sub_vectors=64, metric="cosine")
        idx = "IVF_PQ(16x64)"
    except Exception as e:
        idx = f"brute-force (index skipped: {e})"
    print(f"[p2] LanceDB table 'clips' built; vector index = {idx}")

    # ---- (1) label-consistency recall@k ----
    base = {c: labels.count(c)/n for c in set(labels)}
    for k in (5, 10, 20):
        same, lab = recall_at_k(embs, labels, k=k)
        macro = np.mean([same[lab == c].mean() for c in set(labels)])
        print(f"[p2] recall@{k}: micro={same.mean():.3f}  macro={macro:.3f} "
              f"(random base-rate micro={sum(p*p for p in base.values()):.3f})")

    # ---- (2) rare-scenario mining precision@k ----
    print("[p2] rare-scenario mining precision@20 (query=class centroid):")
    rare_stats = {}
    for c in RARE:
        idxs = [i for i, l in enumerate(labels) if l == c]
        centroid = embs[idxs].mean(0); centroid /= np.linalg.norm(centroid)+1e-8
        hits = tbl.search(centroid).metric("cosine").limit(20).to_list()
        prec = sum(1 for h in hits if h["scenario"] == c) / len(hits)
        rare_stats[c] = {"base_rate": round(base[c], 3), "precision@20": round(prec, 3)}
        print(f"     {c:12s}: base_rate={base[c]:.3f}  precision@20={prec:.3f}  "
              f"(lift {prec/base[c]:.1f}x)")

    # ---- (3) near-duplicate dedup ----
    sims = embs @ embs.T; np.fill_diagonal(sims, -1)
    thr = 0.995
    dup_pairs = int((sims > thr).sum() // 2)
    print(f"[p2] near-dup pairs (cosine>{thr}): {dup_pairs}  "
          f"({100*dup_pairs/(n*(n-1)/2):.3f}% of all pairs)")

    summary = {
        "n_clips": n, "dim": d, "vector_index": idx,
        "recall_at_10_micro": float(recall_at_k(embs, labels, 10)[0].mean()),
        "rare_mining": rare_stats,
        "near_dup_pairs_cos_gt_0.995": dup_pairs,
        "scenario_base_rates": {c: round(v, 3) for c, v in base.items()},
    }
    json.dump(summary, open(HERE / "p2_summary.json", "w"), indent=2)
    print(f"[p2] wrote {HERE/'p2_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
