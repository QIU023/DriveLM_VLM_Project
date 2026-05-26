# P2 — Lance/LanceDB vector index for semantic scenario retrieval (2026-05-26)

Data-centric-AI evidence: semantic indexing + vector DB over the nuScenes VLA, for
rare-event mining / curriculum / dedup. **Reuses P0's cached ViT tokens (no GPU)** —
dequantize int8 `vis_tokens` (700×2560), mean-pool → 2560-d clip embedding,
L2-normalize, index in LanceDB with an IVF_PQ(16×64, cosine) index. 3000 clips.

## Results

**Label-consistency recall@k** (leave-one-out; fraction of k nearest neighbours sharing
the query's scenario label). Random baseline = Σ p² over scenario priors = 0.279.

| k | micro recall | macro recall | vs random |
|---|---|---|---|
| 5 | **0.743** | 0.652 | **2.7×** |
| 10 | 0.649 | 0.545 | 2.3× |
| 20 | 0.554 | 0.438 | 2.0× |

→ The camera embedding genuinely encodes driving-scenario structure: a clip's neighbours
in vector space are 2–3× more likely to be the same maneuver than chance → the index is
usable for retrieval-based mining/curriculum.

**Rare-scenario mining** (query = class centroid, precision@20):

| class | base rate | precision@20 | lift |
|---|---|---|---|
| lane_change | 0.101 | 0.250 | 2.5× |
| braking | 0.096 | 0.200 | 2.1× |
| cruising | 0.059 | **0.000** | **0×** |

**Honest finding — cruising is unmineable from camera alone, and that is consistent with
the project's central thesis.** "Cruising" (steady moderate speed) is *visually
indistinguishable* from "straight"; it differs only by **ego speed**, which is not in the
camera embedding. So the centroid query returns straight neighbours → 0 precision. The
classes that ARE visually distinct (turning, lane_change, braking) mine well. This mirrors
the deploy-side finding that nuScenes planning signal lives in ego/HD-map priors, not
camera pixels — a camera-only index can't recover speed-defined classes. Fix would be to
concatenate ego-state scalars into the index vector (multi-modal embedding).

**Near-duplicate dedup**: 4043 clip pairs with cosine > 0.995 (0.090% of all pairs) — these
are consecutive keyframes within a scene; a dedup/temporal-stride pass would drop them to
cut redundant training compute.

## Why this matters for PB-scale
The index is built from already-cached embeddings (zero extra GPU), so at fleet scale you
maintain one ANN index over all logs and serve "find me cut-ins / hard-brakes / rare
maneuvers" queries for active-learning data selection. Validation-scale here (3000 clips);
the LanceDB IVF_PQ index scales to billions of vectors. To make speed-defined classes
mineable, fuse ego-state into the embedding.

Artifacts: `build_vector_index.py`, `p2_summary.json`, `vecdb/` (LanceDB, gitignored).
