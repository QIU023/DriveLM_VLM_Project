# Data Infrastructure for nuScenes Multimodal VLA — lakehouse + distributed dataloader

Built to demonstrate the data-engineering stack for a PB-scale AD / multimodal-LLM data
role (Iceberg + Lance lakehouse, Ray pipelines, training-throughput optimization, semantic
indexing). **Validation-scale on nuScenes (~34K keyframes, 8× RTX 5090); architected to
scale to PB / 10k-GPU.** All numbers are measured, with honest caveats.

| module | what it shows | headline (measured) | dir |
|---|---|---|---|
| **P0** Lance + cached ViT tokens | AI-optimized columnar format, PyTorch random-access, throughput opt | **31.3× train throughput** (8.76 → 274 samp/s) by caching post-FasterVLM int8 tokens & skipping the ViT | `lance/` |
| **P1** Ray Data DAG | Ray, scalable streaming ingestion, back-pressure, sharding | steady-state **62.5 rows/s > Pool 55.8**; sharded loader disjoint+balanced+complete | `ray/` |
| **P2** LanceDB vector index | semantic indexing, vector DB, data-centric AI | scenario **recall@5 0.743 vs 0.279 random (2.7×)**; rare-mining lane_change 2.5× | `vecindex/` |
| **P3** Iceberg catalog | Iceberg, metadata, versioning, schema evolution, lineage | time-travel + schema-evo + partition-prune + lineage; snapshot **== scored data (exact)** | `iceberg/` |

## How each maps to the JD
- **Modern Lakehouse (Iceberg + Lance)** → P3 Iceberg catalog (metadata/versioning/time-travel/
  schema-evo) + P0 Lance columnar store. Semantic indexing → P2.
- **Scalable Data Pipelines (PB-scale ingest/clean/process)** → P1 Ray Data streaming DAG
  (raw nuScenes → model-ready rows), constant-memory + back-pressure → multi-node pattern.
- **Training Throughput Optimization** → P0: cache the 31-TFLOP / 80%-of-FLOPs ViT stage as
  int8 Lance columns; shard-aware streaming (P1) feeds DP ranks disjointly → 31× dataloader
  throughput, dataloader stall 0.1% → 92% (now host/IO-bound, the right target to scale next).
- **Raw logs → model-ready tokens** → P0 build + token cache; P3 lineage ties data snapshot →
  ckpt → eval L2.

## Honest scope
- Subset/validation scale (P0 3000 rows, P1/P2 2000-3000, P3 5119); full-corpus token cache
  is a disk-provisioning decision (~50-61 GB or FasterVLM×8 ~25 GB), not a code change.
- P1: bare multiprocessing.Pool wins end-to-end on ONE 8-CPU box (Ray cold-start tax); Ray's
  win is off-box (autoscaling/streaming) + steady-state compute already exceeds Pool.
- P2: camera-only embedding can't separate speed-defined classes (cruising) — fuse ego-state
  to fix; consistent with the deploy finding that planning signal is ego/HD-map-prior dominated.

Reproduce: each subdir has its build script + `P*_REPORT.md`. Commits b418d54 / 738981b /
01336fe / ef2f5bc (author Yiqiao Qiu).
