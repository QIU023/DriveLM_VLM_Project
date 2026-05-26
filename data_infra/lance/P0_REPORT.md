# P0 — Lance columnar lakehouse + cached vision tokens + throughput A/B

nuScenes multimodal VLA (Qwen3-VL-4B planning checkpoint). Goal: demonstrate a
Lance-columnar dataset, a derived cached-vision-token column, and a real
DataLoader throughput A/B that proves caching the ViT output off the per-step
critical path. Built at subset scale (3000 train samples) to validate the
pattern within disk budget.

## Files (all under `data_infra/lance/`)

| File | Role |
|---|---|
| `build_lance_dataset.py` | Convert a subset of nuScenes multimodal → `nusc_mm.lance` (paths, not pixels, for cameras). |
| `cache_vision_tokens.py` | Run the ViT over each clip, FasterVLM x4, int8-quantize, write 3 columns. Multi-GPU (1 proc/GPU, shard by row idx). |
| `lance_dataset.py` | `torch.utils.data.Dataset` reading cached int8 tokens by row idx (zero-copy `take` + dequant). The train-time path that SKIPS the ViT. |
| `bench_throughput.py` | A/B throughput: baseline `MultiModalPlanningDataset` (JPEG+processor+ViT) vs Lance cached. |
| `nusc_mm.lance` | The dataset (3000 rows). |

## Dataset

`nusc_mm.lance` — 3000 train samples (first 3000 full-future keyframes, matching
the trainer's `require_full_future` filter so the subset == the trainer's first
N samples).

### Schema (one row per sample)

| column | type | notes |
|---|---|---|
| `sample_token` | string | nuScenes keyframe token (Lance join key) |
| `scene_token` | string | parent scene |
| `timestamp` | int64 | lidar canonical µs |
| `scenario` | string | ego-motion bucket — **reuses `scripts/planning_eval.classify_scenario`** |
| `hdmap_png` | binary | 224×224 HD-map BEV PNG bytes (black substitute on 29 cache misses) |
| `bbox_text` | string | 3D bbox + ego text from `bbox_egostate_train.jsonl` |
| `ego_speed` | float32 | CAN-bus speed scalar (m/s) |
| `traj_gt` | list<float32> | future xy waypoints, flattened `[x0,y0,…,x5,y5]` (6 wp) |
| `camera_paths` | list<string> | **absolute paths** to the 4 past CAM_FRONT frames (paths, not pixels) |
| `vis_tokens_int8` | binary | row-major int8 vision tokens, shape `vis_tokens_shape` |
| `vis_tokens_scale` | float32 | per-tensor symmetric scale; `dequant = int8·scale` |
| `vis_tokens_shape` | list<int64> | `[700, 2560]` |

Scenario histogram (3000): `straight 1393, turning 490, stationary 349,
lane_change 303, braking 288, cruising 177`.

### Why paths, not camera pixels
The JPEGs already live on disk; duplicating them as blobs would balloon the file
with no access-pattern win. The lakehouse value is the **derived** artifact — the
int8 vision-token cache — which IS stored columnar and is what the train-time
path reads.

## Vision-token cache

- Tower: `model.model.visual` via `model.model.get_video_features(pixel_values_videos, video_grid_thw)`, bf16, from the deploy ckpt `checkpoints_qwen25/nusc_planning_b5pp_1cam_qwen3vl_multimodal/final`.
- 1-cam native: video grid `[2,56,100]` → pooler `(2800, 2560)`.
- **FasterVLM x4** via `scripts/visual_compress.compress_visual_tokens(method="fastervlm", ratio=4)` — top-K by L2 norm, the repo's documented training-free CLS-attention proxy. 2800 → **700** tokens.
- int8 symmetric per-tensor quant: `s = max|x|/127`. **Round-trip dequant error 3.2–3.6%** relative (per-shard max, asserted in `--verify`).
- 8-GPU sharded: 3000 rows in **~4 min** wall.

Per-row token cache = **1.79 MB** (700×2560 int8). Compare bf16 uncompressed
pooler = 2800×2560×2 ≈ 14.3 MB/sample → the cache is ~8x smaller than the raw
bf16 ViT output (x4 token prune × ~2x int8-vs-bf16).

## On-disk size: Lance vs raw

| artifact | bytes |
|---|---|
| `nusc_mm.lance` total (3000 rows, incl. vis tokens + hdmap PNGs + meta) | **3.83 GB** |
| └ vis-token cache portion (3000 × 1.79 MB) | ~5.1 GB logical → 3.8 GB on disk (Lance packs/compacts) |
| └ HD-map PNG blobs in-file | 8.99 MB |
| └ meta (tokens/traj/text/paths) | ~10 MB |
| raw CAM_FRONT JPEGs the baseline decodes — unique current frames (3000) | 449 MB |
| raw CAM_FRONT JPEGs — all 4-frame refs the ViT actually consumes (with overlap) | 1.80 GB |

The Lance file (3.83 GB) is larger than the raw JPEG bytes (0.45–1.8 GB) because
it *adds* the derived 700×2560 int8 token tensor per row — that is the point: we
trade disk for never re-running the ViT. The win is on the time axis, below.

## Throughput A/B (real DataLoader, num_workers=8, pin_memory, prefetch=2, 200 steady-state steps, batch=4, 1×A100-class GPU)

| metric | A: baseline (JPEG+proc+**ViT** per sample) | B: Lance (cached int8 tokens, **no ViT**) |
|---|---:|---:|
| **samples/sec** | **8.76** | **274.42** |
| p50 batch latency | 450.1 ms | 12.7 ms |
| p90 batch latency | 506.1 ms | 32.7 ms |
| GPU-idle / stall % | 0.1% | 92.1% |
| **speedup (B/A)** | — | **31.3×** |

### Reading the stall numbers (honest interpretation)
- **Baseline stall 0.1%** — the GPU is *saturated* (99.9% busy) re-running the ViT vision tower on every step. The bottleneck is on-GPU recompute, not data fetch.
- **Lance stall 92.1%** — the vision stage is now a trivial int8→GPU copy (~0.8 ms/batch), so the GPU is 92% *idle waiting for data*. That idle is the headroom freed up: caching the ViT output removes it from the per-step critical path, so the GPU is free for the downstream LM forward/backward that the real train loop would do.

The 31× figure is the throughput of the *vision-feature stage in isolation*. End-to-end train speedup is bounded by the LM stage (not benchmarked here), but the data-infra result stands: the redundant per-epoch ViT recompute is eliminated.

## Honest scaling note
This validates the pattern at **subset scale (3000 / 28130 train ≈ 11%)**.
Full-corpus token cache extrapolates to **~50 GB** (1.79 MB/row × 28130) for the
train split alone, plus ~11 GB for the 6019-sample val split — ~61 GB total. That
exceeds the current ~53 GB free `/workspace` headroom, so the full cache would
need either more disk or a lower-precision/higher-prune config (e.g. FasterVLM x8
→ 350 tok → ~0.9 MB/row → ~25 GB train). The subset proves the build → cache →
zero-copy-read → throughput-win loop works; full-corpus is a disk-provisioning
decision, not a code change (`build_lance_dataset.py --n` + re-run the cache).

## Fallbacks taken
- **FasterVLM**: used the repo's existing `compress_visual_tokens(method="fastervlm")` — top-K by L2 norm. This IS the documented FasterVLM CLS-attention proxy in this codebase (no separate trt_b5ppp hook was needed). No fallback to an ad-hoc top-k was required; the existing hook applied directly to the pooler output by passing a flat `[1,1,N]` grid.
- Lance is not fork-safe → the bench DataLoader for Path B uses a `forkserver` multiprocessing context; workers re-open the dataset lazily.
