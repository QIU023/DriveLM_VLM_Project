# LiDAR BEV occupancy cache (phase13 multi-modal VLA prep, modality 3)

Per-keyframe frozen BEV feature for the full-modal AD-VLA. The cache lets the
training dataloader serve a `(8, 128, 128)` BEV tensor next to the camera
embeddings; a small learnable projector then maps it into the LM context as
~64 visual tokens. **The encoder is frozen by construction** — there are no
learnable parameters in the BEV computation itself, so the cache only needs
to be regenerated when the encoding hyper-params change (range, grid, z-slice
edges).

## TL;DR

```bash
python3 scripts/prep_lidar_bev.py \
    --infos-pkl  data/nuscenes/_hf_meta/nuscenes_mmdet3d-12Hz/nuscenes_interp_12Hz_infos_train.pkl \
    --lidar-root data/nuscenes \
    --output-dir data_processed/lidar_bev_occ_v1/train \
    --encoder    occupancy \
    --device     cpu \
    --workers    8
```

Repeat with `*_infos_val.pkl` writing to the same `data_processed/lidar_bev_occ_v1/val/`
output dir (sample tokens never collide across splits).

For a 1-sample smoke before launching the full run:

```bash
python3 scripts/_smoke_lidar_bev_roundtrip.py
```

(auto-picks any `.pcd.bin` under `data/nuscenes/samples/LIDAR_TOP_smoke/` or
`data/nuscenes/samples/LIDAR_TOP/`).

## Encoder choice

| candidate | status | reason |
|---|---|---|
| `pointpillars` (mmdet3d zoo) | **NOT INSTALLABLE on this box** | torch 2.11 + cu13.0 + RTX 5090 (sm_120 / Blackwell). No prebuilt mmcv / mmdet3d / spconv wheel exists for this combo. Source builds of mmcv 2.x fail with `ModuleNotFoundError: pkg_resources` plus subsequent CUDA-op compile errors against cu13. We refuse to slap workarounds (see CLAUDE.md `feedback_no_lazy_shortcuts`). |
| `centerpoint`  (mmdet3d zoo) | NOT INSTALLABLE — same reason | |
| `occupancy` (hand-rolled)   | **chosen** | Pure numpy, zero install risk, deterministic, no learnable params, ~50 samples/sec/worker on CPU. |

If/when Blackwell-compatible mmdet3d wheels ship, add an `--encoder pointpillars`
branch in `prep_lidar_bev.py` that loads the nuScenes detection checkpoint and
extracts the BEV feature map (the projector head can be discarded). Until
then, **the occupancy encoder is what we ship**, and we document that it is
NOT a pretrained detector — it is a hand-rolled multi-channel BEV summary
that the downstream learnable projector must interpret.

## Output format

For each keyframe with `info['token'] == t`, the script writes
`<output-dir>/<t>.npz` with a single key `bev` of shape `(8, 128, 128)`,
dtype `float16`. Observed compressed size: ~14 KB/sample (the occupancy
planes are very sparse, so zlib does excellent work).

### Channels (C = 8)

| idx | content | z-range (m) | normalization |
|---|---|---|---|
| 0 | occupancy mask | [-5, -3) | {0, 1} |
| 1 | occupancy mask | [-3, -1) | {0, 1} |
| 2 | occupancy mask | [-1, 1) | {0, 1} |
| 3 | occupancy mask | [1, 2) | {0, 1} |
| 4 | occupancy mask | [2, 3) | {0, 1} |
| 5 | max intensity in cell | all | intensity / 255 -> [0, 1] |
| 6 | max height in cell    | all | (z + 5) / 8 -> [0, 1] |
| 7 | log(1 + density)      | all | clamped to [0, 1] |

Z-slice edges chosen so the road surface (~0 m above ego) sits inside slice 2
and roof-level returns (~2-3 m) sit in slice 4. Range `x, y in [-50, 50] m`
matches PointPillars / nuScenes detection. Grid 128 -> 0.78 m / cell.

### Per-sample size

```
8 channels x 128 x 128 x 2 B (fp16) = 256 KB   raw
~ 14 KB                              compressed (.npz, occupancy is sparse)
```

## Cache size estimate

Measured on 7 real keyframes pulled from `v1.0-trainval01_blobs.tgz`:
mean compressed size ≈ 14 KB / file.

| split                  | samples | est. cache (compressed) |
|---|---|---|
| UniAD 2 Hz train       | 28,130  | ~480 MB |
| UniAD 2 Hz val         |  6,019  | ~94 MB  |
| **UniAD 2 Hz total**   | 34,149  | **~540 MB** |
| mmdet3d-12Hz train     | 165,280 | ~2.6 GB |
| mmdet3d-12Hz val       | 35,364  | ~550 MB |
| **mmdet3d-12Hz total** | 200,644 | **~3.2 GB** |

Both well under the 10-15 GB disk budget (see CLAUDE.md disk-panic protocol).
Either set is safe to materialize when /workspace has >= 20 GB free.

## Throughput / time estimate

Real smoke at `--workers 2`: 41 samples/sec. Scaling to `--workers 8` on
this 8-core box: ~150-200 samples/sec sustained.
- UniAD 2Hz (~34K samples): ~3-4 minutes wall time
- mmdet3d-12Hz (~200K samples): ~18-25 minutes wall time

Both are CPU-only and do not interfere with GPU training. They DO load the
LiDAR `.bin` files from disk, so avoid running concurrent with a heavy disk
job.

## Frozen-encoder verification

The smoke (`scripts/_smoke_lidar_bev_roundtrip.py`) calls the encoder twice
on the same input and asserts byte-for-byte equality. There are no
`torch.nn.Module` weights, no RNG, no quantization noise. The cache is a
deterministic function of `(point_cloud, encoder_hparams)`, so:

1. The VLA training loop **must not** treat this as a trainable encoder.
2. The downstream projector (BEV -> ~64 LM tokens) IS trainable; it should
   be the only thing that updates gradients off the BEV path.
3. If we ever swap to PointPillars with a learnable backbone, the README in
   that PR must explicitly state the new freeze policy (most likely: freeze
   the pretrained backbone, train only the projector — same policy, just
   richer features).

## LiDAR data prerequisite

nuScenes `samples/LIDAR_TOP/*.pcd.bin` is **not** part of the camera-only
extraction we did so far. Pulling it costs ~25-30 GB additional disk for
the full keyframe-only set (sweeps would 10x that — we never need sweeps for
keyframe BEV).

A streaming pull script that filters keyframes-only out of each blob part:

```bash
# pull part XX, keyframes only:
curl -sS https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval${PART}_blobs.tgz \
  | tar -xz -C data/nuscenes --wildcards 'samples/LIDAR_TOP/*'
```

Defer this until `/workspace` has >= 50 GB free AND A.3 v2 training is done.
The companion shell `scripts/run_lidar_bev_after_a3v2.sh` orchestrates the
pull + encode in the right order with a disk-free guard.

## Boot warnings

The smoke run captures all warnings raised at import + encode time. Current
status: **0 warnings**. If the encoder ever emits a warning (e.g. an empty
point cloud) the smoke logs it and aborts with non-zero exit.

## Do NOT touch from A.3 v2

This script is **read-only** with respect to the A.3 v2 training paths:
- it does NOT import `train_lora.py`, `planning_dataset.py`, or any
  `configs/` entry,
- it does NOT touch `checkpoints_qwen25/`,
- it writes only to `data_processed/lidar_bev_occ_v1/`.

Safe to run during A.3 v2 if you stay on CPU (default).
