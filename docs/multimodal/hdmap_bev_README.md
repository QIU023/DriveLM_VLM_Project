# HD-Map BEV renderer (phase13 multi-modal VLA prep, modality 1)

Pre-renders each nuScenes keyframe's surrounding HD map into a 224x224 RGB BEV
PNG with the ego vehicle always pointing up. The cache lets the training
dataloader serve a map "image" with the same shape/budget as a camera frame.

CPU-only. No model loaded. Cache is rebuilt by re-running the script; idempotent
(skips already-rendered samples on disk).

## Script

`scripts/prep_hdmap_bev.py`

```bash
python3 scripts/prep_hdmap_bev.py \
    --infos-pkl data/uniad_infos/nuscenes_infos_temporal_train.pkl \
    --maps-root data/nuscenes \
    --output-dir data/preproc/hdmap_bev \
    --size 224 --range 50.0 --num-workers 8
```

Repeat for `nuscenes_infos_temporal_val.pkl` writing to the same
`--output-dir` (sample tokens never collide across train/val).

Smoke (`--max-samples 1 --num-workers 1`) renders one frame and exits, used to
verify the data layout before running the full sweep.

## Flags

| flag | default | meaning |
|---|---|---|
| `--infos-pkl` | required | UniAD-style temporal infos pkl |
| `--maps-root` | required | nuScenes root (expects `maps/expansion/*.json` and `v1.0-trainval/*.json`) |
| `--meta-root` | `<maps-root>/v1.0-trainval` | dir with `scene.json` and `log.json` |
| `--output-dir` | required | per-sample PNG output dir |
| `--size` | 224 | canvas side (square) |
| `--range` | 50.0 | half-edge in metres -> 100m x 100m crop @ ~0.45 m/px |
| `--num-workers` | 8 | mp.Pool workers, each holds its own `NuScenesMap` cache |
| `--max-samples` | None | cap for smoke runs |

## Map data requirement

The script uses `nuscenes.map_expansion.map_api.NuScenesMap.get_map_mask`
under the hood. The devkit looks for files at:

```
<maps-root>/maps/expansion/<location>.json
```

for the four nuScenes locations:

- `singapore-onenorth`
- `singapore-hollandvillage`
- `singapore-queenstown`
- `boston-seaport`

If absent, the script aborts with exit code 3 BEFORE spawning any workers:

```
ERROR: Map expansion dir not found: <maps-root>/maps/expansion
Hint: download nuScenes Map Expansion v1.3 and unpack into <maps-root>/maps/expansion/
```

The 67MB map-expansion archive is a free public download from
[nuscenes.org/download](https://www.nuscenes.org/download) (account required;
it ships separately from the camera/lidar tar that we already mirror via
HuggingFace). Once unpacked the directory tree should look like:

```
data/nuscenes/
├── maps/
│   └── expansion/
│       ├── boston-seaport.json
│       ├── singapore-hollandvillage.json
│       ├── singapore-onenorth.json
│       └── singapore-queenstown.json
└── v1.0-trainval/
    ├── log.json
    ├── map.json
    ├── scene.json
    └── ...
```

## Cache format

- One PNG per sample: `data/preproc/hdmap_bev/{sample_token}.png`
- RGB, 224x224 (`size`x`size`), uint8, vehicle centred and facing up
- ~5-30 KB per PNG (sparse layers compress very well)

## Layer / colour code

Layers are drawn back-to-front (later layers paint over earlier ones):

| layer | RGB | role |
|---|---|---|
| `road_segment` | (80, 80, 100) | road plane |
| `drivable_area` | (100, 100, 100) | drivable surface |
| `walkway` | (60, 120, 60) | walkway polygon |
| `carpark_area` | (140, 100, 60) | parking polygon |
| `lane` | (70, 200, 100) | lane polygon |
| `ped_crossing` | (220, 60, 60) | crosswalk |
| `stop_line` | (200, 0, 0) | stop-line bar |
| `road_divider` | (255, 255, 255) | white between-road divider |
| `lane_divider` | (255, 240, 80) | yellow between-lane divider |

The nuScenes `traffic_light` layer is dropped (point geometry only; visual
contribution <0.05 px/sample under our colour map after rasterising). If
needed later, add to `LAYER_ORDER` and `LAYER_COLOURS` in the script.

## Loading at training time

```python
from PIL import Image
import numpy as np

img = np.array(Image.open(f"data/preproc/hdmap_bev/{sample_token}.png"))
# img.shape == (224, 224, 3), uint8, vehicle centre at (112, 112) facing up.
```

## Disk budget

29 K samples (28 130 train + 6 019 val) x ~10-30 KB/PNG = ~0.3-1.0 GB total.
Well under the 20 GB project budget; no special storage handling required.

## Open issues / known limitations

- The script depends on the offline nuScenes Map Expansion v1.3 JSONs; not
  obtainable via the project's existing HuggingFace mirror. Caller must place
  them at `<maps-root>/maps/expansion/*.json` before running the full sweep.
- Yaw is extracted from `ego2global_rotation` quaternion in the infos pkl. If
  a future infos pkl drops that field, fall back to `lidar2ego_rotation` or
  the CAN-bus log; the script raises a per-sample error in that case.
- Multiprocessing context is `spawn` so the child processes don't inherit any
  torch / numpy state from the parent (relevant if this script is run from a
  notebook that has imported heavy DL libs).
- Rendering uses `get_map_mask`, not `render_map_patch`, to skip matplotlib
  overhead — ~3-10x faster per sample on 8-core CPUs.
