# 3D bbox + ego-state text serializer (phase13 multi-modal VLA prep)

Modality 2 of the multi-modal autonomous-driving VLA prefix: given a nuScenes
keyframe, produce a compact text description of (a) the surrounding 3D
detection boxes the perception module would output, and (b) the ego vehicle's
current driving state. Concatenated, this becomes the prompt prefix the LM sees
right before predicting the planning trajectory.

Lives in CPU-only preprocessing — no model is run here, so the cache can be
rebuilt on any node in seconds.

## Script

`scripts/prep_bbox_egostate.py`

```
python scripts/prep_bbox_egostate.py \
  --infos-pkl data/uniad_infos/nuscenes_infos_temporal_train.pkl \
  --infos-pkl data/uniad_infos/nuscenes_infos_temporal_val.pkl \
  --output    data/preproc/bbox_egostate_train.jsonl \
  --output    data/preproc/bbox_egostate_val.jsonl \
  --token-audit
```

Flags:
- `--smoke-only`: just runs the 3-sample print + round-trip check (used in CI).
- `--token-audit`: after writing, samples 100 records from the first output and
  reports Qwen2.5-VL tokenizer length stats. Flags if `max > 500 tokens`.

Each output line is:

```json
{"sample_token": "...", "bbox_text": "...", "egostate_text": "..."}
```

## Serialization format

### 3D bboxes

```
Detected objects in ego frame:
- car at (-1.9, -17.0, -1.3) m, size 4.5x1.9x1.5, yaw -1.37 rad, vel (5.2, -1.8) m/s
- truck at (0.1, 24.2, -0.2) m, size 6.2x2.2x2.6, yaw -0.31 rad
- pedestrian at (-14.2, 3.0, -0.6) m, size 0.5x0.7x1.7, yaw -4.18 rad
...
```

Rules:
- Whitelist: `car, truck, bus, pedestrian, motorcycle, bicycle, construction_vehicle, trailer`.
  Skips `barrier`, `traffic_cone`, `movable_object.*`, `static_object.bicycle_rack`,
  `human.pedestrian.stroller` — these don't drive planner behaviour.
- Drops boxes where `valid_flag[i] == False`.
- Sorted by ascending `sqrt(cx**2 + cy**2)` (planar distance from ego).
- Capped at top **K = 10**.
- Velocity field is omitted when `|vel| < 0.3 m/s` (to reduce sensor noise).
- Size order in the text is `LxWxH` (length × width × height) — note the raw
  pkl stores boxes as `[cx, cy, cz, w, l, h, yaw]`, we reorder for display.

### Ego state

```
Ego state: speed 4.49 m/s, yaw rate 0.39 rad/s, prev trajectory (Δx, Δy) over last 4 keyframes: (-7.4, 3.1), (-5.8, 1.8), (-4.1, 0.8), (-2.2, 0.2).
```

- `speed` ← `can_bus[13]` (m/s), clamped to `>= 0` (sensor sometimes reports
  small negative values, see `scripts/planning_dataset.py` header).
- `yaw rate` ← `can_bus[12]` (rotation_rate.z, rad/s).
- `prev trajectory` ← walk the `prev` token chain back up to 4 keyframes and
  project each ego2global translation into the **current** ego frame using
  `p_e = R_curr^T @ (p_prev_world - t_curr)`. Listed oldest → most recent so
  the reader sees temporal order matching the future prediction the LM is
  about to make.
- If `prev` is empty (scene-start frames), the past-trajectory clause is
  replaced with `prev trajectory unavailable`.

## Coordinate-frame note

`gt_boxes` in the UniAD infos pkl is in the **LIDAR frame**, not strictly the
ego frame. The `lidar2ego` transform on every sample is a near-identity
rotation plus a fixed `[+0.94, 0, +1.84]` m translation. The planner targets
(`fut_traj`, the model's regression output) live in the same LIDAR frame, so
treating LIDAR-frame xy directly as "ego frame xy" is consistent with UniAD's
own convention and is what every downstream open-loop nuScenes planner does.
The 0.94 m forward offset of the LIDAR vs the rear-axle frame is negligible
compared to typical object distances (5-100 m).

`gt_velocity` in the pkl has already been rotated into the same frame by the
UniAD preprocessing pipeline — verified by checking that velocity magnitudes
agree with finite-difference object motion in consecutive keyframes.

## Outputs

| File                                       | Lines | Size |
| ------------------------------------------ | ----- | ---- |
| `data/preproc/bbox_egostate_train.jsonl`   | 28130 | ~26M |
| `data/preproc/bbox_egostate_val.jsonl`     | 6019  | ~5M  |

These files are gitignored (whole `data/` tree is in `.gitignore`).

## Token-length audit (Qwen2.5-VL-3B-Instruct tokenizer, N=100)

| Stat   | Tokens |
| ------ | ------ |
| min    | 145    |
| median | 549    |
| p90    | 619    |
| p99    | 654    |
| max    | 655    |

The max exceeds the 500-token "balloon" threshold flagged in the task spec.
This is the cost of carrying TOP_K=10 objects × ~50 tokens/line. If we need
to shrink the prefix later, the cheapest reductions are:
- drop the `yaw N.NN rad` clause (worth ~6 tokens/line × ~10 lines = 60 t),
- drop the `size LxWxH` clause (~8 t/line = 80 t),
- reduce K from 10 → 6 (saves ~4 × 50 = 200 t).

For the initial phase13 ablation we keep the verbose format so the LM sees
the richest possible perception context; a follow-up run can ablate the
trade-off.

## nuScenes pkl quirks worth knowing

1. **Coordinate frame**: `gt_boxes` columns are `[cx, cy, cz, w, l, h, yaw]`
   in LIDAR (≈ ego) frame, **not** `[x, y, z, l, w, h, yaw]` as some other
   nuScenes loaders use.
2. **+y is forward**: in the LIDAR/ego frame, +y points to the front of the
   vehicle, +x points right. Make sure plotting code matches.
3. **Velocity frame**: `gt_velocity` is already rotated into the ego frame
   by the UniAD pipeline (raw nuScenes ships it in the global frame).
4. **`valid_flag`**: some boxes are present in the array but flagged invalid
   (low visibility, etc.); we honor the flag.
5. **`prev`/`next`**: keyframe-level (2 Hz), not sweep-level. A 4-step
   history therefore spans 2 s, which lines up with the model's 3 s future
   horizon at the same cadence.
6. **`can_bus[13]` clamp**: the raw CAN reading occasionally goes slightly
   negative (~-0.8 m/s) due to sensor noise; we clamp at 0.
7. **Class names**: nuScenes detection labels in this pkl are already in
   "simple" form (e.g. `car`, not `vehicle.car`), but a handful retain dotted
   forms (`movable_object.pushable_pullable`, `human.pedestrian.stroller`) —
   our whitelist filters these out anyway.
