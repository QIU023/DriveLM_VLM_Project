# P3 — Local Apache Iceberg Lakehouse over nuScenes VLA Metadata

A self-contained Apache Iceberg lakehouse standing over the nuScenes
vision-language-action (VLA) planning metadata. It demonstrates the four
lakehouse capabilities that matter for a production vehicle-log data platform —
**snapshots / time-travel, schema evolution, hidden-partition pruning, and
lineage** — each with a printed proof, and ties a real model checkpoint
(`b5ppp` 3-cam, open-loop L2 = 0.626) back to the exact data snapshot it was
evaluated on.

No model training. CPU-only. SQLite-backed catalog + local `file://`
warehouse, so the whole thing is reproducible on a laptop.

## Files

| Path | Purpose |
|------|---------|
| `data_infra/iceberg/nusc_extract.py` | Pure-Python (numpy only) extraction of the `keyframes` rows from the `infos` pkl + bbox/ego JSONL. Reproduces `scripts/planning_eval.py::classify_scenario` **byte-for-byte** so the Iceberg `scenario` partition matches the eval taxonomy. |
| `data_infra/iceberg/smoke_iceberg.py` | SMOKE: throwaway catalog, 10-row table, 2-commit snapshot round-trip (6 → 10 → time-travel back to 6). Run first. |
| `data_infra/iceberg/build_iceberg_catalog.py` | Full build: catalog, `nusc_vla` namespace, partitioned `keyframes` table (5119 val rows), `training_runs` lineage table, and the four feature demos with printed proofs. |
| `data_infra/iceberg/warehouse/` | SQLite catalog (`catalog.db`) + Iceberg table data/metadata (~1 MB). |
| `data_infra/iceberg/build_summary.json` | Machine-readable summary of the last build (snapshot ids, row counts, schema, file counts). |

Run order: `smoke_iceberg.py` → `build_iceberg_catalog.py`
(both with `/usr/bin/python3`; requires `sqlalchemy`, installed for the system
interpreter alongside the pre-existing `pyiceberg 0.11.1` / `pyarrow 24`).

## Catalog & table layout

- **Catalog**: `SqlCatalog(uri="sqlite:///.../warehouse/catalog.db",
  warehouse="file://.../warehouse")`
- **Namespace**: `nusc_vla`
- **Table `nusc_vla.keyframes`**, partitioned by **identity(`split`),
  identity(`scenario`)** (hidden partitioning):

  | column | type | source |
  |--------|------|--------|
  | `sample_token` | string (required) | `infos[i].token` |
  | `scene_token` | string | `infos[i].scene_token` |
  | `timestamp` | long | `infos[i].timestamp` (µs) |
  | `split` | string | `val` (train extends identically) |
  | `scenario` | string | reproduction of eval's `classify_scenario` on the ego future-waypoint trajectory |
  | `ego_speed` | double | `can_bus[13]` (m/s) |
  | `n_objects` | int | count of `- ` lines in `bbox_text` |
  | `has_hdmap` | bool | a BEV HD-map PNG exists for the token |
  | `tod` | string | **added later via schema evolution** |

- **Table `nusc_vla.training_runs`** (lineage):
  `ckpt_name`, `train_data_snapshot_id`, `eval_L2_avg`, `eval_json_path`.

### Why the row set is exactly 5119

`build_rows(..., full_future_only=True)` keeps only keyframes with all 6 future
waypoints valid — this is precisely the eval-scored set. The resulting
`scenario` histogram is **identical** to `scenario_counts` in
`docs/eval_results/B5ppp_3cam_autovla_res_eval_results.json`:

```
stationary 939 | straight 2674 | turning 541 | lane_change 346 | braking 314 | cruising 305   (sum 5119)
```

That exact match is what makes the lineage claim trustworthy: the Iceberg table
is demonstrably the same data the checkpoint was scored on, not an
approximation. (Without the filter the table holds 6019 rows including
900 scene-tail frames; the eval drops those, `n_scored = 5119`.)

## Feature proofs (printed by `build_iceberg_catalog.py`)

> Snapshot IDs are assigned by Iceberg per commit and therefore differ on every
> run; the **structure** of the proof (row-count deltas, schema diff, file-count
> drop, count match) is stable. Representative values from a build are shown.

### 1. Snapshots / time-travel
Two appends → two snapshots, then read the older one:

```
commit#1 (low-dynamics subset): snapshot_id=<S1>  rows=3918
commit#2 (+ dynamic subset)   : snapshot_id=<S2>  rows=5119
snapshot history: S1 (parent=None, APPEND) -> S2 (parent=S1, APPEND)
TIME-TRAVEL read @ snapshot#1 = 3918 rows  (vs current 5119) -> older snapshot sees fewer rows
```
`tbl.scan(snapshot_id=S1)` reproduces the historical 3918-row state without
touching current data. Proof: 3918 < 5119, asserted.

### 2. Schema evolution (no rewrite)
```
schema BEFORE: [..., n_objects, has_hdmap]
add_column("tod", string)
schema AFTER : [..., n_objects, has_hdmap, tod]
read @ pre-evolution snapshot <S2>: 'tod' absent
after backfill commit <S3>: current scan has 'tod' with 200 non-null values
```
The column is added as pure metadata (a new schema id + field id 9); the
pre-evolution snapshot's data files are **not rewritten** — reading `S2` shows
no `tod` column, while the current table does. This is Iceberg's
read-on-write-free schema evolution.

### 3. Hidden-partition pruning
```
unfiltered scan plans          6 data files (all partitions)
scenario='turning' scan plans  1 data file  -> other scenario partitions skipped
scenario='turning' rows returned = 541 (matches eval scenario_counts['turning'])
```
The query uses a **plain column predicate** (`EqualTo("scenario","turning")`)
— no partition column on a path, no directory globbing. Iceberg derives the
partition from the identity transform on the `scenario` column and prunes 5 of
6 data files. The 541 rows returned match the eval taxonomy exactly.

### 4. Lineage (data snapshot → ckpt → eval)
`nusc_vla.training_runs` row:
```
ckpt_name              = nusc_planning_b5ppp_3cam_qwen3vl_multimodal/final
train_data_snapshot_id = <S2>                     (the 5119-row keyframes snapshot)
eval_L2_avg            = 0.6261894834879974        (from the eval JSON)
eval_json_path         = docs/eval_results/B5ppp_3cam_autovla_res_eval_results.json
```
Cross-check printed at build time: the `scenario_counts` computed from the
keyframes rows at snapshot `S2` equal the `scenario_counts` inside the eval
JSON — `MATCH: True`. So the lineage row points at a snapshot that is provably
the evaluated dataset, closing the loop **data version → checkpoint → metric**.

## How this maps to a PB-scale vehicle-log catalog

- **`keyframes` → fleet keyframe/clip catalog.** Same schema scales to billions
  of rows across many `split`s, cities, sensor configs. Partitioning by a
  hidden transform (here `split`,`scenario`; in production add
  `bucket(scene_token)`, `day(timestamp)`, `region`) lets curation queries
  ("all `turning` clips at night with HD-map coverage") prune to a handful of
  files out of millions — the file-count drop shown above is the same mechanism
  at scale.
- **Snapshots / time-travel → reproducible training sets.** "Train run X used
  the data as of snapshot S2" is a single immutable id. Re-mining or
  re-labelling appends new snapshots; old runs still read their exact historical
  view, so an L2 number is always reproducible against a frozen dataset.
- **Schema evolution → additive metadata without re-ETL.** New derived columns
  (`tod`, weather, auto-label confidence, sensor-health flags) land as metadata
  with zero rewrite of petabytes of existing files; historical snapshots stay
  byte-identical.
- **`training_runs` lineage → audit + regression triage.** Joining
  `training_runs.train_data_snapshot_id` to the `keyframes` snapshot answers
  "which data version produced this metric, and what exactly was in it" —
  essential for debugging a regression ("did the scenario mix shift between
  snapshots?") and for compliance/audit of an autonomy stack.
