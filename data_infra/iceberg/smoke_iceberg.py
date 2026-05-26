"""SMOKE: SqlCatalog + 10-row keyframes table + one snapshot round-trip.

Run BEFORE the full build. Creates a throwaway catalog under
warehouse/_smoke so it never touches the real catalog.db.
"""
import os
import shutil
import sys

import pyarrow as pa
from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.schema import Schema
from pyiceberg.types import (BooleanType, DoubleType, IntegerType, LongType,
                             NestedField, StringType)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nusc_extract import build_rows

HERE = os.path.dirname(os.path.abspath(__file__))
WAREHOUSE = os.path.join(HERE, "warehouse", "_smoke")

# fresh
if os.path.isdir(WAREHOUSE):
    shutil.rmtree(WAREHOUSE)
os.makedirs(WAREHOUSE, exist_ok=True)

db_path = os.path.join(WAREHOUSE, "catalog.db")
catalog = SqlCatalog(
    "smoke",
    uri=f"sqlite:///{db_path}",
    warehouse=f"file://{WAREHOUSE}",
)
print("[smoke] catalog created:", db_path)

catalog.create_namespace_if_not_exists("nusc_vla")

schema = Schema(
    NestedField(1, "sample_token", StringType(), required=True),
    NestedField(2, "scene_token", StringType(), required=False),
    NestedField(3, "timestamp", LongType(), required=False),
    NestedField(4, "split", StringType(), required=False),
    NestedField(5, "scenario", StringType(), required=False),
    NestedField(6, "ego_speed", DoubleType(), required=False),
    NestedField(7, "n_objects", IntegerType(), required=False),
    NestedField(8, "has_hdmap", BooleanType(), required=False),
)

tbl = catalog.create_table("nusc_vla.keyframes_smoke", schema=schema)
print("[smoke] table created")

rows = build_rows(
    os.path.join(os.path.dirname(HERE), "..", "data/uniad_infos/nuscenes_infos_temporal_val.pkl"),
    os.path.join(os.path.dirname(HERE), "..", "data/preproc/bbox_egostate_val.jsonl"),
    "val",
    hdmap_dir=os.path.join(os.path.dirname(HERE), "..", "data/preproc/hdmap_bev/val"),
    limit=10,
)
print("[smoke] built", len(rows), "rows")

arrow_schema = pa.schema([
    pa.field("sample_token", pa.string(), nullable=False),
    ("scene_token", pa.string()),
    ("timestamp", pa.int64()),
    ("split", pa.string()),
    ("scenario", pa.string()),
    ("ego_speed", pa.float64()),
    ("n_objects", pa.int32()),
    ("has_hdmap", pa.bool_()),
])


def to_table(rs):
    cols = {k: [r[k] for r in rs] for k in arrow_schema.names}
    return pa.Table.from_pydict(cols, schema=arrow_schema)


# commit 1: first 6 rows
tbl.append(to_table(rows[:6]))
snap1 = tbl.metadata.current_snapshot_id
n1 = len(tbl.scan().to_arrow())
print(f"[smoke] commit1 snapshot={snap1} rows={n1}")

# commit 2: remaining 4
tbl.append(to_table(rows[6:]))
snap2 = tbl.metadata.current_snapshot_id
n2 = len(tbl.scan().to_arrow())
print(f"[smoke] commit2 snapshot={snap2} rows={n2}")

# time-travel back to snap1
n1_tt = len(tbl.scan(snapshot_id=snap1).to_arrow())
print(f"[smoke] time-travel to snap1 rows={n1_tt}")

assert n1 == 6 and n2 == 10 and n1_tt == 6, "snapshot round-trip FAILED"
print("[smoke] OK: snapshot round-trip verified (6 -> 10, time-travel back to 6)")

# cleanup smoke warehouse
shutil.rmtree(WAREHOUSE)
print("[smoke] cleaned up", WAREHOUSE)
