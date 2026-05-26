#!/usr/bin/env python3
"""Ray Data streaming DAG for nuScenes-planning -> veRL parquet/lance ingestion.

Re-architects the legacy 8-worker ``multiprocessing.Pool`` script
(``grpo_vla/build_parquet.py``) into a proper Ray Data lazy DAG:

    from_items(keep_indices)              # tiny driver-side index list
        -> map(decode_resize_cam)         # open + resize CAM_FRONT current frame
        -> map(attach_hdmap)              # load + resize HD-map BEV PNG
        -> map(serialize_bbox_ego)        # parse bbox text + ego speed -> struct
        -> map(build_traj)                # ego2global -> local future waypoints
        -> map_batches(encode_row)        # JPEG-encode imgs + assemble veRL row
        -> write_parquet / write_lance    # streaming sink

Why this shape:
  * The driver materializes only a list of integer keep-indices (cheap). All
    heavy IO/decode happens inside the streaming operators, so memory stays
    bounded by `override_num_blocks` * block-size, NOT N rows.
  * Each map worker lazily builds ONE `SampleIndex` (amortizes the ~0.7 s pkl
    load) cached in a module global keyed by paths — Ray reuses the worker
    process across the block stream, so the pkl loads once per worker not per row.
  * Back-pressure: Ray Data's streaming executor only schedules new blocks when
    downstream operators / the writer drain — bounded by the configured
    resource limits and block count. We also cap block size via
    `override_num_blocks` to keep per-block memory small.

Each conceptual stage (decode_resize_cam / attach_hdmap / serialize_bbox_ego /
build_traj) is a clean function. They share one `SampleIndex.build_row` because
the underlying repo logic is tightly coupled (history walk feeds both camera
load and waypoints); to keep the DAG *expressive* without re-walking the scene
chain four times, the stage functions are thin wrappers that progressively
populate the row dict and the final map_batches does the JPEG encode + assembly.

Usage:
    /usr/bin/python3 ray_ingest.py --n 16   --out out_smoke.lance   # smoke
    /usr/bin/python3 ray_ingest.py --n 2000 --out out.lance         # default
    /usr/bin/python3 ray_ingest.py --n 2000 --out out.parquet --format parquet
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from typing import Any, Dict, List

import ray

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from nusc_common import SampleIndex, default_paths  # noqa: E402


# ---------------------------------------------------------------------------
# Per-worker SampleIndex cache. Ray reuses the worker process across blocks, so
# the ~0.7 s pkl load + keep-list filtering happens once per worker, not per row.
# ---------------------------------------------------------------------------
_INDEX_CACHE: Dict[str, SampleIndex] = {}


def _get_index(cfg: Dict[str, Any]) -> SampleIndex:
    key = cfg["infos_path"] + "|" + cfg["split"]
    idx = _INDEX_CACHE.get(key)
    if idx is None:
        idx = SampleIndex(
            infos_path=cfg["infos_path"],
            nusc_root=cfg["nusc_root"],
            hdmap_dir=cfg["hdmap_dir"],
            bbox_jsonl=cfg["bbox_jsonl"],
            split=cfg["split"],
            planning_cams=cfg.get("planning_cams"),
        )
        _INDEX_CACHE[key] = idx
    return idx


# ---------------------------------------------------------------------------
# DAG stage functions. Each receives + returns a single row dict (Ray Data row).
# To stay faithful to the repo's coupled scene-chain logic while still exposing
# the DAG as distinct stages, the stages progressively attach data, and the
# final encode stage assembles the veRL row.
# ---------------------------------------------------------------------------
def make_decode_resize_cam(cfg):
    """Stage 1: resolve CAM_FRONT current-frame path, open + RGB-convert.
    Carries the PIL image forward (in-memory object block)."""
    def _fn(row: Dict[str, Any]) -> Dict[str, Any]:
        from PIL import Image  # local import: safe inside worker
        idx = _get_index(cfg)
        keep_i = int(row["keep_i"])
        base_idx = idx.keep[keep_i]
        hist = idx._walk_history(base_idx)
        cam0 = idx.planning_cams[0]
        cur = Image.open(idx._image_path(hist[-1], cam0)).convert("RGB")
        row["sample_token"] = idx.infos[base_idx]["token"]
        row["base_idx"] = base_idx
        row["cam_img"] = cur
        return row
    return _fn


def make_attach_hdmap(cfg):
    """Stage 2: load + RGB-convert the HD-map BEV PNG (black fallback)."""
    def _fn(row: Dict[str, Any]) -> Dict[str, Any]:
        idx = _get_index(cfg)
        row["hdmap_img"] = idx._load_hdmap(row["sample_token"])
        return row
    return _fn


def make_serialize_bbox_ego(cfg):
    """Stage 3: bbox text lookup -> structured dicts + ego speed scalar."""
    def _fn(row: Dict[str, Any]) -> Dict[str, Any]:
        from nusc_common import parse_bbox_text, _ego_speed_mps, BBOX_NONE_TEXT
        idx = _get_index(cfg)
        info = idx.infos[row["base_idx"]]
        bbox_text = idx._lookup_bbox(row["sample_token"])
        if not bbox_text.strip():
            bbox_text = BBOX_NONE_TEXT
        row["bbox_text"] = bbox_text
        row["bbox_3d_list"] = parse_bbox_text(bbox_text)
        row["ego_speed_mps"] = _ego_speed_mps(info)
        return row
    return _fn


def make_build_traj(cfg):
    """Stage 4: ego2global geometry -> next-6 local waypoints + valid mask."""
    def _fn(row: Dict[str, Any]) -> Dict[str, Any]:
        idx = _get_index(cfg)
        wp, mask = idx._compute_waypoints(row["base_idx"])
        row["gt_waypoints"] = wp.astype("float32").tolist()
        row["valid_mask"] = mask.astype("float32").tolist()
        return row
    return _fn


def make_encode_row(cfg, max_edge: int, jpeg_q: int):
    """Stage 5 (batched): JPEG-encode the two images + assemble the veRL row.

    Batched (map_batches) so JPEG encode amortizes Python overhead and the
    block hands a compact serialized table to the writer. Returns the FINAL
    persisted schema (drops the in-flight PIL objects).
    """
    import json as _json
    import numpy as _np
    from nusc_common import jpeg_bytes, PLANNING_PROMPT, VIDEO_FPS, NUM_FUTURE_WP

    def _default(o):
        if isinstance(o, _np.ndarray):
            return o.tolist()
        if isinstance(o, (_np.floating,)):
            return float(o)
        if isinstance(o, (_np.integer,)):
            return int(o)
        raise TypeError(f"not serializable: {type(o)}")

    def _fn(batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        n = len(batch["keep_i"])
        prompts, images, extra_infos, reward_models, data_sources, tokens = (
            [], [], [], [], [], []
        )
        for k in range(n):
            cam_b = jpeg_bytes(batch["cam_img"][k], max_edge, jpeg_q)
            hd_b = jpeg_bytes(batch["hdmap_img"][k], max_edge, jpeg_q)
            bbox_text = batch["bbox_text"][k]
            ego = float(batch["ego_speed_mps"][k])
            gt_raw = batch["gt_waypoints"][k]
            gt = gt_raw.tolist() if hasattr(gt_raw, "tolist") else gt_raw
            vm_raw = batch["valid_mask"][k]
            vm = vm_raw.tolist() if hasattr(vm_raw, "tolist") else vm_raw
            bbl_raw = batch["bbox_3d_list"][k]
            bbl = list(bbl_raw) if hasattr(bbl_raw, "__iter__") and not isinstance(bbl_raw, list) else bbl_raw
            content = (
                "Camera FRONT (current): <image>\n"
                "HD-map BEV: <image>"
                f"\n\nDetected objects in ego frame:\n{bbox_text}\n"
                f"\nEgo speed at current frame: {ego:.2f} m/s\n"
                f"\n{PLANNING_PROMPT}"
            )
            extra = {
                "sample_token": batch["sample_token"][k],
                "gt_waypoints": gt,
                "valid_mask": vm,
                "ego_state": {"speed_mps": ego},
                "bbox_3d_list": bbl,
                "bbox_text": bbox_text,
                "horizon_s": float(NUM_FUTURE_WP) / float(VIDEO_FPS),
            }
            # Serialize struct fields to JSON strings for a stable, columnar
            # parquet/lance schema (mirrors veRL's extra_info json handling and
            # avoids pyarrow struct-inference churn across heterogeneous rows).
            prompts.append(_json.dumps([{"role": "user", "content": content}]))
            images.append([cam_b, hd_b])
            extra_infos.append(_json.dumps(extra, default=_default))
            reward_models.append(_json.dumps({"style": "rule", "ground_truth": gt}, default=_default))
            data_sources.append("nusc_planning")
            tokens.append(batch["sample_token"][k])
        return {
            "prompt": prompts,
            "images": images,
            "extra_info": extra_infos,
            "reward_model": reward_models,
            "data_source": data_sources,
            "sample_token": tokens,
        }
    return _fn


# ---------------------------------------------------------------------------
# Pipeline assembly
# ---------------------------------------------------------------------------
def build_dataset(cfg, n, seed, blocks, max_edge, jpeg_q, concurrency):
    """Construct (lazily) the full Ray Data DAG and return the Dataset handle."""
    # Driver builds keep-list once to pick the SAME subset the legacy path uses
    # (random sample, fixed seed, sorted) so bench A vs B is apples-to-apples.
    # NOTE: build a LOCAL index here (do NOT populate the module-global
    # _INDEX_CACHE) — otherwise cloudpickle would capture the ~200MB infos dict
    # into every map UDF closure. Workers build their own via _get_index.
    idx = SampleIndex(
        infos_path=cfg["infos_path"], nusc_root=cfg["nusc_root"],
        hdmap_dir=cfg["hdmap_dir"], bbox_jsonl=cfg["bbox_jsonl"],
        split=cfg["split"], planning_cams=cfg.get("planning_cams"),
    )
    n_full = len(idx)
    rng = random.Random(seed)
    if n >= n_full:
        keep_is = list(range(n_full))
    else:
        keep_is = rng.sample(range(n_full), n)
        keep_is.sort()
    print(f"[ray_ingest] split={cfg['split']} keep_full={n_full} -> {len(keep_is)} rows "
          f"(seed={seed})")

    items = [{"keep_i": i} for i in keep_is]
    ds = ray.data.from_items(items, override_num_blocks=blocks)

    # DAG: explicit stages, each with bounded concurrency (back-pressure).
    ds = ds.map(make_decode_resize_cam(cfg), concurrency=concurrency)
    ds = ds.map(make_attach_hdmap(cfg), concurrency=concurrency)
    ds = ds.map(make_serialize_bbox_ego(cfg), concurrency=concurrency)
    ds = ds.map(make_build_traj(cfg), concurrency=concurrency)
    ds = ds.map_batches(
        make_encode_row(cfg, max_edge, jpeg_q),
        batch_size=64,
        concurrency=concurrency,
    )
    return ds, len(keep_is)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--n", type=int, default=2000, help="subset size")
    ap.add_argument("--out", default="out.lance",
                    help="output path (relative to data_infra/ray/)")
    ap.add_argument("--format", default="auto", choices=["auto", "lance", "parquet"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--blocks", type=int, default=16,
                    help="override_num_blocks: bounds per-block memory (back-pressure)")
    ap.add_argument("--concurrency", type=int, default=8,
                    help="max parallel actors per map stage")
    ap.add_argument("--num-cpus", type=int, default=8)
    ap.add_argument("--max-edge", type=int, default=448)
    ap.add_argument("--jpeg-q", type=int, default=75)
    args = ap.parse_args()

    out_path = args.out if os.path.isabs(args.out) else os.path.join(_HERE, args.out)
    fmt = args.format
    if fmt == "auto":
        fmt = "lance" if out_path.endswith(".lance") else "parquet"

    cfg = default_paths(args.split)
    cfg["split"] = args.split

    # Local mode: CPU-only, no GPU/cluster contention (IO/decode bound).
    ray.init(num_cpus=args.num_cpus, ignore_reinit_error=True,
             include_dashboard=False, log_to_driver=False)
    try:
        # Bound the streaming executor's object-store memory for clear back-pressure.
        from ray.data import ExecutionResources
        ctx = ray.data.DataContext.get_current()
        ctx.execution_options.resource_limits = ExecutionResources(
            object_store_memory=1 * 1024**3  # 1 GiB
        )

        t0 = time.time()
        ds, n_rows = build_dataset(
            cfg, args.n, args.seed, args.blocks, args.max_edge, args.jpeg_q,
            args.concurrency,
        )

        # Clean any stale output.
        import shutil
        if os.path.exists(out_path):
            if os.path.isdir(out_path):
                shutil.rmtree(out_path)
            else:
                os.remove(out_path)

        if fmt == "lance":
            ds.write_lance(out_path)
        else:
            ds.write_parquet(out_path)

        elapsed = time.time() - t0
        rate = n_rows / max(elapsed, 1e-9)
        # Size on disk.
        if os.path.isdir(out_path):
            size = sum(
                os.path.getsize(os.path.join(dp, f))
                for dp, _, fs in os.walk(out_path) for f in fs
            )
        else:
            size = os.path.getsize(out_path) if os.path.exists(out_path) else 0
        print(f"[ray_ingest] DONE {out_path} ({fmt}) "
              f"rows={n_rows} {size/1e6:.1f} MB "
              f"{elapsed:.1f}s = {rate:.2f} rows/s")
        print(f"RAY_BENCH rows={n_rows} elapsed_s={elapsed:.3f} rows_per_s={rate:.3f}")
    finally:
        ray.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
