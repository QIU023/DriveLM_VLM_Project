"""Tier-2 producer: cache frozen-ViT + FasterVLM x4 vision tokens for the
3-cam NATIVE planning SFT, so training can SKIP the ViT every step.

For a subset of the 3-cam NATIVE train split (config
``configs/nuscenes_planning_3cam_qwen3vl_NATIVE.yaml``, native ~2800 tok/cam),
each row stores everything the cached forward needs to reproduce the LIVE
compressed forward's loss WITHOUT running the vision tower:

  sample_token               (str)
  input_ids / labels / attention_mask / mm_token_type_ids   (already TRIMMED:
        each cam's 2800 video-pad run -> 700 via FasterVLM x4; image-pad kept
        121; text kept) -- IDENTICAL trim to forward_with_video_compression_free
  video_grid_thw / image_grid_thw   (rebuilt: video -> 3x (1,h*2,w*2) with
        h*w=700; image kept (1,22,22))
  video_pooler_int8 (2100x2560) + scale     compressed video tokens (FasterVLM)
  video_deepstack_int8 (L*2100x2560) + scale + L
  image_pooler_int8 (121x2560) + scale       native HD-map image tokens
  image_deepstack_int8 (L*121x2560) + scale + L

The trim + FasterVLM selection are imported from ``native_cache_common`` (the
SAME module the live + cached forwards use), so the cache is layout-identical by
construction -> the HARD PARITY GATE only sees the int8 round-trip error.

8-GPU sharded: one process per visible GPU, rows split by ``idx % world``. Each
shard writes a Lance fragment; rank 0 concatenates them into the final dataset.

Usage:
    export HF_HOME=/workspace/.hf_home; unset HF_HUB_OFFLINE
    # smoke:
    /usr/bin/python3 data_infra/lance/cache_3cam_native.py --n 8 --gpus 1 --verify \
        --out data_infra/lance/nusc_3cam_native_smoke.lance
    # build:
    /usr/bin/python3 data_infra/lance/cache_3cam_native.py --n 450 \
        --out data_infra/lance/nusc_3cam_native.lance
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

import numpy as np
import pyarrow as pa
import lance

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
_SCRIPTS = os.path.join(_REPO, "scripts")
for _p in (_SCRIPTS, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CONFIG = os.path.join(_REPO, "configs", "nuscenes_planning_3cam_qwen3vl_NATIVE.yaml")
DEFAULT_CKPT = None  # default = config's model_id (base Qwen3-VL-4B)

from native_cache_common import (  # noqa: E402
    COMPRESS_RATIO,
    MERGE_SIZE,
    compress_video_features,
    quantize_int8,
    dequantize_int8,
    trim_native_layout,
)


def _dir_bytes(path: str) -> int:
    total = 0
    for root, _d, files in os.walk(path):
        for fn in files:
            total += os.path.getsize(os.path.join(root, fn))
    return total


# ----------------------------------------------------------------------------
# Worker: process one shard of the chosen subset
# ----------------------------------------------------------------------------
def run_shard(out_dir: str, ckpt: str, n: int, rank: int, world: int,
              tmp_dir: str, verify: bool) -> None:
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from train_lora import load_config
    from multimodal_planning_dataset import build_multimodal_planning_dataset

    cfg = load_config(CONFIG)
    model_id = ckpt or cfg["model_id"]
    device = "cuda"

    proc = AutoProcessor.from_pretrained(model_id)
    if hasattr(proc, "image_processor") and proc.image_processor is not None:
        proc.image_processor.min_pixels = cfg["min_pixels"]
        proc.image_processor.max_pixels = cfg["max_pixels"]
    # NATIVE: no video_*_pixels in cfg -> leave video processor at default (native).

    ds = build_multimodal_planning_dataset(cfg, proc, split="train")
    n = min(n, len(ds))
    my_rows = list(range(rank, n, world))
    print(f"[shard {rank}/{world}] {len(my_rows)} of first {n} rows", flush=True)

    video_token_id = proc.tokenizer.convert_tokens_to_ids("<|video_pad|>")
    image_token_id = proc.tokenizer.convert_tokens_to_ids("<|image_pad|>")

    model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16).to(device).eval()
    inner = model.model

    cols = {k: [] for k in (
        "sample_token", "input_ids", "labels", "attention_mask", "mm_token_type_ids",
        "video_grid_thw", "image_grid_thw",
        "video_pooler_int8", "video_pooler_scale", "video_pooler_shape",
        "video_deepstack_int8", "video_deepstack_scale", "video_deepstack_shape",
        "image_pooler_int8", "image_pooler_scale", "image_pooler_shape",
        "image_deepstack_int8", "image_deepstack_scale", "image_deepstack_shape",
        "n_video_pad", "n_image_pad",
    )}
    max_rel = 0.0

    for ridx in my_rows:
        item = ds[ridx]
        ids = item["input_ids"].to(device)
        attn = item["attention_mask"].to(device)
        labels = item["labels"].to(device)
        mm = item.get("mm_token_type_ids")
        mm = mm.to(device) if mm is not None else None
        vgrid = item["video_grid_thw"].to(device)
        if vgrid.dim() == 1:
            vgrid = vgrid.unsqueeze(0)
        igrid = item.get("image_grid_thw")
        if igrid is not None:
            igrid = igrid.to(device)
            if igrid.dim() == 1:
                igrid = igrid.unsqueeze(0)
        pv = item["pixel_values_videos"].to(device, torch.bfloat16)
        pvi = item["pixel_values"].to(device, torch.bfloat16)

        per_orig = [int(vgrid[i, 0]) * (int(vgrid[i, 1]) // MERGE_SIZE) * (int(vgrid[i, 2]) // MERGE_SIZE)
                    for i in range(vgrid.shape[0])]
        per_comp = [max(1, c // COMPRESS_RATIO) for c in per_orig]

        # 1. Trim layout (SAME helper as live forward).
        trimmed = trim_native_layout(ids, attn, labels, mm, vgrid, video_token_id,
                                     compress_ratio=COMPRESS_RATIO, merge_size=MERGE_SIZE)

        # 2. Run frozen ViT (no_grad) on video + image; compress video w/ FasterVLM.
        with torch.no_grad():
            vo = inner.get_video_features(pv, vgrid)
            io = inner.get_image_features(pvi, igrid)
        v_pooler = vo.pooler_output
        if not isinstance(v_pooler, (list, tuple)):
            v_pooler = list(__import__("torch").split(v_pooler, per_orig))
        v_deep = list(getattr(vo, "deepstack_features", []) or [])
        comp_pooler, comp_deep = compress_video_features(
            list(v_pooler), v_deep, per_orig, per_comp, COMPRESS_RATIO)
        # image: native (no compression)
        i_pooler = io.pooler_output
        i_pooler = (i_pooler[0] if isinstance(i_pooler, (list, tuple)) else i_pooler)
        i_deep = list(getattr(io, "deepstack_features", []) or [])

        # 3. int8 quantize (per-tensor) each feature group.
        vp_np = comp_pooler.float().cpu().numpy()                                  # (2100, D)
        vd_np = np.stack([d.float().cpu().numpy() for d in comp_deep], axis=0)      # (L, 2100, D)
        ip_np = i_pooler.float().cpu().numpy()                                     # (121, D)
        id_np = np.stack([d.float().cpu().numpy() for d in i_deep], axis=0)        # (L, 121, D)

        vp_b, vp_s, vp_sh = quantize_int8(vp_np)
        vd_b, vd_s, vd_sh = quantize_int8(vd_np)
        ip_b, ip_s, ip_sh = quantize_int8(ip_np)
        id_b, id_s, id_sh = quantize_int8(id_np)

        if verify:
            for raw, b, s, sh in ((vp_np, vp_b, vp_s, vp_sh), (vd_np, vd_b, vd_s, vd_sh),
                                  (ip_np, ip_b, ip_s, ip_sh), (id_np, id_b, id_s, id_sh)):
                deq = dequantize_int8(b, s, sh)
                rel = float(np.abs(deq - raw).mean() / (np.abs(raw).mean() + 1e-8))
                max_rel = max(max_rel, rel)

        cols["sample_token"].append(item["_meta_token"])
        cols["input_ids"].append(trimmed["input_ids"].cpu().numpy().astype(np.int64).tolist())
        cols["labels"].append(trimmed["labels"].cpu().numpy().astype(np.int64).tolist())
        cols["attention_mask"].append(trimmed["attention_mask"].cpu().numpy().astype(np.int64).tolist())
        cols["mm_token_type_ids"].append(
            (trimmed["mm_token_type_ids"].cpu().numpy().astype(np.int64).tolist())
            if trimmed["mm_token_type_ids"] is not None else [])
        cols["video_grid_thw"].append(trimmed["video_grid_thw"].cpu().numpy().astype(np.int64).reshape(-1).tolist())
        cols["image_grid_thw"].append(
            (igrid.cpu().numpy().astype(np.int64).reshape(-1).tolist()) if igrid is not None else [])
        cols["video_pooler_int8"].append(vp_b)
        cols["video_pooler_scale"].append(vp_s)
        cols["video_pooler_shape"].append([int(x) for x in vp_sh])
        cols["video_deepstack_int8"].append(vd_b)
        cols["video_deepstack_scale"].append(vd_s)
        cols["video_deepstack_shape"].append([int(x) for x in vd_sh])
        cols["image_pooler_int8"].append(ip_b)
        cols["image_pooler_scale"].append(ip_s)
        cols["image_pooler_shape"].append([int(x) for x in ip_sh])
        cols["image_deepstack_int8"].append(id_b)
        cols["image_deepstack_scale"].append(id_s)
        cols["image_deepstack_shape"].append([int(x) for x in id_sh])
        cols["n_video_pad"].append(int((trimmed["input_ids"] == video_token_id).sum()))
        cols["n_image_pad"].append(int((trimmed["input_ids"] == image_token_id).sum()))

    if verify:
        print(f"[shard {rank}] max dequant rel-err = {max_rel*100:.3f}%", flush=True)

    table = pa.table({
        "sample_token": pa.array(cols["sample_token"], pa.string()),
        "input_ids": pa.array(cols["input_ids"], pa.list_(pa.int64())),
        "labels": pa.array(cols["labels"], pa.list_(pa.int64())),
        "attention_mask": pa.array(cols["attention_mask"], pa.list_(pa.int64())),
        "mm_token_type_ids": pa.array(cols["mm_token_type_ids"], pa.list_(pa.int64())),
        "video_grid_thw": pa.array(cols["video_grid_thw"], pa.list_(pa.int64())),
        "image_grid_thw": pa.array(cols["image_grid_thw"], pa.list_(pa.int64())),
        "video_pooler_int8": pa.array(cols["video_pooler_int8"], pa.binary()),
        "video_pooler_scale": pa.array(cols["video_pooler_scale"], pa.float32()),
        "video_pooler_shape": pa.array(cols["video_pooler_shape"], pa.list_(pa.int64())),
        "video_deepstack_int8": pa.array(cols["video_deepstack_int8"], pa.binary()),
        "video_deepstack_scale": pa.array(cols["video_deepstack_scale"], pa.float32()),
        "video_deepstack_shape": pa.array(cols["video_deepstack_shape"], pa.list_(pa.int64())),
        "image_pooler_int8": pa.array(cols["image_pooler_int8"], pa.binary()),
        "image_pooler_scale": pa.array(cols["image_pooler_scale"], pa.float32()),
        "image_pooler_shape": pa.array(cols["image_pooler_shape"], pa.list_(pa.int64())),
        "image_deepstack_int8": pa.array(cols["image_deepstack_int8"], pa.binary()),
        "image_deepstack_scale": pa.array(cols["image_deepstack_scale"], pa.float32()),
        "image_deepstack_shape": pa.array(cols["image_deepstack_shape"], pa.list_(pa.int64())),
        "n_video_pad": pa.array(cols["n_video_pad"], pa.int64()),
        "n_image_pad": pa.array(cols["n_image_pad"], pa.int64()),
    })
    shard_out = os.path.join(tmp_dir, f"shard_{rank}.lance")
    if os.path.exists(shard_out):
        shutil.rmtree(shard_out)
    lance.write_dataset(table, shard_out)
    print(f"[shard {rank}] wrote {table.num_rows} rows -> {shard_out}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(_HERE, "nusc_3cam_native.lance"))
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--n", type=int, default=450, help="number of train samples to cache")
    ap.add_argument("--gpus", type=int, default=0, help="0 = all visible")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--_rank", type=int, default=-1)
    ap.add_argument("--_world", type=int, default=-1)
    ap.add_argument("--_tmp", default="")
    args = ap.parse_args()

    if args._rank >= 0:
        run_shard(args.out, args.ckpt, args.n, args._rank, args._world, args._tmp, args.verify)
        return 0

    import torch
    n_gpus = args.gpus if args.gpus > 0 else torch.cuda.device_count()
    n_gpus = max(1, min(n_gpus, args.n))
    print(f"[cache] caching {args.n} rows across {n_gpus} GPU(s) -> {args.out}")

    tmp_dir = args.out + ".shards_tmp"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)

    procs = []
    for rank in range(n_gpus):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(rank)
        env["HF_HOME"] = env.get("HF_HOME", "/workspace/.hf_home")
        env.pop("HF_HUB_OFFLINE", None)
        cmd = [sys.executable, os.path.abspath(__file__),
               "--out", args.out, "--n", str(args.n),
               "--_rank", str(rank), "--_world", str(n_gpus), "--_tmp", tmp_dir]
        if args.ckpt:
            cmd += ["--ckpt", args.ckpt]
        if args.verify:
            cmd.append("--verify")
        procs.append(subprocess.Popen(cmd, env=env))
    rc = 0
    for p in procs:
        rc |= p.wait()
    if rc != 0:
        print(f"[cache] a shard worker failed (rc={rc}); aborting before merge")
        return rc

    parts = [lance.dataset(os.path.join(tmp_dir, f"shard_{r}.lance")).to_table()
             for r in range(n_gpus)]
    merged = pa.concat_tables(parts)
    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    lance.write_dataset(merged, args.out)
    shutil.rmtree(tmp_dir)

    ds = lance.dataset(args.out)
    on_disk = _dir_bytes(args.out)
    r0 = ds.take([0]).to_pylist()[0]
    print("\n[verify] ---------------------------------------------------------")
    print(f"[verify] rows = {ds.count_rows()}  on-disk = {on_disk/1e9:.3f} GB")
    print(f"[verify] n_video_pad = {r0['n_video_pad']}  n_image_pad = {r0['n_image_pad']}")
    print(f"[verify] video_pooler_shape = {r0['video_pooler_shape']}  "
          f"video_deepstack_shape = {r0['video_deepstack_shape']}")
    print(f"[verify] image_pooler_shape = {r0['image_pooler_shape']}  "
          f"image_deepstack_shape = {r0['image_deepstack_shape']}")
    print(f"[verify] video_grid_thw = {r0['video_grid_thw']}  image_grid_thw = {r0['image_grid_thw']}")
    per_row_mb = on_disk / max(1, ds.count_rows()) / 1e6
    print(f"[verify] per-row ~ {per_row_mb:.2f} MB")
    print("[verify] ---------------------------------------------------------")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
