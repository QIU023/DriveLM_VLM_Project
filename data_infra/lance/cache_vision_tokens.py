"""Cache compressed + int8-quantized vision tokens into the Lance dataset (P0).

For every row already in ``nusc_mm.lance`` this runs the Qwen3-VL vision tower
(``model.model.visual`` via ``model.model.get_video_features``) over the sample's
4-frame CAM_FRONT clip, takes the ViT pooler output (one block of
(n_tokens, 2560) for 1-cam native, n_tokens=2800 for grid [2,56,100]), applies
FasterVLM x4 (reuse ``scripts/visual_compress.compress_visual_tokens(method=
"fastervlm")`` — top-K by L2 norm, the documented training-free CLS-attention
proxy used everywhere else in this repo), then quantizes the kept tokens to int8
with a single per-tensor symmetric scale.

It adds three columns to the Lance dataset (via lance merge by ``sample_token``):
    vis_tokens_int8   (binary)  row-major int8 bytes, shape = vis_tokens_shape
    vis_tokens_scale  (float)   per-tensor scale s; dequant = int8.astype(f32)*s
    vis_tokens_shape  (list<int64>)  [n_tokens_kept, 2560]

Multi-GPU: one process per visible GPU, sharded by ``row_idx % world_size``.
Each shard writes a small intermediate Lance file under a tmp dir; rank 0 then
concatenates all shards and merges the 3 columns into the main dataset keyed on
sample_token.

Quantization (symmetric per-tensor int8):
    s = max(|x|) / 127
    q = round(x / s).clip(-127, 127).astype(int8)
    x_hat = q.astype(f32) * s
Round-trip error on bf16 vision tokens is typically < ~1-2% relative (verified
in the smoke run); we assert it in --verify mode.

Usage:
    export HF_HOME=/workspace/.hf_home; unset HF_HUB_OFFLINE
    # full subset (spawns one proc/GPU internally):
    /usr/bin/python3 data_infra/lance/cache_vision_tokens.py --lance .../nusc_mm.lance
    # smoke:
    /usr/bin/python3 data_infra/lance/cache_vision_tokens.py --lance .../nusc_mm_smoke.lance --gpus 1 --verify
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

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SCRIPTS = os.path.join(_REPO, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

DEFAULT_CKPT = os.path.join(
    _REPO, "checkpoints_qwen25/nusc_planning_b5pp_1cam_qwen3vl_multimodal/final"
)
COMPRESS_METHOD = "fastervlm"   # top-K by L2 norm (FasterVLM CLS-attention proxy)
COMPRESS_RATIO = 4              # x4 -> keep 1/4 tokens (2800 -> 700)
HIDDEN_DIM = 2560


# ----------------------------------------------------------------------------
# Quantization helpers
# ----------------------------------------------------------------------------
def quantize_int8(x: np.ndarray) -> tuple[bytes, float, list[int]]:
    """Symmetric per-tensor int8 quant. x: float32 (N, D). Returns (bytes, scale, shape)."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    amax = float(np.abs(x).max())
    scale = amax / 127.0 if amax > 0 else 1.0
    q = np.round(x / scale).clip(-127, 127).astype(np.int8)
    return q.tobytes(), scale, list(x.shape)


def dequantize_int8(buf: bytes, scale: float, shape: list[int]) -> np.ndarray:
    q = np.frombuffer(buf, dtype=np.int8).reshape(shape).astype(np.float32)
    return q * scale


# ----------------------------------------------------------------------------
# Worker: process one shard of rows
# ----------------------------------------------------------------------------
def run_shard(lance_path: str, ckpt: str, rank: int, world: int, tmp_dir: str,
              verify: bool) -> None:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from transformers.video_utils import VideoMetadata
    from PIL import Image
    from visual_compress import compress_visual_tokens

    device = "cuda"
    ds = lance.dataset(lance_path)
    n_rows = ds.count_rows()
    my_rows = list(range(rank, n_rows, world))
    print(f"[shard {rank}/{world}] {len(my_rows)} rows", flush=True)

    model = AutoModelForImageTextToText.from_pretrained(ckpt, dtype=torch.bfloat16).to(device).eval()
    proc = AutoProcessor.from_pretrained(ckpt)
    inner = model.model

    out_tokens, out_scale, out_shape, out_token = [], [], [], []
    max_rel_err = 0.0

    for ridx in my_rows:
        row = ds.take([ridx], columns=["sample_token", "camera_paths"]).to_pylist()[0]
        frames = [Image.open(p).convert("RGB") for p in row["camera_paths"]]
        meta = [VideoMetadata(total_num_frames=len(frames), fps=2.0,
                              frames_indices=list(range(len(frames))),
                              height=frames[0].height, width=frames[0].width)]
        inputs = proc(text=["<|vision_start|><|video_pad|><|vision_end|>"],
                      videos=[frames], video_metadata=meta, return_tensors="pt")
        pv = inputs["pixel_values_videos"].to(device, torch.bfloat16)
        grid = inputs["video_grid_thw"].to(device)

        with torch.no_grad():
            pooler = inner.get_video_features(pv, grid).pooler_output
            embeds = pooler[0] if isinstance(pooler, (list, tuple)) else pooler  # (N, 2560) bf16
            # FasterVLM x4. compress_visual_tokens expects grid_thw rows whose
            # product == n_tokens; pooler tokens are already post-merge so use
            # [1, 1, N] (a flat token list — _fastervlm only reads t*h*w == N).
            n_tok = embeds.shape[0]
            comp_grid = torch.tensor([[1, 1, n_tok]], device=device)
            comp, _ = compress_visual_tokens(embeds, comp_grid, COMPRESS_METHOD, COMPRESS_RATIO)

        comp_f32 = comp.float().cpu().numpy()  # (k, 2560)
        buf, scale, shape = quantize_int8(comp_f32)

        if verify:
            deq = dequantize_int8(buf, scale, shape)
            denom = np.abs(comp_f32).mean() + 1e-8
            rel = float(np.abs(deq - comp_f32).mean() / denom)
            max_rel_err = max(max_rel_err, rel)

        out_token.append(row["sample_token"])
        out_tokens.append(buf)
        out_scale.append(float(scale))
        out_shape.append([int(s) for s in shape])

    if verify:
        print(f"[shard {rank}] max dequant rel-err = {max_rel_err*100:.3f}%", flush=True)

    table = pa.table({
        "sample_token": pa.array(out_token, pa.string()),
        "vis_tokens_int8": pa.array(out_tokens, pa.binary()),
        "vis_tokens_scale": pa.array(out_scale, pa.float32()),
        "vis_tokens_shape": pa.array(out_shape, pa.list_(pa.int64())),
    })
    shard_out = os.path.join(tmp_dir, f"shard_{rank}.lance")
    if os.path.exists(shard_out):
        shutil.rmtree(shard_out)
    lance.write_dataset(table, shard_out)
    print(f"[shard {rank}] wrote {table.num_rows} rows -> {shard_out}", flush=True)


# ----------------------------------------------------------------------------
# Orchestrator: spawn one proc/GPU, then merge columns into the main dataset
# ----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lance", required=True, help="path to nusc_mm.lance to augment")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--gpus", type=int, default=0,
                    help="number of GPUs (default 0 = all visible)")
    ap.add_argument("--verify", action="store_true", help="check int8 round-trip rel-err")
    ap.add_argument("--_rank", type=int, default=-1, help="(internal) shard rank")
    ap.add_argument("--_world", type=int, default=-1, help="(internal) world size")
    ap.add_argument("--_tmp", default="", help="(internal) shard tmp dir")
    args = ap.parse_args()

    # Worker mode (re-entrant).
    if args._rank >= 0:
        run_shard(args.lance, args.ckpt, args._rank, args._world, args._tmp, args.verify)
        return 0

    # Orchestrator mode.
    import torch
    n_gpus = args.gpus if args.gpus > 0 else torch.cuda.device_count()
    n_rows = lance.dataset(args.lance).count_rows()
    n_gpus = min(n_gpus, n_rows)
    print(f"[cache] {n_rows} rows across {n_gpus} GPU(s); method={COMPRESS_METHOD} x{COMPRESS_RATIO}")

    tmp_dir = args.lance + ".vis_shards_tmp"
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
               "--lance", args.lance, "--ckpt", args.ckpt,
               "--_rank", str(rank), "--_world", str(n_gpus), "--_tmp", tmp_dir]
        if args.verify:
            cmd.append("--verify")
        procs.append(subprocess.Popen(cmd, env=env))
    rc = 0
    for p in procs:
        rc |= p.wait()
    if rc != 0:
        print(f"[cache] a shard worker failed (rc={rc}); aborting before merge")
        return rc

    # Concatenate all shard tables, then merge the 3 vis columns into main ds
    # keyed on sample_token (lance.merge does a left join on the join column).
    parts = []
    for rank in range(n_gpus):
        sp = os.path.join(tmp_dir, f"shard_{rank}.lance")
        parts.append(lance.dataset(sp).to_table())
    merged = pa.concat_tables(parts)
    print(f"[cache] merging {merged.num_rows} vis-token rows into {args.lance}")

    main_ds = lance.dataset(args.lance)
    # Drop pre-existing vis columns so re-runs are idempotent.
    existing = set(main_ds.schema.names)
    drop = [c for c in ("vis_tokens_int8", "vis_tokens_scale", "vis_tokens_shape") if c in existing]
    if drop:
        main_ds.drop_columns(drop)
        main_ds = lance.dataset(args.lance)
    main_ds.merge(merged, left_on="sample_token", right_on="sample_token")

    shutil.rmtree(tmp_dir)

    # Verify + size report.
    ds = lance.dataset(args.lance)
    on_disk = _dir_bytes(args.lance)
    row0 = ds.take([0], columns=["sample_token", "vis_tokens_int8", "vis_tokens_scale",
                                 "vis_tokens_shape"]).to_pylist()[0]
    shape = row0["vis_tokens_shape"]
    n_bytes_tok = len(row0["vis_tokens_int8"])
    print("\n[verify] ---------------------------------------------------------")
    print(f"[verify] rows = {ds.count_rows()}  on-disk = {on_disk/1e9:.3f} GB")
    print(f"[verify] vis_tokens_shape = {shape}  ({n_bytes_tok} bytes/row int8)")
    print(f"[verify] vis_tokens_scale[0] = {row0['vis_tokens_scale']:.6g}")
    per_row_mb = n_bytes_tok / 1e6
    print(f"[verify] per-row token cache = {per_row_mb:.2f} MB; "
          f"extrapolated full-corpus (28130 train) = {per_row_mb*28130/1e3:.1f} GB")
    print("[verify] ---------------------------------------------------------")
    return 0


def _dir_bytes(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for fn in files:
            total += os.path.getsize(os.path.join(root, fn))
    return total


if __name__ == "__main__":
    raise SystemExit(main())
