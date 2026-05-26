"""Throughput A/B: original ViT-in-the-loop dataset vs Lance cached-token path.

Path A (BASELINE): scripts/multimodal_planning_dataset.MultiModalPlanningDataset
  per sample -> JPEG decode + Qwen processor (chat template, video grid) + the
  ViT vision-tower forward (model.model.get_video_features) to produce the
  pooler tokens. This is the cost the train loop pays today on every step.

Path B (LANCE): data_infra/lance/lance_dataset.LanceMMDataset
  per sample -> Lance zero-copy take + int8 dequant. NO JPEG decode, NO
  processor, NO ViT forward. The cached tokens ARE the ViT output.

Both paths run over the SAME subset (the Lance file's sample_tokens), through a
real DataLoader (num_workers, pin_memory, prefetch). We measure steady-state
samples/sec, p50 batch latency, and a GPU-idle/stall estimate.

Stall estimate: for a given path the "GPU-bound" portion is the consumer work
done on-GPU per batch (Path A: ViT forward; Path B: a tiny dequant->GPU copy
that stands in for the consumer). We time (a) end-to-end per-batch latency
including the dataloader fetch and (b) the pure GPU-consume time. stall% =
1 - gpu_time / wall_time, i.e. the fraction of wall-clock the GPU spent waiting
on data. A high stall% on the baseline + low stall% on Lance is the headline:
caching the ViT output moves the bottleneck off the per-step critical path.

Usage:
    export HF_HOME=/workspace/.hf_home; unset HF_HUB_OFFLINE
    /usr/bin/python3 data_infra/lance/bench_throughput.py --lance .../nusc_mm.lance \
        --steps 200 --batch-size 4 --num-workers 8
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SCRIPTS = os.path.join(_REPO, "scripts")
_LANCEDIR = os.path.dirname(os.path.abspath(__file__))
for p in (_SCRIPTS, _LANCEDIR):
    if p not in sys.path:
        sys.path.insert(0, p)

DEFAULT_CKPT = os.path.join(
    _REPO, "checkpoints_qwen25/nusc_planning_b5pp_1cam_qwen3vl_multimodal/final"
)
DEFAULT_INFOS = os.path.join(_REPO, "data/uniad_infos/nuscenes_infos_temporal_train.pkl")
DEFAULT_NUSC = os.path.join(_REPO, "data/nuscenes")
DEFAULT_HDMAP = os.path.join(_REPO, "data/preproc/hdmap_bev")
DEFAULT_BBOX = os.path.join(_REPO, "data/preproc/bbox_egostate_train.jsonl")


def _percentile(xs, q):
    return float(np.percentile(np.asarray(xs, dtype=np.float64), q))


# ----------------------------------------------------------------------------
# Path A: baseline dataset + ViT forward
# ----------------------------------------------------------------------------
def bench_baseline(lance_path, ckpt, steps, batch_size, num_workers):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from multimodal_planning_dataset import MultiModalPlanningDataset
    import lance

    device = "cuda"
    model = AutoModelForImageTextToText.from_pretrained(ckpt, dtype=torch.bfloat16).to(device).eval()
    proc = AutoProcessor.from_pretrained(ckpt)
    inner = model.model

    # Limit the baseline dataset to the SAME subset size as the Lance file so
    # both paths cover identical sample counts.
    n_subset = lance.dataset(lance_path).count_rows()

    ds = MultiModalPlanningDataset(
        infos_path=DEFAULT_INFOS, nusc_root=DEFAULT_NUSC, processor=proc,
        max_length=2560, num_past_frames=4, num_future_waypoints=6, video_fps=2.0,
        max_samples=n_subset, require_full_future=True,
        planning_cams=["CAM_FRONT"], require_all_cams=True,
        hdmap_dir=DEFAULT_HDMAP, bbox_jsonl=DEFAULT_BBOX, split="train",
        modality_dropout_p=0.0,
    )

    def collate(batch):
        # Each item already has pixel_values_videos + video_grid_thw. We only
        # need those two for the ViT forward; stack into a list (variable grids
        # per item are run one-by-one to mirror the eval path).
        return {
            "pv": [b["pixel_values_videos"] for b in batch],
            "grid": [b["video_grid_thw"] for b in batch],
        }

    dl = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, num_workers=num_workers, shuffle=False,
        pin_memory=True, prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0, collate_fn=collate,
    )
    return _run_loop(dl, steps, batch_size, consume=_make_vit_consume(inner, device),
                     label="A_baseline_vit")


def _make_vit_consume(inner, device):
    def consume(batch):
        # GPU work = the ViT vision-tower forward over each item's clip.
        for pv, grid in zip(batch["pv"], batch["grid"]):
            pv = pv.to(device, torch.bfloat16, non_blocking=True)
            grid = grid.reshape(1, 3).to(device, non_blocking=True)
            with torch.no_grad():
                _ = inner.get_video_features(pv, grid).pooler_output
        torch.cuda.synchronize()
    return consume


# ----------------------------------------------------------------------------
# Path B: Lance cached-token dataset (no ViT)
# ----------------------------------------------------------------------------
def bench_lance(lance_path, steps, batch_size, num_workers):
    from lance_dataset import LanceMMDataset, collate

    device = "cuda"
    ds = LanceMMDataset(lance_path)
    # forkserver: lance is not fork-safe (see its own warning).
    import multiprocessing as mp
    ctx = mp.get_context("forkserver") if num_workers > 0 else None
    dl = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, num_workers=num_workers, shuffle=False,
        pin_memory=True, prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0, collate_fn=collate,
        multiprocessing_context=ctx,
    )

    def consume(batch):
        # GPU work = move the cached tokens to GPU (the train loop would feed
        # these straight into the LM — no ViT). This is the real per-step GPU
        # cost of the cached path's *vision* stage.
        _ = batch["vis_tokens"].to(device, torch.bfloat16, non_blocking=True)
        torch.cuda.synchronize()

    return _run_loop(dl, steps, batch_size, consume=consume, label="B_lance_cached")


# ----------------------------------------------------------------------------
# Shared steady-state loop
# ----------------------------------------------------------------------------
def _run_loop(dl, steps, batch_size, consume, label, warmup=10):
    wall_lat, gpu_lat = [], []
    it = iter(dl)
    n = 0
    total_samples = 0
    t_start_steady = None
    while n < steps + warmup:
        t0 = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl)
            batch = next(it)
        t_fetch = time.perf_counter()
        consume(batch)
        t1 = time.perf_counter()

        if n >= warmup:
            wall_lat.append(t1 - t0)
            gpu_lat.append(t1 - t_fetch)
            bs = (batch["vis_tokens"].shape[0] if "vis_tokens" in batch
                  else len(batch["pv"]))
            total_samples += bs
            if t_start_steady is None:
                t_start_steady = t0
        n += 1
    t_end = time.perf_counter()

    wall = float(np.sum(wall_lat))
    gpu = float(np.sum(gpu_lat))
    elapsed = t_end - t_start_steady
    sps = total_samples / elapsed
    stall = max(0.0, 1.0 - gpu / wall)
    return {
        "label": label,
        "steps": len(wall_lat),
        "samples": total_samples,
        "samples_per_sec": sps,
        "p50_batch_ms": _percentile(wall_lat, 50) * 1e3,
        "p90_batch_ms": _percentile(wall_lat, 90) * 1e3,
        "gpu_busy_frac": gpu / wall,
        "stall_pct": stall * 100.0,
    }


def _print_table(a, b):
    sp = a["samples_per_sec"]
    speedup = b["samples_per_sec"] / sp if sp > 0 else float("nan")
    print("\n================ THROUGHPUT A/B ================")
    hdr = f"{'metric':<22}{'A baseline (ViT)':>20}{'B lance (cached)':>20}"
    print(hdr)
    print("-" * len(hdr))
    rows = [
        ("samples/sec", f"{a['samples_per_sec']:.2f}", f"{b['samples_per_sec']:.2f}"),
        ("p50 batch latency ms", f"{a['p50_batch_ms']:.1f}", f"{b['p50_batch_ms']:.1f}"),
        ("p90 batch latency ms", f"{a['p90_batch_ms']:.1f}", f"{b['p90_batch_ms']:.1f}"),
        ("GPU-idle/stall %", f"{a['stall_pct']:.1f}", f"{b['stall_pct']:.1f}"),
        ("steady steps", str(a["steps"]), str(b["steps"])),
    ]
    for name, av, bv in rows:
        print(f"{name:<22}{av:>20}{bv:>20}")
    print("-" * len(hdr))
    print(f"SPEEDUP (B/A samples/sec) = {speedup:.2f}x")
    print("================================================\n")
    return speedup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lance", required=True)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--only", choices=["a", "b", "both"], default="both")
    args = ap.parse_args()

    res = {}
    if args.only in ("a", "both"):
        print("[bench] running Path A (baseline ViT-in-loop)...")
        res["a"] = bench_baseline(args.lance, args.ckpt, args.steps,
                                  args.batch_size, args.num_workers)
        print(f"[bench] A: {res['a']}")
    if args.only in ("b", "both"):
        print("[bench] running Path B (Lance cached tokens)...")
        res["b"] = bench_lance(args.lance, args.steps, args.batch_size, args.num_workers)
        print(f"[bench] B: {res['b']}")
    if args.only == "both":
        _print_table(res["a"], res["b"])


if __name__ == "__main__":
    raise SystemExit(main())
