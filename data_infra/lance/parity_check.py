"""HARD PARITY GATE for the Tier-2 NATIVE 3-cam cache.

On N real 3-cam native samples, compute:
  (A) loss via the LIVE forward_with_video_compression_free (frozen ViT +
      FasterVLM x4 + deepstack), and
  (B) loss via forward_with_cached_vision_tokens reading the cache built from the
      SAME samples by cache_3cam_native.py.

ASSERT  mean(|lossA - lossB| / lossA) < TOL  (default 0.03, int8 tolerance).
Prints both losses per sample. Exits non-zero if the gate fails.

Run AFTER the producer smoke, BEFORE the n-build and the A/B bench.

Usage:
    export HF_HOME=/workspace/.hf_home; unset HF_HUB_OFFLINE
    CUDA_VISIBLE_DEVICES=0 /usr/bin/python3 data_infra/lance/parity_check.py \
        --lance data_infra/lance/nusc_3cam_native_smoke.lance --n 8
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
_SCRIPTS = os.path.join(_REPO, "scripts")
for _p in (_SCRIPTS, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CONFIG = os.path.join(_REPO, "configs", "nuscenes_planning_3cam_qwen3vl_NATIVE.yaml")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lance", default=os.path.join(_HERE, "nusc_3cam_native_smoke.lance"))
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--tol", type=float, default=0.03)
    args = ap.parse_args()

    from transformers import AutoProcessor, AutoModelForImageTextToText
    from train_lora import (
        load_config, collate_fn,
        forward_with_video_compression_free, forward_with_cached_vision_tokens,
    )
    from multimodal_planning_dataset import build_multimodal_planning_dataset
    from cached_native_dataset import CachedNativeDataset, collate_cached
    import lance

    cfg = load_config(CONFIG)
    model_id = args.ckpt or cfg["model_id"]
    device = "cuda"

    proc = AutoProcessor.from_pretrained(model_id)
    proc.image_processor.min_pixels = cfg["min_pixels"]
    proc.image_processor.max_pixels = cfg["max_pixels"]
    video_token_id = proc.tokenizer.convert_tokens_to_ids("<|video_pad|>")
    image_token_id = proc.tokenizer.convert_tokens_to_ids("<|image_pad|>")

    live_ds = build_multimodal_planning_dataset(cfg, proc, split="train")
    cache_ds = CachedNativeDataset(args.lance)

    # Map cache sample_token -> cache row idx; and live needs the SAME samples.
    # The producer cached the FIRST n train rows in order, so live row i == cache
    # sample for the matching token. We align by sample_token to be safe.
    cache_tokens = lance.dataset(args.lance).to_table(columns=["sample_token"]).to_pylist()
    cache_tok_list = [r["sample_token"] for r in cache_tokens]
    # The producer cached live_ds[0..N) but the 8-GPU shards were concatenated in
    # shard order, so cache row order != live order. Build a token -> live_idx map
    # by scanning live_ds[0..N) (the producer's contract: cache is the first N).
    n_cached = len(cache_tok_list)
    live_tok_to_idx = {}
    for li in range(min(len(live_ds), n_cached)):
        live_tok_to_idx[live_ds[li]["_meta_token"]] = li

    # token -> cache row idx
    cache_tok_to_idx = {t: ci for ci, t in enumerate(cache_tok_list)}

    model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16).to(device).eval()

    n = min(args.n, len(cache_ds))
    rel_errs, losses_a, losses_b = [], [], []
    print(f"{'idx':>3}  {'sample_token':>34}  {'lossA(live)':>11}  {'lossB(cache)':>12}  {'rel':>7}")
    for i in range(n):
        tok = cache_tok_list[i]
        assert tok in live_tok_to_idx, f"cache token {tok} not in first {n_cached} live rows"
        live_item = live_ds[live_tok_to_idx[tok]]
        assert live_item["_meta_token"] == tok

        live_batch = collate_fn([live_item])
        live_batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in live_batch.items()}
        with torch.no_grad():
            outA = forward_with_video_compression_free(
                model, dict(live_batch), video_token_id, "fastervlm", 4, 4)
            lossA = float(outA.loss)

        cached_batch = collate_cached([cache_ds[i]])
        cached_batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in cached_batch.items()}
        with torch.no_grad():
            outB = forward_with_cached_vision_tokens(
                model, cached_batch, video_token_id, image_token_id)
            lossB = float(outB.loss)

        rel = abs(lossA - lossB) / abs(lossA) if lossA != 0 else float("nan")
        rel_errs.append(rel); losses_a.append(lossA); losses_b.append(lossB)
        print(f"{i:>3}  {tok:>34}  {lossA:>11.5f}  {lossB:>12.5f}  {rel*100:>6.3f}%")

    mean_rel = float(np.mean(rel_errs))
    print("-" * 78)
    print(f"mean lossA = {np.mean(losses_a):.5f}  mean lossB = {np.mean(losses_b):.5f}")
    print(f"mean |lossA-lossB|/lossA = {mean_rel*100:.4f}%   (TOL = {args.tol*100:.2f}%)")
    if mean_rel < args.tol:
        print("PARITY GATE: PASS")
        return 0
    print("PARITY GATE: FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
