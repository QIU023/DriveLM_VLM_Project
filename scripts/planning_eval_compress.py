"""Training-free re-eval of visual-token compression on the nuScenes planning VLA.

Goal (suggestion ① in docs/interview_prep_qwen25vl_architecture.md): produce a
**unified spatial × temporal Pareto frontier on the DRIVING metric** (L2 /
collision vs. total visual-token budget) WITHOUT retraining.

Why no retraining: FasterVLM / PruMerge / PyramidDrop are training-free (they
prune by vision-encoder attention or token similarity at inference). So we take
the already-trained planning checkpoint, insert a compressor in the visual-feature
path, and re-run the same eval as `planning_eval.py`. The temporal compressors
(temporal_pool / VTM / LongVU) plug into the SAME hook → spatial × temporal
combos are just two knobs.

Hook strategy mirrors `train_lora.forward_with_video_xframe_compression`: we
monkey-patch the inner model's `get_video_features` so the compressor runs on the
vision-encoder output before the tokens enter the LLM. That keeps this script
model-agnostic to the generate() internals.

Usage (single GPU; lift the DP-shard loop from planning_eval.main for multi-GPU):

    python scripts/planning_eval_compress.py \
        --ckpt checkpoints_qwen25/nuscenes_planning_3b_full_sft/final \
        --infos-val data/uniad_infos/nuscenes_infos_temporal_val.pkl \
        --spatial-method fastervlm --spatial-ratio 4 \
        --temporal-method temporal_pool --temporal-ratio 2 \
        --max-samples 200 --output eval_results/planning_s4xt2.json

Sweep these knobs (spatial ∈ {none,fastervlm,prumerge,pyramiddrop,crp} ×
{2,4,8,16}, temporal ∈ {none,temporal_pool,vtm,longvu} × {1,2,4}) to draw the
Pareto frontier. Each run records the ACHIEVED token budget so points are
comparable.

NOTE: marked TODO/VERIFY where model-internal attribute names depend on the
installed transformers version — confirm against your build before sweeping.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Reuse everything from planning_eval — metrics, batch builder, runner.
from planning_eval import (  # noqa: E402
    _build_batch_inputs,
    _run_batch,
    decode_waypoints,
    l2_noavg,
    l2_temavg,
)
from planning_dataset import PlanningDataset  # noqa: E402
from trajectory_tokenizer import (  # noqa: E402
    TrajectoryTokenizer,
    TrajectoryTokenizerConfig,
)
from visual_compress import compress_visual_tokens  # noqa: E402  (spatial axis)

try:
    from compressors import make_compressor  # noqa: E402  (temporal axis)
except Exception:  # pragma: no cover
    from scripts.compressors import make_compressor  # type: ignore


# ============================================================================
# The compression hook (the novel part — everything else is reuse)
# ============================================================================

class _FakeVisOut:
    """Mimics the vision-feature return object the LM forward expects."""
    def __init__(self, t):
        self.pooler_output = t


class TokenBudgetMeter:
    """Records visual-token count in -> out so each run logs its Pareto point."""
    def __init__(self):
        self.n_in: List[int] = []
        self.n_out: List[int] = []

    def record(self, n_in: int, n_out: int) -> None:
        self.n_in.append(int(n_in))
        self.n_out.append(int(n_out))

    def summary(self) -> Dict[str, float]:
        if not self.n_in:
            return {"visual_tokens_in": 0, "visual_tokens_out": 0, "ratio": 1.0}
        a = float(np.mean(self.n_in))
        b = float(np.mean(self.n_out))
        return {"visual_tokens_in": a, "visual_tokens_out": b,
                "ratio": (a / b) if b > 0 else float("nan")}


def install_compression_hook(
    model,
    *,
    spatial_method: str,
    spatial_ratio: int,
    temporal_method: str,
    temporal_ratio: int,
    num_past_frames: int,
    meter: TokenBudgetMeter,
):
    """Monkey-patch `inner.get_video_features` to run spatial then temporal
    compression on the (frozen) vision-encoder output. Returns a restore fn.

    Order: SPATIAL (per-frame, intra-frame redundancy) → TEMPORAL (across frames).
    Spatial-first because temporal merging scrambles the per-frame attention
    signal the spatial methods rely on (see prep doc Part 2.5).

    TODO/VERIFY:
      * `model.model` is the Qwen2_5_VLModel inner module on current transformers;
        confirm with `type(model.model).__name__`. Some versions expose
        `get_image_features` for images and `get_video_features` for videos.
      * the vision return exposes `.pooler_output` of shape (sum_T*N, D) — the
        same assumption train_lora.forward_with_video_xframe_compression makes.
      * `compress_visual_tokens(embeds, grid_thw, method, ratio)` expects a
        per-image grid_thw; for the per-frame spatial pass we feed one (1,h,w)
        row per frame. Confirm the (h,w) factorization matches the encoder's
        token layout for your min/max_pixels.
    """
    inner = model.model  # VERIFY: Qwen2_5_VLModel
    orig = inner.get_video_features

    temporal = None
    if temporal_method and temporal_method != "none" and temporal_ratio > 1:
        # e.g. temporal_pool keeps N tokens from T frames; VTM/LongVU prune.
        temporal = make_compressor(temporal_method)  # TODO: pass ratio kwargs per ctor

    def _patched(pixel_values_videos, video_grid_thw):
        with torch.no_grad():
            real = orig(pixel_values_videos, video_grid_thw)
            embeds = real.pooler_output
            if isinstance(embeds, (tuple, list)):
                embeds = torch.cat([e for e in embeds], dim=0)
            embeds = embeds.detach()

        # ---- reshape to (B, T, N, D). VERIFY grid bookkeeping for multi-cam. ----
        D = embeds.shape[-1]
        n_total = embeds.shape[0]
        T = int(num_past_frames)
        # naive single-sample assumption for the skeleton; generalize per batch.
        N = n_total // max(T, 1)
        n_in = n_total

        # ---- SPATIAL axis: prune each frame independently (training-free) ----
        if spatial_method and spatial_method != "none" and spatial_ratio > 1:
            per_frame = embeds.view(T, N, D)
            kept = []
            for f in range(T):
                grid = torch.tensor([[1, *_factor_hw(N)]], device=embeds.device)
                comp, _ = compress_visual_tokens(
                    per_frame[f], grid, spatial_method, spatial_ratio
                )
                kept.append(comp)
            embeds = torch.cat(kept, dim=0)
            N = embeds.shape[0] // T

        # ---- TEMPORAL axis: merge across frames ----
        if temporal is not None:
            frames = embeds.view(1, T, N, D)
            embeds = temporal(frames).reshape(-1, D)  # (N', D)

        meter.record(n_in, embeds.shape[0])
        return _FakeVisOut([embeds])  # one "video" of N' tokens

    inner.get_video_features = _patched

    def _restore():
        inner.get_video_features = orig

    return _restore


def _factor_hw(n: int) -> tuple:
    """Best-effort (h, w) factorization of a per-frame token count for the
    spatial compressor's grid_thw. Replace with the encoder's true layout."""
    h = int(math.isqrt(n))
    while h > 1 and n % h != 0:
        h -= 1
    return h, n // max(h, 1)


# ============================================================================
# Eval loop (single-process core; reuses planning_eval helpers verbatim)
# ============================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--infos-val", required=True)
    p.add_argument("--nusc-root", default=os.path.join(_BASE_DIR, "data", "nuscenes"))
    p.add_argument("--max-samples", type=int, default=200)
    p.add_argument("--output", default=None)
    p.add_argument("--num-past-frames", type=int, default=4)
    p.add_argument("--num-future-waypoints", type=int, default=6)
    p.add_argument("--video-fps", type=float, default=2.0)
    p.add_argument("--planning-cams", default="CAM_FRONT")
    p.add_argument("--max-new-tokens", type=int, default=20)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--batch-size", type=int, default=4)
    # ---- compression knobs (the two axes) ----
    p.add_argument("--spatial-method", default="none",
                   choices=["none", "fastervlm", "prumerge", "pyramiddrop", "crp", "avg_pool"])
    p.add_argument("--spatial-ratio", type=int, default=1)
    p.add_argument("--temporal-method", default="none",
                   choices=["none", "temporal_pool", "vtm", "longvu"])
    p.add_argument("--temporal-ratio", type=int, default=1)
    args = p.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    model = AutoModelForImageTextToText.from_pretrained(
        args.ckpt, torch_dtype=dtype, attn_implementation="sdpa",
    ).to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.ckpt)
    processor.tokenizer.padding_side = "left"

    traj_tok = TrajectoryTokenizer(
        TrajectoryTokenizerConfig(num_waypoints=args.num_future_waypoints)
    )
    planning_cams = [c.strip() for c in args.planning_cams.split(",") if c.strip()]
    eval_max_length = 4096 if len(planning_cams) == 1 else 8192

    ds = PlanningDataset(
        infos_path=args.infos_val, nusc_root=args.nusc_root, processor=processor,
        max_length=eval_max_length, num_past_frames=args.num_past_frames,
        num_future_waypoints=args.num_future_waypoints, video_fps=args.video_fps,
        vla_loss_mode="answer_and_traj", max_samples=args.max_samples,
        require_full_future=True, planning_cams=planning_cams, require_all_cams=True,
    )
    n_total = len(ds)
    print(f"[compress-eval] samples={n_total} spatial={args.spatial_method}x{args.spatial_ratio} "
          f"temporal={args.temporal_method}x{args.temporal_ratio}")

    meter = TokenBudgetMeter()
    restore = install_compression_hook(
        model,
        spatial_method=args.spatial_method, spatial_ratio=args.spatial_ratio,
        temporal_method=args.temporal_method, temporal_ratio=args.temporal_ratio,
        num_past_frames=args.num_past_frames, meter=meter,
    )

    temavg = {k: [] for k in ["L2_1s", "L2_2s", "L2_3s", "L2_avg"]}
    noavg = {k: [] for k in ["L2_1s", "L2_2s", "L2_3s", "L2_avg"]}
    t0 = time.time()
    bs = max(1, args.batch_size)
    try:
        with torch.inference_mode():
            for bstart in range(0, n_total, bs):
                batch_idx = list(range(bstart, min(bstart + bs, n_total)))
                inputs, infos, futures, samples = _build_batch_inputs(
                    ds, processor, args, batch_idx, planning_cams
                )
                new_tokens = _run_batch(
                    model, processor, inputs, device, dtype, args.max_new_tokens
                )
                for j, ids in enumerate(new_tokens.cpu().tolist()):
                    if processor.tokenizer.pad_token_id in ids:
                        ids = ids[:ids.index(processor.tokenizer.pad_token_id)]
                    pred = decode_waypoints(ids, traj_tok, args.num_future_waypoints)
                    gt = samples[j]["_meta_waypoints"].cpu().numpy()
                    valid = samples[j]["_meta_valid_mask"].cpu().numpy()
                    for store, fn in ((temavg, l2_temavg), (noavg, l2_noavg)):
                        m = fn(pred, gt, valid)
                        for k in store:
                            if not math.isnan(m[k]):
                                store[k].append(m[k])
    finally:
        restore()

    def _mean(d):
        return {k: (float(np.mean(v)) if v else float("nan")) for k, v in d.items()}

    result = {
        "ckpt": args.ckpt,
        "n_samples": n_total,
        "compression": {
            "spatial_method": args.spatial_method, "spatial_ratio": args.spatial_ratio,
            "temporal_method": args.temporal_method, "temporal_ratio": args.temporal_ratio,
            **meter.summary(),
        },
        "TemAvg": _mean(temavg),
        "NoAvg": _mean(noavg),
        "eval_seconds": round(time.time() - t0, 1),
        # TODO: collision rate — lift the _planning_metric port from planning_eval.
    }
    out = args.output or os.path.join(os.path.dirname(args.ckpt) or ".",
                                      "eval_results_compress.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result["compression"], indent=2))
    print(f"TemAvg L2_avg={result['TemAvg']['L2_avg']:.3f}  -> {out}")


if __name__ == "__main__":
    main()
