#!/usr/bin/env /usr/bin/python3
"""Stage-A entrypoint: full-modal spatial-compression eval for the 1-cam
Qwen3-VL-4B planning VLA (FasterVLM / PruMerge on VIDEO tokens only).

This is a NEW file (does NOT modify the live training files or the existing
scripts/planning_eval_compress.py). It closes the four gaps that block the
9-cell overnight benchmark on the B.5'' v2 ckpt:

  (G1) FULL-MODAL input. The original planning_eval_compress.py instantiates
       the video-only ``PlanningDataset`` (scripts/planning_eval_compress.py:472)
       so HD-map BEV + bbox + ego are NOT injected. The benchmark requires
       full-modal input with ONLY the video pruned. We instantiate
       ``MultiModalPlanningDataset`` instead; planning_eval._build_batch_inputs
       (scripts/planning_eval.py:778-831) duck-types this and injects HD-map +
       bbox automatically.

  (G2) Qwen3-VL DEEPSTACK. The original hook returns a _FakeVisOut with only
       ``.pooler_output``; Qwen3VLModel.forward reads ``.deepstack_features``
       too (modeling_qwen3_vl.py:1306) and consumes it in _deepstack_process,
       which requires the deepstack rows to match the COMPRESSED token count.
       See deploy/trt_b5ppp/F8_planning_eval_compress.patch for the diagnosis.
       This file installs a deepstack-aware hook (index-aligned pruning of every
       deepstack level), so it works WITHOUT having to apply the F8 patch to the
       original script.

  (G3) COLLISION metric. The original leaves collision as a TODO
       (scripts/planning_eval_compress.py:585). We import the UniAD-port
       collision helper from planning_eval and report collision_{1s,2s,3s,avg}.

  (G4) Compress ONLY video. HD-map image tokens (~121) are never touched; we
       only patch get_VIDEO_features.

Compression ratio knob: ``--spatial-ratio R`` keeps n_video // R tokens. With
the cfg-derived baseline of 2800 video tokens (grid [2,56,100], post-merge
2*28*50=2800), R in {2,4,8,16} -> {1400,700,350,175}. (The deployment plan's
nominal targets {1440,720,360,180} assume a 2880-token baseline; the live ckpt
produces 2800 — the runner records the ACHIEVED visual_tokens_in/out so the
Pareto point is exact regardless.)

Usage (single GPU; one CUDA_VISIBLE_DEVICES per cell):
    CUDA_VISIBLE_DEVICES=0 /usr/bin/python3 scripts/planning_eval_compress_mm.py \
        --ckpt checkpoints_qwen25/nusc_planning_b5pp_1cam_qwen3vl_multimodal/final \
        --spatial-method fastervlm --spatial-ratio 4 \
        --max-samples 0 --output eval_results/compress_bench_v3/fastervlm_r4.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import List

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_BASE_DIR, "scripts"))

# Reuse the compression machinery + metrics from the existing scripts.
from planning_eval import (  # noqa: E402
    _build_batch_inputs,
    decode_waypoints,
    l2_noavg,
    l2_temavg,
)
from planning_eval_compress import (  # noqa: E402
    TokenBudgetMeter,
    _per_item_post_counts,
    _trim_video_pad_for_compression,
    _factor_hw,
)
from multimodal_planning_dataset import MultiModalPlanningDataset  # noqa: E402
from trajectory_tokenizer import (  # noqa: E402
    TrajectoryTokenizer,
    TrajectoryTokenizerConfig,
)
from visual_compress import compress_visual_tokens  # noqa: E402

# Collision port (UniAD) — same helper + horizon indices planning_eval uses.
try:
    from planning_eval import (  # type: ignore  # noqa: E402
        _uniad_compute_collision_per_sample as _collision_fn,
        HORIZON_IDX as _HORIZON_IDX,
    )
except Exception:  # pragma: no cover
    _collision_fn = None
    _HORIZON_IDX = (1, 3, 5)


class _FakeVisOut:
    """Vision-feature return object exposing BOTH pooler_output and
    deepstack_features (Qwen3-VL forward reads both — modeling_qwen3_vl.py:1306)."""
    def __init__(self, pooler, deepstack_features=None):
        self.pooler_output = pooler
        self.deepstack_features = deepstack_features


def _spatial_prune_block(block_embed: torch.Tensor, method: str, ratio: int):
    """Prune ONE video block (post-merge rows) and return (compressed, kept_idx).

    All deployable methods (fastervlm/prumerge/pyramiddrop) are norm-top-k
    selections, so a clean kept-index set exists; we use it to prune deepstack
    levels in lockstep. The compressed embeds come from the shared
    compress_visual_tokens() (same impl as the TRT-side bench)."""
    n = block_embed.shape[0]
    grid = torch.tensor([[1, 1, n]], device=block_embed.device, dtype=torch.int64)
    comp, _ = compress_visual_tokens(block_embed, grid, method, int(ratio))
    k = max(1, n // int(ratio))
    _, idx = block_embed.norm(dim=-1).topk(k)
    kept_idx = idx.sort().values
    return comp, kept_idx


def install_video_compress_hook(model, *, method: str, ratio: int,
                                meter: TokenBudgetMeter, state: dict):
    inner = model.model
    orig = inner.get_video_features

    def _patched(_pv_unused, _grid_unused, **_kw):
        pv = state["orig_pv"]
        grid = state["orig_grid"]
        with torch.no_grad():
            real = orig(pv, grid)
        pooler = real.pooler_output
        if not isinstance(pooler, (tuple, list)):
            pooler = list(torch.split(pooler, state["per_item_orig"]))
        ds_levels = getattr(real, "deepstack_features", None)
        has_deepstack = isinstance(ds_levels, (list, tuple)) and len(ds_levels) > 0
        if has_deepstack:
            ds_split = [list(torch.split(lvl, state["per_item_orig"])) for lvl in ds_levels]

        per_item_comp = state["per_item_comp"]
        out_pooler: List[torch.Tensor] = []
        out_deep: List[List[torch.Tensor]] = ([[] for _ in ds_levels]
                                              if has_deepstack else None)
        n_in = 0
        n_out = 0
        for i, e in enumerate(pooler):
            e = e.detach()
            n_in += e.shape[0]
            if method != "none" and int(ratio) > 1:
                comp, kept_idx = _spatial_prune_block(e, method, ratio)
            else:
                comp, kept_idx = e, torch.arange(e.shape[0], device=e.device)
            target = per_item_comp[i]
            comp = _fit_rows(comp, target)
            out_pooler.append(comp)
            n_out += comp.shape[0]
            if has_deepstack:
                for lvl_i in range(len(ds_levels)):
                    d_block = ds_split[lvl_i][i].detach()
                    sel = kept_idx.to(d_block.device)
                    if sel.numel() > d_block.shape[0]:
                        sel = sel[:d_block.shape[0]]
                    d_comp = d_block.index_select(0, sel)
                    out_deep[lvl_i].append(_fit_rows(d_comp, target))
        meter.record(n_in, n_out)
        if has_deepstack:
            ds_out = [torch.cat(level, dim=0) for level in out_deep]
            return _FakeVisOut(out_pooler, deepstack_features=ds_out)
        return _FakeVisOut(out_pooler)

    inner.get_video_features = _patched
    return lambda: setattr(inner, "get_video_features", orig)


def _fit_rows(t: torch.Tensor, target: int) -> torch.Tensor:
    if t.shape[0] == target:
        return t
    if t.shape[0] > target:
        return t[:target]
    pad = target - t.shape[0]
    return torch.cat([t, t.new_zeros((pad, t.shape[-1]))], dim=0)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--infos-val",
                   default=os.path.join(_BASE_DIR, "data/uniad_infos/nuscenes_infos_temporal_val.pkl"))
    p.add_argument("--nusc-root", default=os.path.join(_BASE_DIR, "data", "nuscenes"))
    p.add_argument("--hdmap-dir", default=os.path.join(_BASE_DIR, "data/preproc/hdmap_bev"))
    p.add_argument("--bbox-jsonl", default=os.path.join(_BASE_DIR, "data/preproc/bbox_egostate_val.jsonl"))
    p.add_argument("--split", default="val")
    p.add_argument("--max-samples", type=int, default=0, help="0 = full val")
    p.add_argument("--output", required=True)
    p.add_argument("--num-past-frames", type=int, default=4)
    p.add_argument("--num-future-waypoints", type=int, default=6)
    p.add_argument("--video-fps", type=float, default=2.0)
    p.add_argument("--planning-cams", default="CAM_FRONT")
    p.add_argument("--max-new-tokens", type=int, default=20)
    p.add_argument("--max-length", type=int, default=6144)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--spatial-method", default="none",
                   choices=["none", "fastervlm", "prumerge", "pyramiddrop", "avg_pool"])
    p.add_argument("--spatial-ratio", type=int, default=1)
    args = p.parse_args()

    # DP: under torchrun each rank loads the ckpt + compression hook and runs its
    # 1/world_size strided shard of val; rank 0 gathers per-sample metric lists
    # via all_gather_object and writes the JSON. Mirrors planning_eval.py's proven
    # DP harness (_setup_distributed + all_gather_object, ~25-30x).
    import torch.distributed as dist
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"]); world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank % max(torch.cuda.device_count(), 1)))
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        is_dist = True
    else:
        rank, world_size, local_rank, is_dist = 0, 1, 0, False
        device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    model = AutoModelForImageTextToText.from_pretrained(
        args.ckpt, torch_dtype=dtype, attn_implementation="sdpa",
    ).to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.ckpt)
    processor.tokenizer.padding_side = "left"

    traj_tok = TrajectoryTokenizer(
        TrajectoryTokenizerConfig(num_waypoints=args.num_future_waypoints))
    planning_cams = [c.strip() for c in args.planning_cams.split(",") if c.strip()]
    max_samples = args.max_samples if args.max_samples and args.max_samples > 0 else None

    ds = MultiModalPlanningDataset(
        infos_path=args.infos_val, nusc_root=args.nusc_root, processor=processor,
        max_length=args.max_length, num_past_frames=args.num_past_frames,
        num_future_waypoints=args.num_future_waypoints, video_fps=args.video_fps,
        vla_loss_mode="answer_and_traj", max_samples=max_samples,
        require_full_future=True, planning_cams=planning_cams, require_all_cams=True,
        hdmap_dir=args.hdmap_dir, bbox_jsonl=args.bbox_jsonl, split=args.split,
        modality_dropout_p=0.0,
    )
    n_total = len(ds)
    my_indices = list(range(rank, n_total, world_size))   # strided shard per rank
    if rank == 0:
        print(f"[compress-mm] samples={n_total} method={args.spatial_method}x{args.spatial_ratio} "
              f"world={world_size} (~{len(my_indices)}/rank) "
              f"(full-modal: HD-map+bbox+ego kept, video pruned)", flush=True)

    merge_size = int(getattr(model.config.vision_config, "spatial_merge_size", 2))
    video_token_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")

    meter = TokenBudgetMeter()
    state: dict = {}
    restore = install_video_compress_hook(
        model, method=args.spatial_method, ratio=args.spatial_ratio,
        meter=meter, state=state)

    temavg = {k: [] for k in ["L2_1s", "L2_2s", "L2_3s", "L2_avg"]}
    noavg = {k: [] for k in ["L2_1s", "L2_2s", "L2_3s", "L2_avg"]}
    coll = {k: [] for k in ["collision_1s", "collision_2s", "collision_3s", "collision_avg"]}
    t0 = time.time()
    bs = max(1, args.batch_size)
    try:
        with torch.inference_mode():
            for bstart in range(0, len(my_indices), bs):
                batch_idx = my_indices[bstart:bstart + bs]
                inputs, infos, futures, samples = _build_batch_inputs(
                    ds, processor, args, batch_idx, planning_cams)
                inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                          for k, v in inputs.items()}
                if "pixel_values_videos" in inputs:
                    inputs["pixel_values_videos"] = inputs["pixel_values_videos"].to(dtype)
                if "pixel_values" in inputs and isinstance(inputs["pixel_values"], torch.Tensor):
                    inputs["pixel_values"] = inputs["pixel_values"].to(dtype)

                ratio = max(1, int(args.spatial_ratio))
                per_item_orig = _per_item_post_counts(inputs["video_grid_thw"], merge_size)
                per_item_comp = [max(1, n // ratio) for n in per_item_orig]
                new_ids, new_mask, new_grid = _trim_video_pad_for_compression(
                    inputs["input_ids"], inputs["attention_mask"],
                    inputs["video_grid_thw"], video_token_id,
                    per_item_orig, per_item_comp, merge_size=merge_size)

                state["orig_pv"] = inputs["pixel_values_videos"]
                state["orig_grid"] = inputs["video_grid_thw"]
                state["per_item_orig"] = per_item_orig
                state["per_item_comp"] = per_item_comp

                prompt_len = new_ids.shape[1]
                gen_kwargs = dict(
                    input_ids=new_ids, attention_mask=new_mask,
                    pixel_values_videos=inputs["pixel_values_videos"],
                    video_grid_thw=new_grid,
                    pixel_values=inputs.get("pixel_values"),
                    image_grid_thw=inputs.get("image_grid_thw"),
                    max_new_tokens=args.max_new_tokens, do_sample=False, num_beams=1,
                    pad_token_id=processor.tokenizer.pad_token_id or 0, use_cache=True,
                )
                gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
                gen = model.generate(**gen_kwargs)
                new_tokens = gen[:, prompt_len:]

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
                    if _collision_fn is not None:
                        try:
                            ch = _collision_fn(
                                pred_wp_ego=pred, gt_wp_ego=gt,
                                future_infos=futures[j], cur_info=infos[j],
                                horizon_indices=_HORIZON_IDX,
                            )
                            for hi, h_idx in enumerate(_HORIZON_IDX):
                                if h_idx >= len(futures[j]) or valid[h_idx] < 1e-6:
                                    ch[hi] = 0
                            coll["collision_1s"].append(int(ch[0]))
                            coll["collision_2s"].append(int(ch[1]))
                            coll["collision_3s"].append(int(ch[2]))
                            coll["collision_avg"].append(int(any(ch)))
                        except Exception:
                            pass
    finally:
        restore()

    # ---- DP gather: merge each rank's per-sample metric lists + token counts.
    payload = {"temavg": temavg, "noavg": noavg, "coll": coll,
               "meter": meter.summary()}
    if is_dist:
        bucket = [None] * world_size
        dist.all_gather_object(bucket, payload)
    else:
        bucket = [payload]
    # concat the per-sample lists across ranks
    def _merge_lists(key):
        out = {}
        for sub in bucket:
            for k, v in sub[key].items():
                out.setdefault(k, []).extend(v)
        return out
    temavg = _merge_lists("temavg"); noavg = _merge_lists("noavg"); coll = _merge_lists("coll")
    # sum token-budget totals across ranks (per-sample ratio is rank-invariant)
    meter_sum = {}
    for sub in bucket:
        for k, v in sub["meter"].items():
            if isinstance(v, (int, float)):
                meter_sum[k] = meter_sum.get(k, 0) + v
            else:
                meter_sum.setdefault(k, v)
    # ratio is a per-sample invariant — recompute from summed totals, don't sum it
    if meter_sum.get("visual_tokens_out"):
        meter_sum["ratio"] = round(meter_sum["visual_tokens_in"] / meter_sum["visual_tokens_out"], 3)

    if rank != 0:
        if is_dist:
            dist.barrier(); dist.destroy_process_group()
        return 0

    def _mean(d):
        return {k: (float(np.mean(v)) if v else float("nan")) for k, v in d.items()}

    result = {
        "ckpt": args.ckpt,
        "n_samples": n_total,
        "method": args.spatial_method,
        "ratio": int(args.spatial_ratio),
        "compression": {"spatial_method": args.spatial_method,
                        "spatial_ratio": int(args.spatial_ratio), **meter_sum},
        "TemAvg": _mean(temavg),
        "NoAvg": _mean(noavg),
        "collision": _mean(coll),
        "n_scored": len(temavg.get("L2_avg", [])),
        "world_size": world_size,
        "eval_seconds": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result["compression"], indent=2))
    print(f"TemAvg L2_avg={result['TemAvg']['L2_avg']}  collision_avg="
          f"{result['collision']['collision_avg']}  n_scored={result['n_scored']}  -> {args.output}")
    if is_dist:
        dist.barrier(); dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
