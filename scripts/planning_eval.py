"""Offline planning evaluation for the nuScenes Phase-B VLA.

Loads any HF checkpoint produced by `train_lora.py --config configs/nuscenes_planning_*.yaml`,
iterates the val infos, runs greedy `model.generate()` to produce trajectory
tokens, decodes to (Δx, Δy) waypoints, then computes:

  * L2 at 1 s (idx 1), 2 s (idx 3), 3 s (idx 5)
  * L2 average — under BOTH VAD's TemAvg protocol AND UniAD's NoAvg protocol
      - TemAvg (VAD): mean L2 over ALL future timesteps that fall within each
        cumulative horizon. For a 1 s horizon we average errors at t in {0.5, 1.0} s.
      - NoAvg  (UniAD): point-wise L2 at exactly t = 1/2/3 s.
  * Collision rate at 1 s, 2 s, 3 s — port of UniAD's footprint-overlap check:
    construct the ego footprint bbox at each predicted future waypoint, and
    check overlap against every annotated agent at that timestamp.

Output: a JSON file matching the AutoVLA/UniAD/VAD table format so it's drop-in
for paper comparison.

Modes:
  - Single-GPU: `python scripts/planning_eval.py --ckpt ... --batch-size 4`
    Falls back to a plain loop over rank 0 only.
  - Multi-GPU data-parallel: launch with
      torchrun --nproc_per_node=8 scripts/planning_eval.py --ckpt ... --batch-size 4
    Each rank loads the same ckpt, processes its 1/world_size shard of val
    infos, and rank 0 aggregates via `dist.gather_object`. ~25-30x speedup
    over the original single-GPU bs=1 path on 8x 5090.

Notes:
  - This script works with FSDP-sharded ckpts produced by `train_lora.py` with
    `train_mode: full_sft` (state_dict gathered on rank 0 -> plain HF dir).
  - Collisions: ground-truth agent boxes come from each future frame's
    `gt_boxes` (in current-frame ego coordinates per UniAD's transform).
  - Batched generation uses left-padding (Qwen2.5-VL tokenizer default is right)
    so the prompt suffix aligns across the batch.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from planning_dataset import (  # noqa: E402
    DEFAULT_PLANNING_CAMS,
    PROMPT_TEXT,
    PlanningDataset,
    _build_user_content_multicam,
    _format_ego_speed_preamble,
)
from trajectory_tokenizer import (  # noqa: E402
    TrajectoryTokenizer,
    TrajectoryTokenizerConfig,
)
from _planning_metric import (  # noqa: E402
    compute_collision_per_sample as _uniad_compute_collision_per_sample,
    H as _UNIAD_EGO_LENGTH,
    W as _UNIAD_EGO_WIDTH,
)


HZ = 2.0
DT = 1.0 / HZ              # 0.5 s
HORIZONS = (1.0, 2.0, 3.0)
HORIZON_IDX = (1, 3, 5)    # 0-indexed waypoint at each horizon (t = 1/2/3 s)
# Ego footprint — matches UniAD/VAD/ST-P3 exactly (Renault Zoe: 4.084 m length,
# 1.85 m width). Both papers use these values in their official planning-metric
# code (see UniAD planning_head_plugin/metric_stp3.py and VAD planner/metric_stp3.py).
EGO_LENGTH_M = 4.084
EGO_WIDTH_M = 1.85
EGO_HALF_LEN_M = EGO_LENGTH_M * 0.5
EGO_HALF_WID_M = EGO_WIDTH_M * 0.5
# nuScenes ego pose is reported at the rear-axle (lidar-top mount point), so
# the box CENTRE is +0.5 m forward of the pose origin along ego +x. Both UniAD
# and VAD shift the box by +0.5 m forward to match this (`[-H/2 + 0.5, ...]`
# in their code). We replicate that shift here.
EGO_BOX_FWD_OFFSET_M = 0.5


# ============================================================================
# Helpers: decoding the model output
# ============================================================================

def _find_traj_block(token_ids: List[int], traj_start_id: int, traj_end_id: int) -> List[int]:
    """Extract the bin tokens between <traj_start> and <traj_end>."""
    try:
        i0 = token_ids.index(traj_start_id)
    except ValueError:
        return []
    try:
        i1 = token_ids.index(traj_end_id, i0 + 1)
    except ValueError:
        i1 = len(token_ids)
    return token_ids[i0:i1 + 1]


def decode_waypoints(generated_ids: List[int], traj_tok: TrajectoryTokenizer,
                     num_waypoints: int) -> np.ndarray:
    """Return (num_waypoints, 2) of decoded (dx, dy) in metres. Pads zeros if
    generation produced fewer."""
    block = _find_traj_block(
        generated_ids, traj_tok.cfg.traj_start_id, traj_tok.cfg.traj_end_id
    )
    if block:
        wp = traj_tok.decode(block)
    else:
        # No boundary tokens at all -> fall back to "raw" decode over the whole
        # generation, which the tokenizer accepts.
        wp = traj_tok.decode(generated_ids)
    out = np.zeros((num_waypoints, 2), dtype=np.float32)
    n = min(num_waypoints, wp.shape[0])
    if n > 0:
        out[:n] = wp[:n]
    return out


# ============================================================================
# Collision math is in scripts/_planning_metric.py (verbatim UniAD port).
# ============================================================================


# ============================================================================
# L2 protocols
# ============================================================================

def l2_temavg(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> Dict[str, float]:
    """VAD-style TemAvg: average error over all timesteps within horizon.
    For 1 s horizon we average idx 0..1; for 2 s -> 0..3; for 3 s -> 0..5."""
    out: Dict[str, float] = {}
    for horizon_idx, horizon_s in zip(HORIZON_IDX, HORIZONS):
        sl = slice(0, horizon_idx + 1)
        diff = pred[sl] - gt[sl]
        l2 = np.sqrt((diff ** 2).sum(axis=-1))
        m = valid[sl]
        if m.sum() < 1e-6:
            out[f"L2_{int(horizon_s)}s"] = float("nan")
        else:
            out[f"L2_{int(horizon_s)}s"] = float((l2 * m).sum() / m.sum())
    vals = [out[k] for k in ["L2_1s", "L2_2s", "L2_3s"] if not math.isnan(out[k])]
    out["L2_avg"] = float(np.mean(vals)) if vals else float("nan")
    return out


def l2_noavg(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> Dict[str, float]:
    """UniAD-style NoAvg: point-wise L2 at exactly t=1/2/3 s."""
    out: Dict[str, float] = {}
    for horizon_idx, horizon_s in zip(HORIZON_IDX, HORIZONS):
        if valid[horizon_idx] < 1e-6:
            out[f"L2_{int(horizon_s)}s"] = float("nan")
            continue
        diff = pred[horizon_idx] - gt[horizon_idx]
        out[f"L2_{int(horizon_s)}s"] = float(math.hypot(*diff))
    vals = [out[k] for k in ["L2_1s", "L2_2s", "L2_3s"] if not math.isnan(out[k])]
    out["L2_avg"] = float(np.mean(vals)) if vals else float("nan")
    return out


# ============================================================================
# Distributed helpers
# ============================================================================

def _init_distributed() -> Tuple[int, int, int, bool]:
    """Initialise torch.distributed if launched under torchrun.

    Returns (rank, world_size, local_rank, is_distributed).
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank % max(torch.cuda.device_count(), 1)))
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
        return rank, world_size, local_rank, True
    return 0, 1, 0, False


def _is_rank0(rank: int) -> bool:
    return rank == 0


def _log(rank: int, msg: str) -> None:
    if _is_rank0(rank):
        print(msg, flush=True)


# ============================================================================
# Batched inference core
# ============================================================================

def _build_batch_inputs(
    ds: PlanningDataset,
    processor,
    args,
    indices_local: List[int],
    planning_cams: List[str],
) -> Tuple[Dict, List[dict], List[List[dict]], List[dict]]:
    """For a list of dataset positions (local indices into ds[]), build the
    processor inputs once and return:

      inputs, per-sample info, per-sample future_infos, per-sample sample dict.
    """
    from transformers.video_utils import VideoMetadata  # local import to avoid cost when DP disabled

    texts: List[str] = []
    all_clips: List[List[Image.Image]] = []
    all_md: List[VideoMetadata] = []
    samples: List[dict] = []
    infos: List[dict] = []
    futures: List[List[dict]] = []

    for i in indices_local:
        sample = ds[i]
        samples.append(sample)
        base_idx = ds._keep[i]
        info = ds.infos[base_idx]
        infos.append(info)
        futures.append(ds._walk_future(base_idx))
        hist = ds._walk_history(base_idx)
        if len(planning_cams) == 1:
            clips = [ds._load_frames(hist, planning_cams[0])]
        else:
            clips = ds._load_frames_multicam(hist)
        user_content = _build_user_content_multicam(info, planning_cams)
        sys_user_messages = [{"role": "user", "content": user_content}]
        text = processor.apply_chat_template(
            sys_user_messages, tokenize=False, add_generation_prompt=True
        )
        texts.append(text)
        for clip in clips:
            all_clips.append(clip)
            all_md.append(
                VideoMetadata(
                    total_num_frames=len(clip),
                    fps=args.video_fps,
                    frames_indices=list(range(len(clip))),
                    height=clip[0].height,
                    width=clip[0].width,
                )
            )

    inputs = processor(
        text=texts,
        videos=all_clips,
        video_metadata=all_md,
        return_tensors="pt",
        padding=True,
    )
    return inputs, infos, futures, samples


def _run_batch(
    model,
    processor,
    inputs: Dict,
    device: torch.device,
    dtype: torch.dtype,
    max_new_tokens: int,
):
    """Move inputs to device, run greedy generate, return (gen_tokens, prompt_len)."""
    inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    if "pixel_values_videos" in inputs:
        inputs["pixel_values_videos"] = inputs["pixel_values_videos"].to(dtype)
    if "pixel_values" in inputs and isinstance(inputs["pixel_values"], torch.Tensor):
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
    prompt_len = inputs["input_ids"].shape[1]
    gen = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=processor.tokenizer.pad_token_id or 0,
        use_cache=True,
    )
    # Slice off the prompt (works for left-padding: prompt is left-aligned to
    # column prompt_len-1 across the batch; newly-generated tokens start at col
    # prompt_len for every row).
    new_tokens = gen[:, prompt_len:]
    return new_tokens


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="HF model checkpoint dir")
    p.add_argument("--infos-val", required=True, help="path to nuscenes_infos_temporal_val.pkl")
    p.add_argument("--nusc-root", default=os.path.join(_BASE_DIR, "data", "nuscenes"))
    p.add_argument("--max-samples", type=int, default=None,
                   help="Cap eval to N samples (default: all val)")
    p.add_argument("--output", default=None, help="Output JSON path (defaults to <ckpt>/eval_results.json)")
    p.add_argument("--num-past-frames", type=int, default=4)
    p.add_argument("--num-future-waypoints", type=int, default=6)
    p.add_argument("--video-fps", type=float, default=2.0)
    p.add_argument(
        "--planning-cams",
        default="CAM_FRONT",
        help="Comma-separated cam list, e.g. 'CAM_FRONT,CAM_FRONT_LEFT,CAM_FRONT_RIGHT' "
             "(AutoVLA 3-cam). Must match training config.",
    )
    p.add_argument("--max-new-tokens", type=int, default=20,
                   help="Greedy generate budget; 1 start + 12 bins + 1 end is enough.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--batch-size", type=int, default=4,
                   help="Per-rank batch size for model.generate (default: 4).")
    args = p.parse_args()

    rank, world_size, local_rank, is_dist = _init_distributed()

    # In distributed mode, pin each rank to its own GPU.
    if is_dist:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    _log(rank, f"[planning_eval] world_size={world_size} rank={rank} local_rank={local_rank} "
               f"device={device} dtype={args.dtype} batch_size={args.batch_size}")
    _log(rank, f"[planning_eval] loading model from {args.ckpt}")

    model = AutoModelForImageTextToText.from_pretrained(
        args.ckpt, torch_dtype=dtype, attn_implementation="sdpa",
    ).to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.ckpt)
    # Batched greedy generation requires left-padding so newly generated
    # tokens start at the same column for every row.
    processor.tokenizer.padding_side = "left"

    traj_cfg = TrajectoryTokenizerConfig(num_waypoints=args.num_future_waypoints)
    traj_tok = TrajectoryTokenizer(traj_cfg)

    planning_cams = [c.strip() for c in args.planning_cams.split(",") if c.strip()]
    # Multi-cam expands visual tokens ~Nx; raise the eval max_length to match
    # the 3-cam training config (8192). Single-cam keeps 4096 for back-compat.
    eval_max_length = 4096 if len(planning_cams) == 1 else 8192
    ds = PlanningDataset(
        infos_path=args.infos_val,
        nusc_root=args.nusc_root,
        processor=processor,
        max_length=eval_max_length,
        num_past_frames=args.num_past_frames,
        num_future_waypoints=args.num_future_waypoints,
        video_fps=args.video_fps,
        vla_loss_mode="answer_and_traj",
        max_samples=args.max_samples,
        require_full_future=True,
        planning_cams=planning_cams,
        require_all_cams=True,
    )

    n_total = len(ds)
    _log(rank, f"[planning_eval] val samples: {n_total}")
    if n_total == 0:
        raise RuntimeError("Empty val set after require_full_future filter.")

    # Stride-shard across ranks (i, i+W, i+2W, ...). This keeps batches roughly
    # balanced even if some samples (the multi-cam tail) are slower than others.
    shard_indices: List[int] = list(range(rank, n_total, world_size))

    # Per-sample local stats; we keep PER-SAMPLE values (not running means) so
    # rank 0 can aggregate exactly with no numerical loss.
    local_temavg: Dict[str, List[float]] = {k: [] for k in ["L2_1s", "L2_2s", "L2_3s", "L2_avg"]}
    local_noavg: Dict[str, List[float]] = {k: [] for k in ["L2_1s", "L2_2s", "L2_3s", "L2_avg"]}
    local_coll: Dict[str, List[int]] = {k: [] for k in ["collision_1s", "collision_2s", "collision_3s", "collision_avg"]}

    t0 = time.time()
    bs = max(1, int(args.batch_size))
    n_local = len(shard_indices)

    with torch.inference_mode():
        for bstart in range(0, n_local, bs):
            batch_idx = shard_indices[bstart:bstart + bs]
            inputs, infos, futures, samples = _build_batch_inputs(
                ds, processor, args, batch_idx, planning_cams
            )
            new_tokens = _run_batch(
                model, processor, inputs, device, dtype, args.max_new_tokens
            )
            new_tokens_cpu = new_tokens.cpu().tolist()

            for j, i_local in enumerate(batch_idx):
                sample = samples[j]
                gt_wp = sample["_meta_waypoints"].cpu().numpy()
                valid = sample["_meta_valid_mask"].cpu().numpy()
                info = infos[j]
                future_infos = futures[j]

                # Stop at the first pad token so trailing pads don't confuse the
                # decoder. (Left-padding only adds pads on the left of the prompt
                # so this slice is right-side trailing pad from EOS-truncation.)
                ids = new_tokens_cpu[j]
                if processor.tokenizer.pad_token_id in ids:
                    cut = ids.index(processor.tokenizer.pad_token_id)
                    ids = ids[:cut]
                pred_wp = decode_waypoints(ids, traj_tok, args.num_future_waypoints)

                t = l2_temavg(pred_wp, gt_wp, valid)
                for k in local_temavg:
                    if not math.isnan(t[k]):
                        local_temavg[k].append(t[k])
                n = l2_noavg(pred_wp, gt_wp, valid)
                for k in local_noavg:
                    if not math.isnan(n[k]):
                        local_noavg[k].append(n[k])

                collisions_per_horizon = _uniad_compute_collision_per_sample(
                    pred_wp_ego=pred_wp,
                    gt_wp_ego=gt_wp,
                    future_infos=future_infos,
                    cur_info=info,
                    horizon_indices=HORIZON_IDX,
                )
                for hi, h_idx in enumerate(HORIZON_IDX):
                    if h_idx >= len(future_infos) or valid[h_idx] < 1e-6:
                        collisions_per_horizon[hi] = 0
                local_coll["collision_1s"].append(collisions_per_horizon[0])
                local_coll["collision_2s"].append(collisions_per_horizon[1])
                local_coll["collision_3s"].append(collisions_per_horizon[2])
                local_coll["collision_avg"].append(int(any(collisions_per_horizon)))

            done = bstart + len(batch_idx)
            if _is_rank0(rank) and (done % max(1, bs * 4) == 0 or done == n_local):
                rate_local = done / max(time.time() - t0, 1e-6)
                global_done = done * world_size
                global_total = n_total
                rate_global = rate_local * world_size
                eta = max(0.0, (global_total - global_done) / max(rate_global, 1e-6))
                print(
                    f"  [rank0 {done}/{n_local} | global {global_done}/{global_total}] "
                    f"{rate_local:.2f} sample/s (rank) | {rate_global:.2f} sample/s (global) "
                    f"| ETA {eta:.1f} s",
                    flush=True,
                )

    # Aggregate across ranks. Each rank packs its per-sample lists into a dict
    # and rank 0 gathers via `dist.gather_object`.
    local_payload = {
        "temavg": local_temavg,
        "noavg": local_noavg,
        "coll": {k: [int(x) for x in v] for k, v in local_coll.items()},
        "n_local": n_local,
    }

    if is_dist:
        gathered: List[Optional[dict]] = [None] * world_size if _is_rank0(rank) else None
        dist.gather_object(local_payload, gathered if _is_rank0(rank) else None, dst=0)
        dist.barrier()
    else:
        gathered = [local_payload]

    if not _is_rank0(rank):
        if is_dist:
            dist.destroy_process_group()
        return

    # Rank 0: merge per-sample lists from every rank.
    temavg_acc: Dict[str, List[float]] = {k: [] for k in ["L2_1s", "L2_2s", "L2_3s", "L2_avg"]}
    noavg_acc: Dict[str, List[float]] = {k: [] for k in ["L2_1s", "L2_2s", "L2_3s", "L2_avg"]}
    coll_acc: Dict[str, List[int]] = {k: [] for k in ["collision_1s", "collision_2s", "collision_3s", "collision_avg"]}
    for payload in gathered:
        if payload is None:
            continue
        for k, v in payload["temavg"].items():
            temavg_acc[k].extend(v)
        for k, v in payload["noavg"].items():
            noavg_acc[k].extend(v)
        for k, v in payload["coll"].items():
            coll_acc[k].extend(v)

    def _mean(xs) -> float:
        return float(np.mean(xs)) if len(xs) else float("nan")

    elapsed = time.time() - t0
    n_scored = len(temavg_acc["L2_avg"]) if temavg_acc["L2_avg"] else n_total
    results = {
        "ckpt": os.path.abspath(args.ckpt),
        "infos_val": os.path.abspath(args.infos_val),
        "n_samples": n_total,
        "n_scored": n_scored,
        "world_size": world_size,
        "batch_size": bs,
        "wall_seconds": round(elapsed, 2),
        "horizon_s": list(HORIZONS),
        "TemAvg": {k: _mean(v) for k, v in temavg_acc.items()},
        "NoAvg": {k: _mean(v) for k, v in noavg_acc.items()},
        "collision_rate": {
            "collision_1s": _mean([float(x) for x in coll_acc["collision_1s"]]),
            "collision_2s": _mean([float(x) for x in coll_acc["collision_2s"]]),
            "collision_3s": _mean([float(x) for x in coll_acc["collision_3s"]]),
            "collision_avg": _mean([float(x) for x in coll_acc["collision_avg"]]),
        },
        # Flat shortcut keys matching the table format requested in the spec.
        "L2_1s": _mean(temavg_acc["L2_1s"]),
        "L2_2s": _mean(temavg_acc["L2_2s"]),
        "L2_3s": _mean(temavg_acc["L2_3s"]),
        "L2_avg": _mean(temavg_acc["L2_avg"]),
        "collision_1s": _mean([float(x) for x in coll_acc["collision_1s"]]),
        "collision_2s": _mean([float(x) for x in coll_acc["collision_2s"]]),
        "collision_3s": _mean([float(x) for x in coll_acc["collision_3s"]]),
        "collision_avg": _mean([float(x) for x in coll_acc["collision_avg"]]),
        "protocol_l2": "TemAvg (VAD) shown in flat L2_*; full both protocols inside this JSON",
        "ego_footprint_m": {
            "length": EGO_LENGTH_M,
            "width": EGO_WIDTH_M,
            "half_length": EGO_HALF_LEN_M,
            "half_width": EGO_HALF_WID_M,
            "fwd_offset_from_pose": EGO_BOX_FWD_OFFSET_M,
        },
    }

    out_path = args.output or os.path.join(args.ckpt, "eval_results.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[planning_eval] wrote {out_path} (wall={elapsed:.1f}s, n_scored={n_scored})", flush=True)
    print(json.dumps(results, indent=2), flush=True)

    if is_dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
