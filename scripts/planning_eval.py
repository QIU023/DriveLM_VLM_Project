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
from types import SimpleNamespace
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


# ============================================================================
# External projector load (qformer / pixelshuffle / resampler)
# ============================================================================

def _maybe_load_external_projector(ckpt_dir: str, device: torch.device,
                                   dtype: torch.dtype):
    """If ``<ckpt_dir>/projector_meta.json`` exists, instantiate the projector
    class for the recorded ``type`` with the recorded ``config`` and load
    weights from ``<ckpt_dir>/projector.pt``.

    Returns ``(projector, projector_type)`` or ``(None, None)`` if no meta
    file is present (legacy / linear-baseline ckpts — R1' compatible).
    """
    meta_path = os.path.join(ckpt_dir, "projector_meta.json")
    weights_path = os.path.join(ckpt_dir, "projector.pt")
    if not os.path.exists(meta_path):
        return None, None
    with open(meta_path) as f:
        meta = json.load(f)
    p_type = meta["type"].lower()
    p_cfg = meta["config"]
    if p_type == "qformer":
        # Lazy import — matches the train_lora.py import pattern.
        try:
            from scripts.qformer_projector_hf import (  # noqa: E402
                Qwen2VLQFormerProjector,
            )
        except ImportError:
            from qformer_projector_hf import (  # type: ignore  # noqa: E402
                Qwen2VLQFormerProjector,
            )
        projector = Qwen2VLQFormerProjector(**p_cfg)
    elif p_type == "pixelshuffle":
        # Lazy import — matches the train_lora.py import pattern.
        try:
            from scripts.pixelshuffle_projector_hf import (  # noqa: E402
                Qwen2VLPixelShufflePlusLinearProjector,
            )
        except ImportError:
            from pixelshuffle_projector_hf import (  # type: ignore  # noqa: E402
                Qwen2VLPixelShufflePlusLinearProjector,
            )
        projector = Qwen2VLPixelShufflePlusLinearProjector(**p_cfg)
    else:
        # Resampler (A.3) lands here as another elif branch when it saves
        # its projector via _save_external_projector.
        raise NotImplementedError(
            f"planning_eval: load path for projector_type={p_type!r} not "
            f"wired yet. Add an import + instantiate branch here."
        )
    state = torch.load(weights_path, map_location="cpu")
    projector.load_state_dict(state)
    projector = projector.to(device=device, dtype=dtype)
    projector.eval()
    return projector, p_type


def _trim_and_pad_for_projector(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    video_grid_thw: torch.Tensor,
    video_token_id: int,
    projector,
    merge_size: int = 2,
):
    """Trim per-sample video-pad placeholders so the LM sees exactly
    ``projector.num_queries`` placeholders per video item. Returns
    ``(new_input_ids, new_attention_mask, new_video_grid_thw, n_post_total,
       (t0, h0, w0), num_items)`` for use by the patched
    ``get_video_features``.

    Mirrors train_lora.forward_with_video_qformer_projector's trim logic
    (steps 1 + 2) but without label handling (generate has no labels).
    """
    device = input_ids.device
    grid = video_grid_thw
    if grid.dim() == 1:
        grid = grid.unsqueeze(0)
    num_items = grid.shape[0]

    t_per_item = grid[:, 0].tolist()
    h_post = (grid[:, 1] // merge_size).tolist()
    w_post = (grid[:, 2] // merge_size).tolist()
    n_post_per_item = [t_per_item[i] * h_post[i] * w_post[i] for i in range(num_items)]
    n_post_total = sum(n_post_per_item)

    first_thw = (t_per_item[0], h_post[0], w_post[0])
    for i in range(num_items):
        if (t_per_item[i], h_post[i], w_post[i]) != first_thw:
            raise RuntimeError(
                f"Q-Former projector requires identical post-merger "
                f"(t, h, w) across all items at eval; item {i}="
                f"{(t_per_item[i], h_post[i], w_post[i])} != "
                f"item 0={first_thw}."
            )
    t0, h0, w0 = first_thw

    B_lm = input_ids.shape[0]
    items_per_sample = num_items // B_lm
    if items_per_sample * B_lm != num_items:
        raise RuntimeError(
            f"video_grid_thw num_items={num_items} not divisible by LM "
            f"batch size B_lm={B_lm}."
        )
    n_per_item = n_post_per_item[0]
    n_compressed_per_item = int(projector.num_queries)

    new_ids_list, new_mask_list = [], []
    for b in range(B_lm):
        ids = input_ids[b]
        msk = attention_mask[b]
        vid_pos = (ids == video_token_id).nonzero(as_tuple=True)[0]
        n_vid = len(vid_pos)
        if n_vid == 0:
            new_ids_list.append(ids)
            new_mask_list.append(msk)
            continue
        expected_uncompressed = items_per_sample * n_per_item
        if n_vid != expected_uncompressed:
            raise RuntimeError(
                f"sample {b}: found {n_vid} video-pad tokens but expected "
                f"{expected_uncompressed} ({items_per_sample} items x "
                f"{n_per_item} post-merger tokens)."
            )
        drop_positions = []
        for k in range(items_per_sample):
            item_start = k * n_per_item
            item_end = item_start + n_per_item
            keep_until = item_start + n_compressed_per_item
            drop_positions.extend(vid_pos[keep_until:item_end].tolist())
        if drop_positions:
            keep = torch.ones(len(ids), dtype=torch.bool, device=device)
            keep[torch.tensor(drop_positions, device=device)] = False
            new_ids_list.append(ids[keep])
            new_mask_list.append(msk[keep])
        else:
            new_ids_list.append(ids)
            new_mask_list.append(msk)

    # Left-pad to common length (planning_eval uses left-padding for
    # batched greedy generate; preserve that convention).
    max_len = max(t.shape[0] for t in new_ids_list)
    pad_id = 0  # generate is told pad_token_id below; the dummy 0 here is masked.
    for i in range(B_lm):
        pad = max_len - new_ids_list[i].shape[0]
        if pad > 0:
            new_ids_list[i] = torch.cat([
                torch.full((pad,), pad_id, dtype=new_ids_list[i].dtype, device=device),
                new_ids_list[i],
            ])
            new_mask_list[i] = torch.cat([
                torch.zeros(pad, dtype=new_mask_list[i].dtype, device=device),
                new_mask_list[i],
            ])
    new_input_ids = torch.stack(new_ids_list)
    new_attn_mask = torch.stack(new_mask_list)

    # Rebuild video_grid_thw to the compressed shape (1, h_pre, w_pre) with
    # h_post * w_post == num_queries. Inline the factor helper from
    # train_lora._factor_grid_thw_for_count to avoid the cross-import.
    def _factor_grid(target: int, ms: int) -> "tuple[int, int, int]":
        best = None
        for h in range(1, int(target ** 0.5) + 1):
            if target % h == 0:
                w = target // h
                ar = max(h, w) / min(h, w)
                if best is None or ar < best[0]:
                    best = (ar, h, w)
        if best is None:
            return (1, 1 * ms, target * ms)
        _, h, w = best
        return (1, h * ms, w * ms)

    _, h_pre_new, w_pre_new = _factor_grid(n_compressed_per_item, merge_size)
    new_grid = torch.tensor(
        [[1, h_pre_new, w_pre_new]] * num_items,
        dtype=grid.dtype, device=device,
    )
    return new_input_ids, new_attn_mask, new_grid, n_post_total, (t0, h0, w0), num_items


def _trim_and_pad_for_pixelshuffle_projector(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    video_grid_thw: torch.Tensor,
    video_token_id: int,
    projector,
    merge_size: int = 2,
):
    """Same structure as ``_trim_and_pad_for_projector`` (qformer) but the
    per-item compressed token count comes from
    ``projector.output_token_count(n_per_item)`` rather than a fixed
    ``num_queries`` — PixelShuffle's output is ``n_per_item / shuffle_ratio**2``
    and depends on the input grid (not a constant). Mirrors
    ``train_lora.forward_with_video_pixelshuffle_projector``'s trim logic.
    """
    device = input_ids.device
    grid = video_grid_thw
    if grid.dim() == 1:
        grid = grid.unsqueeze(0)
    num_items = grid.shape[0]

    t_per_item = grid[:, 0].tolist()
    h_post = (grid[:, 1] // merge_size).tolist()
    w_post = (grid[:, 2] // merge_size).tolist()
    n_post_per_item = [t_per_item[i] * h_post[i] * w_post[i] for i in range(num_items)]
    n_post_total = sum(n_post_per_item)

    first_thw = (t_per_item[0], h_post[0], w_post[0])
    for i in range(num_items):
        if (t_per_item[i], h_post[i], w_post[i]) != first_thw:
            raise RuntimeError(
                f"PixelShuffle projector requires identical post-merger "
                f"(t, h, w) across all items at eval; item {i}="
                f"{(t_per_item[i], h_post[i], w_post[i])} != "
                f"item 0={first_thw}."
            )
    t0, h0, w0 = first_thw
    r = int(projector.shuffle_ratio)
    if h0 % r != 0 or w0 % r != 0:
        raise RuntimeError(
            f"PixelShuffle projector with shuffle_ratio={r} requires "
            f"post-merger h ({h0}) and w ({w0}) to be divisible by {r}."
        )

    B_lm = input_ids.shape[0]
    items_per_sample = num_items // B_lm
    if items_per_sample * B_lm != num_items:
        raise RuntimeError(
            f"video_grid_thw num_items={num_items} not divisible by LM "
            f"batch size B_lm={B_lm}."
        )
    n_per_item = n_post_per_item[0]
    n_compressed_per_item = int(projector.output_token_count(n_per_item))

    new_ids_list, new_mask_list = [], []
    for b in range(B_lm):
        ids = input_ids[b]
        msk = attention_mask[b]
        vid_pos = (ids == video_token_id).nonzero(as_tuple=True)[0]
        n_vid = len(vid_pos)
        if n_vid == 0:
            new_ids_list.append(ids)
            new_mask_list.append(msk)
            continue
        expected_uncompressed = items_per_sample * n_per_item
        if n_vid != expected_uncompressed:
            raise RuntimeError(
                f"sample {b}: found {n_vid} video-pad tokens but expected "
                f"{expected_uncompressed} ({items_per_sample} items x "
                f"{n_per_item} post-merger tokens)."
            )
        drop_positions = []
        for k in range(items_per_sample):
            item_start = k * n_per_item
            item_end = item_start + n_per_item
            keep_until = item_start + n_compressed_per_item
            drop_positions.extend(vid_pos[keep_until:item_end].tolist())
        if drop_positions:
            keep = torch.ones(len(ids), dtype=torch.bool, device=device)
            keep[torch.tensor(drop_positions, device=device)] = False
            new_ids_list.append(ids[keep])
            new_mask_list.append(msk[keep])
        else:
            new_ids_list.append(ids)
            new_mask_list.append(msk)

    # Left-pad to common length.
    max_len = max(t.shape[0] for t in new_ids_list)
    pad_id = 0
    for i in range(B_lm):
        pad = max_len - new_ids_list[i].shape[0]
        if pad > 0:
            new_ids_list[i] = torch.cat([
                torch.full((pad,), pad_id, dtype=new_ids_list[i].dtype, device=device),
                new_ids_list[i],
            ])
            new_mask_list[i] = torch.cat([
                torch.zeros(pad, dtype=new_mask_list[i].dtype, device=device),
                new_mask_list[i],
            ])
    new_input_ids = torch.stack(new_ids_list)
    new_attn_mask = torch.stack(new_mask_list)

    # Rebuild video_grid_thw to the compressed shape (1, h_pre, w_pre) with
    # h_post * w_post == n_compressed_per_item. Inline factor helper.
    def _factor_grid(target: int, ms: int) -> "tuple[int, int, int]":
        best = None
        for h in range(1, int(target ** 0.5) + 1):
            if target % h == 0:
                w = target // h
                ar = max(h, w) / min(h, w)
                if best is None or ar < best[0]:
                    best = (ar, h, w)
        if best is None:
            return (1, 1 * ms, target * ms)
        _, h, w = best
        return (1, h * ms, w * ms)

    _, h_pre_new, w_pre_new = _factor_grid(n_compressed_per_item, merge_size)
    new_grid = torch.tensor(
        [[1, h_pre_new, w_pre_new]] * num_items,
        dtype=grid.dtype, device=device,
    )
    return new_input_ids, new_attn_mask, new_grid, n_post_total, (t0, h0, w0), num_items


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
    *,
    external_projector=None,
    projector_type: Optional[str] = None,
    video_token_id: Optional[int] = None,
):
    """Move inputs to device, run greedy generate, return new tokens.

    When ``external_projector`` is given (qformer / pixelshuffle / resampler),
    we mirror the training-side forward shim
    (``forward_with_video_qformer_projector`` in train_lora.py):

      1. Trim ``input_ids`` / ``attention_mask`` to keep only
         ``projector.num_queries`` ``<|video_pad|>`` placeholders per item.
      2. Rebuild ``video_grid_thw`` to the compressed shape.
      3. Monkey-patch ``inner.get_video_features`` to run the vision tower
         and then the projector, returning ``num_queries * num_items`` rows
         (a `_FakeVisOut.pooler_output` carrying a list of (Nq, lm_dim)).
      4. Call ``model.generate(**trimmed_inputs)`` — the generate path
         calls ``get_video_features`` once during the prefill, so the
         projector runs on the raw vision features and the LM sees the
         compressed tokens for the rest of the decode loop.

    Note ``inputs_embeds`` is NOT manually built here — the standard HF
    Qwen2.5-VL generate path takes care of scattering the (now-projector-
    produced) visual features into the embeddings at the placeholder
    positions, as long as the placeholder count matches the projector's
    output count (which it does after the trim above).
    """
    inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    if "pixel_values_videos" in inputs:
        inputs["pixel_values_videos"] = inputs["pixel_values_videos"].to(dtype)
    if "pixel_values" in inputs and isinstance(inputs["pixel_values"], torch.Tensor):
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype)

    if external_projector is not None:
        if video_token_id is None:
            raise ValueError("video_token_id is required when external_projector is set")
        if projector_type not in ("qformer", "pixelshuffle"):
            # Resampler (A.3) will land here as another branch once its
            # training-side shim exists.
            raise NotImplementedError(
                f"planning_eval generate path for projector_type="
                f"{projector_type!r} not implemented yet."
            )
        if projector_type == "qformer":
            new_input_ids, new_attn_mask, new_grid, n_post_total, (t0, h0, w0), num_items = (
                _trim_and_pad_for_projector(
                    inputs["input_ids"],
                    inputs["attention_mask"],
                    inputs["video_grid_thw"],
                    video_token_id,
                    external_projector,
                )
            )
        else:  # pixelshuffle
            new_input_ids, new_attn_mask, new_grid, n_post_total, (t0, h0, w0), num_items = (
                _trim_and_pad_for_pixelshuffle_projector(
                    inputs["input_ids"],
                    inputs["attention_mask"],
                    inputs["video_grid_thw"],
                    video_token_id,
                    external_projector,
                )
            )
        pv = inputs["pixel_values_videos"]
        orig_grid = inputs["video_grid_thw"]

        # Unwrap PEFT/etc if present. AutoModelForImageTextToText returns
        # the top-level Qwen2_5_VL model; the inner Qwen2_5_VLModel that
        # owns get_video_features is at .model on that.
        inner = model.model

        _orig_get_video_features = inner.get_video_features

        class _FakeVisOut:
            def __init__(self, t):
                self.pooler_output = t

        # PixelShuffle needs grid_thw_post per-item; precompute once outside
        # the patched closure (all items share shape, asserted in
        # _trim_and_pad_for_pixelshuffle_projector).
        _grid_thw_post_row = None
        if projector_type == "pixelshuffle":
            _grid_thw_post_row = torch.tensor(
                [[t0, h0, w0]], dtype=torch.long, device=inputs["input_ids"].device,
            )

        def _patched_get_video_features(_pv, _grid):  # noqa: ARG001
            with torch.no_grad():
                real = _orig_get_video_features(pv, orig_grid)
                embeds = real.pooler_output
                if isinstance(embeds, (tuple, list)):
                    embeds = torch.cat([e for e in embeds], dim=0)
                embeds = embeds.detach()
            if embeds.shape[0] != n_post_total:
                raise RuntimeError(
                    f"vision pooler_output rows {embeds.shape[0]} != expected "
                    f"post-merger total {n_post_total}"
                )
            D = embeds.shape[-1]
            per_item = embeds.view(num_items, t0 * h0 * w0, D)
            proj_dtype = next(external_projector.parameters()).dtype
            compressed_items = []
            for i in range(num_items):
                if projector_type == "pixelshuffle":
                    out_i = external_projector(
                        per_item[i:i+1].to(proj_dtype),
                        grid_thw_post=_grid_thw_post_row,
                    )
                else:
                    out_i = external_projector(per_item[i:i+1].to(proj_dtype))
                compressed_items.append(out_i.squeeze(0))
            return _FakeVisOut(compressed_items)

        inner.get_video_features = _patched_get_video_features
        try:
            prompt_len = new_input_ids.shape[1]
            gen = model.generate(
                input_ids=new_input_ids,
                attention_mask=new_attn_mask,
                pixel_values_videos=pv,
                video_grid_thw=new_grid,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=processor.tokenizer.pad_token_id or 0,
                use_cache=True,
            )
        finally:
            inner.get_video_features = _orig_get_video_features
        new_tokens = gen[:, prompt_len:]
        return new_tokens

    # Vanilla / linear-baseline path (R1' byte-compatible).
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
# Reusable evaluation function (for mid-training validate + standalone main)
# ============================================================================

def evaluate_planning_l2_collision(
    model,
    processor,
    val_dataset: PlanningDataset,
    accelerator,
    *,
    external_projector=None,
    projector_type: Optional[str] = None,
    batch_size: int = 4,
    num_samples: Optional[int] = None,
    max_new_tokens: int = 20,
    video_fps: float = 2.0,
    planning_cams: Optional[List[str]] = None,
    silent: bool = False,
) -> Dict[str, float]:
    """Run DP greedy-decode L2 + UniAD-port collision eval on ``val_dataset``.

    Reuses the EXISTING accelerator (does NOT call init_process_group). The
    val samples are stride-sharded across ``accelerator.num_processes`` ranks
    (i, i+W, i+2W, ...); each rank greedy-decodes its shard, then per-sample
    metric lists are gathered to every rank via ``gather_object`` and merged.

    When ``external_projector`` is provided (``projector_type='qformer'``),
    the generate path mirrors the training-time forward shim
    (forward_with_video_qformer_projector): placeholder trim + grid rebuild +
    monkey-patch ``get_video_features`` so the projector runs during the
    prefill. ``pixelshuffle`` / ``resampler`` are not yet wired — passing
    those raises ``NotImplementedError``.

    When ``external_projector is None`` (R1' linear baseline path), we fall
    back to the vanilla ``model.generate(...)`` flow (no monkey-patch).

    Parameters
    ----------
    model : the (already prepared) HF model on the right device.
    processor : the matching AutoProcessor; ``tokenizer.padding_side`` is
        FORCED to ``"left"`` inside this function (batched greedy generate
        requires it). We do NOT restore the prior value, since the LM
        forward path doesn't depend on it.
    val_dataset : a PlanningDataset instance (multi-cam aware).
    accelerator : the ``accelerate.Accelerator`` that owns the model. We
        read ``.device``, ``.num_processes``, ``.process_index``,
        ``.is_main_process`` and use ``gather_object`` for cross-rank merge.
    external_projector : optional qformer/pixelshuffle/resampler module.
    projector_type : 'qformer' (only one wired) | 'pixelshuffle' | 'resampler'.
    batch_size : per-rank generate batch (default 4).
    num_samples : cap eval to first N samples (None = all). The cap is
        applied to the GLOBAL index list BEFORE stride-sharding so each
        rank receives the same N / world_size shard size.
    max_new_tokens : greedy decode budget (default 20: 1 start + 12 bins
        + 1 end + slack).
    video_fps : passed into VideoMetadata (default 2.0).
    planning_cams : optional override; defaults to ``val_dataset.planning_cams``.
    silent : if True, suppress per-batch progress prints (validate path
        typically wants this).

    Returns
    -------
    dict with at minimum these keys (every rank returns the same dict — the
    gather + merge happens on every rank so it's safe to read on rank>0):

        L2_avg, L2_1s, L2_2s, L2_3s           # TemAvg (VAD) protocol
        noavg_L2_avg, noavg_L2_1s, ...        # NoAvg  (UniAD) protocol
        collision_avg, collision_1s, ...      # UniAD-port collision (fractions)
        n_scored                              # number of samples actually scored
        wall_seconds                          # wall-clock of the full eval
        protocol_l2                           # protocol marker string

    Returns NaN for any horizon with no valid samples.

    FSDP contract
    -------------
    This function calls ``model.generate(...)`` directly. If the model was
    wrapped by ``FullyShardedDataParallel`` (FULL_SHARD) the per-rank
    parameters are 1-D FlatParameters and the inner ``nn.Embedding`` /
    ``nn.Linear`` calls will raise ``RuntimeError: 'weight' must be 2-D``.
    Callers (e.g. ``train_lora.validate()``) MUST wrap the call in
    ``FSDP.summon_full_params(model, writeback=False, recurse=True)`` so the
    full unsharded weights are materialised for the eval. The standalone
    post-training ``main()`` path loads from disk via
    ``AutoModel.from_pretrained(...)`` and is never FSDP-wrapped, so it does
    not need this. We intentionally do NOT auto-summon inside this function
    to keep the standalone path free of FSDP-import side-effects.
    """
    device = accelerator.device
    rank = int(accelerator.process_index)
    world_size = int(accelerator.num_processes)
    is_dist = world_size > 1

    # Fail-fast on unsupported projector types BEFORE any data work or model
    # touches — gives callers a clean NotImplementedError they can catch.
    if external_projector is not None and projector_type and projector_type not in ("qformer", "pixelshuffle"):
        raise NotImplementedError(
            f"planning_eval generate path for {projector_type!r} not yet wired"
        )

    # Early-out on empty datasets BEFORE touching the model / processor, so a
    # caller that wants to probe the return-dict shape can pass None for both.
    n_total = len(val_dataset)
    if num_samples is not None:
        n_total = min(n_total, int(num_samples))
    if n_total == 0:
        return {
            "L2_avg": float("nan"), "L2_1s": float("nan"),
            "L2_2s": float("nan"), "L2_3s": float("nan"),
            "noavg_L2_avg": float("nan"), "noavg_L2_1s": float("nan"),
            "noavg_L2_2s": float("nan"), "noavg_L2_3s": float("nan"),
            "collision_avg": float("nan"), "collision_1s": float("nan"),
            "collision_2s": float("nan"), "collision_3s": float("nan"),
            "n_scored": 0, "wall_seconds": 0.0,
            "protocol_l2": "TemAvg (VAD)",
        }

    # Honour the trainer-style dtype.
    dtype = next(model.parameters()).dtype

    # Batched generate requires left-padding so new tokens start at the same
    # column for every row.
    processor.tokenizer.padding_side = "left"

    if planning_cams is None:
        planning_cams = list(getattr(val_dataset, "planning_cams", DEFAULT_PLANNING_CAMS))

    traj_cfg = TrajectoryTokenizerConfig(num_waypoints=val_dataset.num_future)
    traj_tok = TrajectoryTokenizer(traj_cfg)

    # Standalone main() uses an argparse.Namespace as the "args" carrier; the
    # batch-builder only reads ``args.video_fps``. Reuse that pattern with a
    # lightweight stand-in so we don't have to refactor _build_batch_inputs.
    _args_shim = SimpleNamespace(video_fps=float(video_fps))

    # Stride-shard across ranks (same convention as main()).
    shard_indices: List[int] = list(range(rank, n_total, world_size))

    local_temavg: Dict[str, List[float]] = {k: [] for k in ["L2_1s", "L2_2s", "L2_3s", "L2_avg"]}
    local_noavg: Dict[str, List[float]] = {k: [] for k in ["L2_1s", "L2_2s", "L2_3s", "L2_avg"]}
    local_coll: Dict[str, List[int]] = {k: [] for k in ["collision_1s", "collision_2s", "collision_3s", "collision_avg"]}

    # video_token_id is only needed on the projector path. Resolve once.
    # (Projector-type validity was already checked at function entry.)
    if external_projector is not None:
        video_token_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
    else:
        video_token_id = None

    t0 = time.time()
    bs = max(1, int(batch_size))
    n_local = len(shard_indices)

    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for bstart in range(0, n_local, bs):
                batch_idx = shard_indices[bstart:bstart + bs]
                inputs, infos, futures, samples = _build_batch_inputs(
                    val_dataset, processor, _args_shim, batch_idx, planning_cams
                )
                new_tokens = _run_batch(
                    model, processor, inputs, device, dtype, max_new_tokens,
                    external_projector=external_projector,
                    projector_type=projector_type,
                    video_token_id=video_token_id,
                )
                new_tokens_cpu = new_tokens.cpu().tolist()

                for j, _i_local in enumerate(batch_idx):
                    sample = samples[j]
                    gt_wp = sample["_meta_waypoints"].cpu().numpy()
                    valid = sample["_meta_valid_mask"].cpu().numpy()
                    info = infos[j]
                    future_infos = futures[j]

                    ids = new_tokens_cpu[j]
                    if processor.tokenizer.pad_token_id in ids:
                        cut = ids.index(processor.tokenizer.pad_token_id)
                        ids = ids[:cut]
                    pred_wp = decode_waypoints(ids, traj_tok, val_dataset.num_future)

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

                if not silent and accelerator.is_main_process:
                    done = bstart + len(batch_idx)
                    if done % max(1, bs * 4) == 0 or done == n_local:
                        rate_local = done / max(time.time() - t0, 1e-6)
                        global_done = done * world_size
                        rate_global = rate_local * world_size
                        eta = max(0.0, (n_total - global_done) / max(rate_global, 1e-6))
                        print(
                            f"  [planning_eval] rank0 {done}/{n_local} | "
                            f"global {global_done}/{n_total} | "
                            f"{rate_global:.2f} sample/s (global) | ETA {eta:.1f} s",
                            flush=True,
                        )
    finally:
        if was_training:
            model.train()

    local_payload = {
        "temavg": local_temavg,
        "noavg": local_noavg,
        "coll": {k: [int(x) for x in v] for k, v in local_coll.items()},
        "n_local": n_local,
    }

    # All-gather so EVERY rank gets the merged result (the validate caller
    # may want to print on rank 0 but compute on all ranks).
    if is_dist and dist.is_available() and dist.is_initialized():
        bucket: List[Optional[dict]] = [None] * world_size
        dist.all_gather_object(bucket, local_payload)
        gathered = bucket
    else:
        gathered = [local_payload]

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

    return {
        # TemAvg (VAD) — flat keys (paper-comparable).
        "L2_avg": _mean(temavg_acc["L2_avg"]),
        "L2_1s": _mean(temavg_acc["L2_1s"]),
        "L2_2s": _mean(temavg_acc["L2_2s"]),
        "L2_3s": _mean(temavg_acc["L2_3s"]),
        # NoAvg (UniAD).
        "noavg_L2_avg": _mean(noavg_acc["L2_avg"]),
        "noavg_L2_1s": _mean(noavg_acc["L2_1s"]),
        "noavg_L2_2s": _mean(noavg_acc["L2_2s"]),
        "noavg_L2_3s": _mean(noavg_acc["L2_3s"]),
        # UniAD-port collision (fractions in [0, 1]).
        "collision_avg": _mean([float(x) for x in coll_acc["collision_avg"]]),
        "collision_1s": _mean([float(x) for x in coll_acc["collision_1s"]]),
        "collision_2s": _mean([float(x) for x in coll_acc["collision_2s"]]),
        "collision_3s": _mean([float(x) for x in coll_acc["collision_3s"]]),
        "n_scored": int(n_scored),
        "wall_seconds": round(elapsed, 2),
        "protocol_l2": "TemAvg (VAD) shown in L2_*; NoAvg under noavg_L2_*",
    }


# ============================================================================
# Main
# ============================================================================

class _StandaloneAcceleratorShim:
    """Minimal Accelerator-API stand-in for the standalone CLI path.

    ``evaluate_planning_l2_collision`` only reads ``.device``,
    ``.process_index``, ``.num_processes`` and ``.is_main_process``, so this
    is enough to reuse the same function without pulling in the full
    ``accelerate`` package on the eval-from-shell path. We don't initialise
    a process group here — ``_init_distributed`` (called by main) already
    handled that.
    """

    def __init__(self, device, rank: int, world_size: int):
        self.device = device
        self.process_index = rank
        self.num_processes = world_size
        self.is_main_process = (rank == 0)


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

    # ---- Optional external projector (qformer / pixelshuffle / resampler) ---
    # Detect via <ckpt>/projector_meta.json. Absent -> vanilla / linear path
    # (R1' byte-compatible). Present -> load + wire into _run_batch.
    external_projector, projector_type = _maybe_load_external_projector(
        args.ckpt, device, dtype,
    )
    if external_projector is not None:
        _log(rank, f"[planning_eval] external projector loaded: "
                   f"type={projector_type} "
                   f"params={sum(p.numel() for p in external_projector.parameters())/1e6:.2f}M")

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

    # Thin wrapper: hand off to evaluate_planning_l2_collision (the same
    # function used by validate() in train_lora.py). _StandaloneAcceleratorShim
    # provides the four attrs the function reads from accelerator.
    acc_shim = _StandaloneAcceleratorShim(device=device, rank=rank, world_size=world_size)
    bs = max(1, int(args.batch_size))
    metrics = evaluate_planning_l2_collision(
        model, processor, ds, acc_shim,
        external_projector=external_projector,
        projector_type=projector_type,
        batch_size=bs,
        num_samples=None,  # standalone always walks the full ds (already capped via max_samples)
        max_new_tokens=args.max_new_tokens,
        video_fps=args.video_fps,
        planning_cams=planning_cams,
        silent=False,
    )

    if not _is_rank0(rank):
        if is_dist:
            dist.destroy_process_group()
        return

    # Rank 0: assemble the legacy JSON schema (TemAvg/NoAvg/collision_rate
    # nested dicts + flat shortcut keys + ego_footprint_m) and write it.
    elapsed = metrics["wall_seconds"]
    n_scored = metrics["n_scored"]
    results = {
        "ckpt": os.path.abspath(args.ckpt),
        "infos_val": os.path.abspath(args.infos_val),
        "n_samples": n_total,
        "n_scored": n_scored,
        "world_size": world_size,
        "batch_size": bs,
        "wall_seconds": elapsed,
        "horizon_s": list(HORIZONS),
        "TemAvg": {
            "L2_1s": metrics["L2_1s"], "L2_2s": metrics["L2_2s"],
            "L2_3s": metrics["L2_3s"], "L2_avg": metrics["L2_avg"],
        },
        "NoAvg": {
            "L2_1s": metrics["noavg_L2_1s"], "L2_2s": metrics["noavg_L2_2s"],
            "L2_3s": metrics["noavg_L2_3s"], "L2_avg": metrics["noavg_L2_avg"],
        },
        "collision_rate": {
            "collision_1s": metrics["collision_1s"],
            "collision_2s": metrics["collision_2s"],
            "collision_3s": metrics["collision_3s"],
            "collision_avg": metrics["collision_avg"],
        },
        # Flat shortcut keys (table-format).
        "L2_1s": metrics["L2_1s"],
        "L2_2s": metrics["L2_2s"],
        "L2_3s": metrics["L2_3s"],
        "L2_avg": metrics["L2_avg"],
        "collision_1s": metrics["collision_1s"],
        "collision_2s": metrics["collision_2s"],
        "collision_3s": metrics["collision_3s"],
        "collision_avg": metrics["collision_avg"],
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
