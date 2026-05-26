"""Shared layout + vision-feature helpers for the Tier-2 NATIVE 3-cam cache.

This module is the SINGLE SOURCE OF TRUTH for the token layout used by BOTH:
  * the LIVE compressed forward (``forward_with_video_compression_free`` in
    ``scripts/train_lora.py``), and
  * the cached forward (``forward_with_cached_vision_tokens``), and
  * the cache producer (``cache_3cam_native.py``).

Keeping the trim/rebuild logic in one place is what makes the HARD PARITY GATE
hold: the cached path replays the EXACT same ``input_ids`` / ``labels`` /
``video_grid_thw`` trim that the live path computes, so the only numerical
difference between A (live) and B (cached) is the int8 round-trip on the vision
tokens.

Qwen3-VL specifics handled here (which the original repo hook did NOT):
  * deepstack features: ``get_video_features`` / ``get_image_features`` return
    ``deepstack_features`` (a list of per-layer (N, D) tensors injected at deep
    decoder layers via ``visual_pos_masks``). The video deepstack is the
    CONCATENATION of all video items (8400 = 3*2800 for 3-cam native). When we
    compress the video pooler with FasterVLM (top-K by L2 norm), we apply the
    SAME selection indices to every deepstack layer so deepstack stays aligned
    with the surviving video-pad positions (2100 = 3*700).
  * the HD-map image branch (121 image-pad tokens) is kept un-compressed; both
    its pooler and deepstack features are run/cached verbatim.
  * ``mm_token_type_ids`` is trimmed in lockstep with ``input_ids`` (required by
    Qwen3-VL M-RoPE; the model raises if grids are passed without it).
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SCRIPTS = os.path.join(_REPO, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

MERGE_SIZE = 2
COMPRESS_METHOD = "fastervlm"
COMPRESS_RATIO = 4


# ----------------------------------------------------------------------------
# int8 quantization (symmetric per-tensor) — identical to P0 cache_vision_tokens
# ----------------------------------------------------------------------------
def quantize_int8(x: np.ndarray) -> Tuple[bytes, float, List[int]]:
    x = np.ascontiguousarray(x, dtype=np.float32)
    amax = float(np.abs(x).max())
    scale = amax / 127.0 if amax > 0 else 1.0
    q = np.round(x / scale).clip(-127, 127).astype(np.int8)
    return q.tobytes(), scale, list(x.shape)


def dequantize_int8(buf: bytes, scale: float, shape: List[int]) -> np.ndarray:
    return np.frombuffer(buf, dtype=np.int8).reshape(shape).astype(np.float32) * scale


# ----------------------------------------------------------------------------
# Layout helpers (mirror planning_eval_compress / train_lora)
# ----------------------------------------------------------------------------
def per_item_post_counts(video_grid_thw: torch.Tensor, merge_size: int = MERGE_SIZE) -> List[int]:
    g = video_grid_thw
    if g.dim() == 1:
        g = g.unsqueeze(0)
    return [int(g[i, 0]) * (int(g[i, 1]) // merge_size) * (int(g[i, 2]) // merge_size)
            for i in range(g.shape[0])]


def factor_grid_thw_for_count(target: int, merge_size: int = MERGE_SIZE) -> Tuple[int, int, int]:
    best = None
    for h in range(1, int(target ** 0.5) + 1):
        if target % h == 0:
            w = target // h
            ar = max(h, w) / min(h, w)
            if best is None or ar < best[0]:
                best = (ar, h, w)
    if best is None:
        return (1, 1 * merge_size, target * merge_size)
    _, h, w = best
    return (1, h * merge_size, w * merge_size)


def fastervlm_indices(tokens: torch.Tensor, ratio: int) -> torch.Tensor:
    """Return the sorted indices kept by FasterVLM (top-K by L2 norm).

    Matches ``visual_compress._fastervlm`` exactly so the kept pooler tokens are
    identical to ``compress_visual_tokens(...)`` output, and the SAME indices can
    be applied to deepstack layers.
    """
    n = tokens.shape[0]
    k = max(1, n // ratio)
    _, idx = tokens.norm(dim=-1).topk(k)
    return idx.sort().values


def trim_native_layout(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    mm_token_type_ids: torch.Tensor,
    video_grid_thw: torch.Tensor,
    video_token_id: int,
    *,
    compress_ratio: int = COMPRESS_RATIO,
    merge_size: int = MERGE_SIZE,
) -> Dict[str, object]:
    """Trim each video item's ``<|video_pad|>`` run down to its compressed count.

    Single-sample (un-batched) version: all tensors are 1-D (seq,) except
    ``video_grid_thw`` which is (num_items, 3). Returns trimmed tensors plus the
    rebuilt ``video_grid_thw`` and the per-item kept-index lists (so the caller
    can subselect the cached vision tokens with the SAME FasterVLM selection).

    This is the lockstep trim used by the live path; we keep it identical so
    parity holds.
    """
    device = input_ids.device
    grid = video_grid_thw
    if grid.dim() == 1:
        grid = grid.unsqueeze(0)
    per_item_orig = per_item_post_counts(grid, merge_size)
    per_item_comp = [max(1, n // int(compress_ratio)) for n in per_item_orig]

    vid_pos = (input_ids == video_token_id).nonzero(as_tuple=True)[0]
    expected_total = sum(per_item_orig)
    if len(vid_pos) != expected_total:
        raise RuntimeError(
            f"found {len(vid_pos)} <|video_pad|> tokens but grid says {expected_total}"
        )

    flat_pos_list = vid_pos.tolist()
    drop_positions: List[int] = []
    flat_idx = 0
    for orig_n, comp_n in zip(per_item_orig, per_item_comp):
        item_positions = flat_pos_list[flat_idx: flat_idx + orig_n]
        if comp_n < orig_n:
            drop_positions.extend(item_positions[comp_n:])
        flat_idx += orig_n

    if drop_positions:
        keep = torch.ones(len(input_ids), dtype=torch.bool, device=device)
        keep[torch.tensor(drop_positions, device=device)] = False
        input_ids = input_ids[keep]
        attention_mask = attention_mask[keep]
        labels = labels[keep]
        if mm_token_type_ids is not None:
            mm_token_type_ids = mm_token_type_ids[keep]

    new_rows = [list(factor_grid_thw_for_count(per_item_comp[k], merge_size))
                for k in range(len(per_item_comp))]
    new_grid_thw = torch.tensor(new_rows, dtype=grid.dtype, device=device)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "mm_token_type_ids": mm_token_type_ids,
        "video_grid_thw": new_grid_thw,
        "per_item_orig": per_item_orig,
        "per_item_comp": per_item_comp,
    }


# ----------------------------------------------------------------------------
# Compress the ViT output (pooler + deepstack) for one sample's video clips.
# ----------------------------------------------------------------------------
def compress_video_features(
    pooler_items: List[torch.Tensor],          # list of (n_orig_i, D) per cam
    deepstack_layers: List[torch.Tensor],      # list of (sum n_orig, D) per layer
    per_item_orig: List[int],
    per_item_comp: List[int],
    compress_ratio: int = COMPRESS_RATIO,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Apply FasterVLM x``ratio`` to the pooler, and the SAME selection to every
    deepstack layer. Returns:
        comp_pooler   : (sum per_item_comp, D)  — concatenated across cams
        comp_deepstack: list of (sum per_item_comp, D), one per layer
    Token order is item-major then within-item spatial order (matches the
    trimmed video-pad layout: first ``comp[0]`` positions are cam0, etc.).
    """
    comp_pooler_parts: List[torch.Tensor] = []
    # Per-cam kept indices into the concatenated deepstack (which is item-major).
    kept_idx_global: List[torch.Tensor] = []
    off = 0
    for i, e in enumerate(pooler_items):
        e = e.detach()
        n = int(per_item_orig[i])
        if e.shape[0] != n:
            raise RuntimeError(f"video block {i}: pooler {e.shape[0]} != orig {n}")
        idx = fastervlm_indices(e, int(compress_ratio))
        target = int(per_item_comp[i])
        if idx.shape[0] != target:
            if idx.shape[0] > target:
                idx = idx[:target]
            else:
                # pad by repeating last index (mirrors the live zero-pad guard,
                # but index-based so deepstack stays aligned)
                pad = idx.new_full((target - idx.shape[0],), int(idx[-1]))
                idx = torch.cat([idx, pad])
        comp_pooler_parts.append(e[idx])
        kept_idx_global.append(idx + off)
        off += n

    comp_pooler = torch.cat(comp_pooler_parts, dim=0)
    kept_all = torch.cat(kept_idx_global, dim=0)

    comp_deepstack = []
    for layer in deepstack_layers:
        layer = layer.detach()
        comp_deepstack.append(layer[kept_all])
    return comp_pooler, comp_deepstack
