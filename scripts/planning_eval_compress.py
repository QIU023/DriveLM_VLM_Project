"""Training-free re-eval of visual-token compression on the nuScenes planning VLA.

Goal (suggestion (1) in docs/interview_prep_qwen25vl_architecture.md): produce a
**unified spatial x temporal Pareto frontier on the DRIVING metric** (L2 /
collision vs. total visual-token budget) WITHOUT retraining.

Why no retraining: FasterVLM / PruMerge / PyramidDrop are training-free (they
prune by vision-encoder attention or token similarity at inference). So we take
the already-trained planning checkpoint, insert a compressor in the visual-feature
path, and re-run the same eval as ``planning_eval.py``. The temporal compressors
(temporal_pool / VTM / LongVU) plug into the SAME hook -> spatial x temporal
combos are just two knobs.

Hook strategy mirrors the qformer/pixelshuffle external-projector path in
``planning_eval._run_batch``:

  1. Pre-compute per-item compressed token count from ``video_grid_thw`` plus
     the user-requested spatial/temporal ratios.
  2. Trim the prompt's ``<|video_pad|>`` placeholder runs (per video block,
     since multi-cam = multiple runs) down to the compressed count, updating
     ``attention_mask`` in lockstep and left-padding the batch back to a common
     length so batched ``generate`` still works.
  3. Rebuild ``video_grid_thw`` to a shape whose post-merger product equals
     the compressed per-item count.
  4. Monkey-patch ``inner.get_video_features`` so it runs the spatial/temporal
     compressor on the raw vision-tower output and returns a list of tensors
     whose total row count matches the new placeholder count. This satisfies
     transformers 5.6's strict ``torch_compilable_check`` in
     ``get_placeholder_mask``.

Usage (single GPU; lift the DP-shard loop from planning_eval.main for multi-GPU):

    python scripts/planning_eval_compress.py \
        --ckpt checkpoints_qwen25/nuscenes_planning_3b_full_sft/final \
        --infos-val data/uniad_infos/nuscenes_infos_temporal_val.pkl \
        --spatial-method fastervlm --spatial-ratio 4 \
        --temporal-method temporal_pool --temporal-ratio 2 \
        --max-samples 200 --output eval_results/planning_s4xt2.json

Sweep these knobs (spatial in {none,fastervlm,prumerge,pyramiddrop,crp} x
{2,4,8,16}, temporal in {none,temporal_pool,vtm,longvu} x {1,2,4}) to draw the
Pareto frontier. Each run records the ACHIEVED token budget so points are
comparable.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Reuse everything from planning_eval -- metrics, batch builder, ETC.
from planning_eval import (  # noqa: E402
    _build_batch_inputs,
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
# The compression hook (the novel part -- everything else is reuse)
# ============================================================================

class _FakeVisOut:
    """Mimics the vision-feature return object the LM forward expects.

    The forward path does ``torch.cat(self.pooler_output, dim=0)``, so we
    return a list of compressed (Nq_i, D) tensors, one per video block.
    """
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


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _factor_grid(target: int, ms: int) -> Tuple[int, int, int]:
    """Pick (1, h*ms, w*ms) with h*w == target and aspect close to 1.

    Mirrors planning_eval._trim_and_pad_for_projector's inline ``_factor_grid``;
    we want the rebuilt ``video_grid_thw`` to land on the SAME post-merger
    token count as the compressed feature count so transformers' grid bookkeeping
    stays consistent (rope deltas, etc.).
    """
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


def _per_item_post_counts(video_grid_thw: torch.Tensor, merge_size: int) -> List[int]:
    """Return per-video-item post-spatial-merge token count = T*H*W / ms**2."""
    g = video_grid_thw
    if g.dim() == 1:
        g = g.unsqueeze(0)
    out = []
    for i in range(g.shape[0]):
        t = int(g[i, 0]); h = int(g[i, 1]); w = int(g[i, 2])
        out.append(t * (h // merge_size) * (w // merge_size))
    return out


def _trim_video_pad_for_compression(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    video_grid_thw: torch.Tensor,
    video_token_id: int,
    per_item_orig: List[int],
    per_item_comp: List[int],
    merge_size: int = 2,
):
    """Slice each contiguous run of ``<|video_pad|>`` tokens down to its
    per-item compressed count.

    The chat template emits one ``<|vision_start|><|video_pad|>...<|vision_end|>``
    triple per ``{"type":"video"}`` content block, so the placeholder tokens
    form NUM_ITEMS contiguous runs per sample. We walk the input_ids row by
    row, locate each run, and drop the tail to keep exactly
    ``per_item_comp[k]`` placeholders for the k-th video block in that row.

    Returns (new_input_ids, new_attention_mask, new_video_grid_thw).
    """
    device = input_ids.device
    B = input_ids.shape[0]
    num_items = int(video_grid_thw.shape[0])
    if num_items == 0:
        return input_ids, attention_mask, video_grid_thw
    items_per_sample = num_items // B
    if items_per_sample * B != num_items:
        raise RuntimeError(
            f"video_grid_thw num_items={num_items} not divisible by batch B={B}"
        )

    new_ids_list, new_mask_list = [], []
    for b in range(B):
        ids = input_ids[b]
        msk = attention_mask[b]
        vid_mask = (ids == video_token_id)
        vid_pos = vid_mask.nonzero(as_tuple=True)[0]
        if len(vid_pos) == 0:
            new_ids_list.append(ids)
            new_mask_list.append(msk)
            continue

        # Split positions into contiguous runs (one run per video block).
        runs: List[List[int]] = []
        cur: List[int] = [int(vid_pos[0].item())]
        for p in vid_pos[1:].tolist():
            if p == cur[-1] + 1:
                cur.append(p)
            else:
                runs.append(cur)
                cur = [p]
        runs.append(cur)
        if len(runs) != items_per_sample:
            raise RuntimeError(
                f"sample {b}: found {len(runs)} <|video_pad|> runs but "
                f"video_grid_thw says {items_per_sample} items per sample"
            )

        drop_positions: List[int] = []
        for k, run in enumerate(runs):
            global_k = b * items_per_sample + k
            orig_n = per_item_orig[global_k]
            comp_n = per_item_comp[global_k]
            if len(run) != orig_n:
                raise RuntimeError(
                    f"sample {b} block {k}: run length {len(run)} != "
                    f"expected post-merger count {orig_n}"
                )
            if comp_n < orig_n:
                drop_positions.extend(run[comp_n:])

        if drop_positions:
            keep = torch.ones(len(ids), dtype=torch.bool, device=device)
            keep[torch.tensor(drop_positions, device=device)] = False
            new_ids_list.append(ids[keep])
            new_mask_list.append(msk[keep])
        else:
            new_ids_list.append(ids)
            new_mask_list.append(msk)

    # Left-pad to common length (batched greedy generate convention).
    max_len = max(t.shape[0] for t in new_ids_list)
    pad_id = 0
    for i in range(B):
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

    # Rebuild video_grid_thw so each row reports a (1, h*ms, w*ms) shape whose
    # post-merger product equals per_item_comp[k]. Keep dtype/device.
    new_rows = []
    for k in range(num_items):
        comp_n = per_item_comp[k]
        new_rows.append(list(_factor_grid(comp_n, merge_size)))
    new_grid = torch.tensor(new_rows, dtype=video_grid_thw.dtype, device=video_grid_thw.device)
    return new_input_ids, new_attn_mask, new_grid


def _compress_per_item(
    embeds: torch.Tensor,
    n_per_item: int,
    spatial_method: str,
    spatial_ratio: int,
    temporal_compressor,
    temporal_ratio: int,
    T: int,
) -> torch.Tensor:
    """Apply spatial-then-temporal compression on ONE video block's tokens.

    Input  : embeds of shape (n_per_item, D) = T * N_post_per_frame rows.
    Output : (n_compressed, D).
    """
    if n_per_item % max(T, 1) != 0:
        # T is the dataset's num_past_frames; if the grid_thw says a different
        # temporal length, fall back to using grid_thw's T (already factored in
        # by the caller since n_per_item = T*H*W/ms**2 and the [t,h,w] row
        # records T=num_past_frames for video).
        pass
    N_per_frame = n_per_item // max(T, 1)
    out = embeds  # (n_per_item, D)

    # ---- SPATIAL axis: prune each frame independently ----
    if spatial_method and spatial_method != "none" and spatial_ratio > 1:
        per_frame = out.view(T, N_per_frame, embeds.shape[-1])
        kept = []
        for f in range(T):
            grid = torch.tensor([[1, *_factor_hw(N_per_frame)]], device=embeds.device)
            comp, _ = compress_visual_tokens(
                per_frame[f], grid, spatial_method, spatial_ratio
            )
            kept.append(comp)
        out = torch.cat(kept, dim=0)
        N_per_frame = out.shape[0] // T

    # ---- TEMPORAL axis: collapse T frames into T/temporal_ratio groups ----
    if temporal_compressor is not None and temporal_ratio > 1:
        D = out.shape[-1]
        # Group T frames into groups of size temporal_ratio, average within
        # each group. If T is not divisible, the trailing tail is averaged
        # together (degenerate but safe).
        full_groups = T // temporal_ratio
        out_groups = []
        for g in range(full_groups):
            s = g * temporal_ratio
            chunk = out.view(T, N_per_frame, D)[s:s + temporal_ratio]  # (tr, N, D)
            comp = temporal_compressor(chunk.unsqueeze(0))  # (1, N, D)
            out_groups.append(comp.squeeze(0))
        tail = T - full_groups * temporal_ratio
        if tail > 0:
            chunk = out.view(T, N_per_frame, D)[-tail:]
            comp = temporal_compressor(chunk.unsqueeze(0))
            out_groups.append(comp.squeeze(0))
        out = torch.cat(out_groups, dim=0)  # (N_groups * N_per_frame, D)

    return out


def install_compression_hook(
    model,
    *,
    spatial_method: str,
    spatial_ratio: int,
    temporal_method: str,
    temporal_ratio: int,
    num_past_frames: int,
    meter: TokenBudgetMeter,
    state: dict,
):
    """Monkey-patch ``inner.get_video_features`` to run spatial then temporal
    compression on the (frozen) vision-encoder output and return per-item
    compressed tensors whose row count matches the trimmed placeholder count.

    ``state`` is a mutable dict the outer batch loop populates with the keys
    ``orig_pv``, ``orig_grid``, ``per_item_orig``, ``per_item_comp``, ``T``.
    This lets the closure see the un-modified pixel_values / grid even though
    the model is being called with the trimmed input_ids / rebuilt grid.

    Returns a restore fn.
    """
    inner = model.model  # Qwen2_5_VLModel
    orig = inner.get_video_features

    temporal = None
    if temporal_method and temporal_method != "none" and temporal_ratio > 1:
        temporal = make_compressor(temporal_method)

    def _patched(_pv_unused, _grid_unused, **_kw):
        # Use the ORIGINAL pixel_values / video_grid_thw the dataset produced,
        # not the trimmed/rebuilt copies that were passed into generate(). The
        # vision tower needs raw inputs to produce the pre-compression features.
        pv = state["orig_pv"]
        grid = state["orig_grid"]
        with torch.no_grad():
            real = orig(pv, grid)
            embeds = real.pooler_output  # tuple of (n_post_i, D) tensors

        if not isinstance(embeds, (tuple, list)):
            # Defensive: older transformers returned a single tensor.
            embeds = torch.split(embeds, state["per_item_orig"])

        per_item_orig = state["per_item_orig"]
        per_item_comp = state["per_item_comp"]
        T = state["T"]
        compressed_items: List[torch.Tensor] = []
        n_in = 0
        n_out = 0
        for i, e in enumerate(embeds):
            e = e.detach()
            n_in += e.shape[0]
            comp = _compress_per_item(
                e,
                n_per_item=per_item_orig[i],
                spatial_method=spatial_method,
                spatial_ratio=spatial_ratio,
                temporal_compressor=temporal,
                temporal_ratio=temporal_ratio,
                T=T,
            )
            # Adjust if our compression produced something slightly off from
            # per_item_comp (round-down/up from non-divisible factors). Pad or
            # truncate to per_item_comp[i] so placeholders match exactly.
            target = per_item_comp[i]
            if comp.shape[0] != target:
                if comp.shape[0] > target:
                    comp = comp[:target]
                else:
                    pad = target - comp.shape[0]
                    pad_t = comp.new_zeros((pad, comp.shape[-1]))
                    comp = torch.cat([comp, pad_t], dim=0)
            compressed_items.append(comp)
            n_out += comp.shape[0]

        meter.record(n_in, n_out)
        return _FakeVisOut(compressed_items)

    inner.get_video_features = _patched

    def _restore():
        inner.get_video_features = orig

    return _restore


def _factor_hw(n: int) -> tuple:
    """Best-effort (h, w) factorization of a per-frame token count for the
    spatial compressor's grid_thw. Replace with the encoder's true layout if
    you need exact spatial bookkeeping inside the compressor."""
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

    # Pull config knobs needed for token bookkeeping.
    merge_size = int(getattr(model.config.vision_config, "spatial_merge_size", 2))
    video_token_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")

    meter = TokenBudgetMeter()
    # ``state`` is the closure-shared dict between the batch loop and the
    # patched ``get_video_features`` (see install_compression_hook docstring).
    state: dict = {}
    restore = install_compression_hook(
        model,
        spatial_method=args.spatial_method, spatial_ratio=args.spatial_ratio,
        temporal_method=args.temporal_method, temporal_ratio=args.temporal_ratio,
        num_past_frames=args.num_past_frames, meter=meter, state=state,
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

                # ---- Move inputs to device + cast pixels (mirrors planning_eval._run_batch).
                inputs = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in inputs.items()
                }
                if "pixel_values_videos" in inputs:
                    inputs["pixel_values_videos"] = inputs["pixel_values_videos"].to(dtype)
                if "pixel_values" in inputs and isinstance(inputs["pixel_values"], torch.Tensor):
                    inputs["pixel_values"] = inputs["pixel_values"].to(dtype)

                # ---- Plan the per-item compressed counts ----
                eff_ratio = max(1, int(args.spatial_ratio)) * max(1, int(args.temporal_ratio))
                per_item_orig = _per_item_post_counts(inputs["video_grid_thw"], merge_size)
                per_item_comp = [max(1, n // eff_ratio) for n in per_item_orig]

                # ---- Trim the prompt's placeholder runs to the compressed counts ----
                new_ids, new_mask, new_grid = _trim_video_pad_for_compression(
                    inputs["input_ids"], inputs["attention_mask"],
                    inputs["video_grid_thw"], video_token_id,
                    per_item_orig, per_item_comp, merge_size=merge_size,
                )

                # ---- Publish state for the patched get_video_features closure ----
                state["orig_pv"] = inputs["pixel_values_videos"]
                state["orig_grid"] = inputs["video_grid_thw"]
                state["per_item_orig"] = per_item_orig
                state["per_item_comp"] = per_item_comp
                state["T"] = int(args.num_past_frames)

                # ---- Generate with the rewritten prompt ----
                prompt_len = new_ids.shape[1]
                gen_kwargs = dict(
                    input_ids=new_ids,
                    attention_mask=new_mask,
                    pixel_values_videos=inputs["pixel_values_videos"],
                    video_grid_thw=new_grid,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=processor.tokenizer.pad_token_id or 0,
                    use_cache=True,
                )
                if "second_per_grid_ts" in inputs and inputs["second_per_grid_ts"] is not None:
                    gen_kwargs["second_per_grid_ts"] = inputs["second_per_grid_ts"]
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
        # TODO: collision rate -- lift the _planning_metric port from planning_eval.
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
