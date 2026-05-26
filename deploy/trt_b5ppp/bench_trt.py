#!/usr/bin/env /venv/trt_llm/bin/python
"""B.5'' v2 TRT-LLM bench: TTFT / decode / throughput / mem / parity / L2.

Reuses the existing B.5'' mm-disagg pattern (phase1_5_vision_embeds +
phase2_mrope_config + DisaggregatedParams with pre-computed
multimodal_embedding handles) — vision tower forward happens once per sample
on HF (bf16), the LM forward runs on TRT in the requested precision.

For B.5'' v2 the visual payload is 1-cam × 4f + HD-map (2800 + 121 = 2921
visual LM tokens). For the prior 3-cam B.5''' baseline override
EXPECTED_VIDEO_TOKENS / EXPECTED_IMAGE_TOKENS via _common.py or feed a 3-cam
ckpt + 3-cam config_yaml.

Output JSON schema matches deploy/trt_bench/B5pp_trt_qwen3vl_multimodal_bf16.json
so the comparison table builder works unchanged. New fields:
  compress_method, compress_ratio, visual_tokens_in, visual_tokens_out.

Usage:
    /venv/trt_llm/bin/python bench_trt.py \
        --ckpt /path/to/{final,quant_fp8,quant_nvfp4} \
        --precision {bf16,fp8,nvfp4} \
        [--n-warmup 3] [--n-runs 20] [--max-new-tokens 14] \
        [--mm-payload {real,text-only}] \
        [--l2-n 50]      # 0 = skip L2 eval (default skips in-proc; see F6)
        [--compress-method {none,fastervlm,prumerge,pyramiddrop,avg_pool}] \
        [--compress-ratio 4] \
        [--out /path/to/out.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from _common import apply_venv_shims  # noqa: E402
apply_venv_shims()

from _common import (  # noqa: E402
    DEFAULT_BENCH_OUT_DIR,
    DEFAULT_CKPT,
    DEFAULT_CONFIG_YAML,
    EXPECTED_IMAGE_TOKENS,
    EXPECTED_VIDEO_TOKENS,
    add_project_paths,
    build_calib_dataset,
    peak_mem_gb,
    project_base,
    reset_peak_mem,
)


def percentile(values, p):
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    f = int(k); c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f) if f != c else s[f]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TRT-LLM bench for B.5'' v2 Qwen3-VL-4B 1-cam (multimodal mm-disagg)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    p.add_argument("--precision", choices=["bf16", "fp8", "nvfp4"], required=True)
    p.add_argument("--n-warmup", type=int, default=3)
    p.add_argument("--n-runs", type=int, default=20)
    p.add_argument("--max-new-tokens", type=int, default=14,
                   help="6 waypoints × 2 dims + 2 boundary = 14 traj tokens")
    p.add_argument("--mm-payload", choices=["real", "text-only"], default="real",
                   help="real = 1-cam video + HD-map + bbox/ego/prompt; "
                        "text-only = no images/videos, LM-only timing")
    p.add_argument("--l2-n", type=int, default=50,
                   help="Number of val samples for L2 eval (0 = skip; 5000 ~ full)")
    p.add_argument("--out", default=None,
                   help="Output JSON path (default: derived from ckpt + precision)")
    p.add_argument("--config-yaml", default=DEFAULT_CONFIG_YAML)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-seq-len", type=int, default=12288)
    p.add_argument("--max-batch-size", type=int, default=1)
    p.add_argument("--free-gpu-mem-frac", type=float, default=0.4,
                   help="Lower for bf16+fp8 (need room for HF vision tower), "
                        "raise for text-only mode")
    p.add_argument("--skip-parity-gate", action="store_true",
                   help="Skip HF baseline top-1 comparison (faster, but no parity check)")
    # ---- F6: L2 in-process gate (default OFF — reloading HF after TRT OOMs 32GB)
    p.add_argument("--l2-skip-inproc", action="store_true", default=True,
                   help="Skip the in-process HF-reload+L2-eval block (default: True). "
                        "Reloading HF after TRT load OOMs single 32GB cards. To enable "
                        "anyway pass --no-l2-skip-inproc; preferred path is to run L2 "
                        "via scripts/planning_eval.py in a separate process.")
    p.add_argument("--no-l2-skip-inproc", dest="l2_skip_inproc", action="store_false")
    # ---- F7: training-free visual-token compression (applied to VIDEO only) ----
    p.add_argument("--compress-method",
                   choices=["none", "fastervlm", "prumerge", "pyramiddrop", "avg_pool"],
                   default="none",
                   help="Spatial compression on video tokens BEFORE TRT injection. "
                        "HD-map image tokens (~121) are NEVER compressed. Uses the "
                        "same compress_visual_tokens() implementation as "
                        "scripts/planning_eval_compress.py.")
    p.add_argument("--compress-ratio", type=int, default=1,
                   help="Compression ratio (e.g. 4 = keep 1/4 of video tokens; "
                        "2800 -> 700). Ignored when --compress-method=none.")
    return p.parse_args()


def _derive_out(args: argparse.Namespace) -> str:
    if args.out is not None:
        return args.out
    name = f"B5ppp_trt_qwen3vl_multimodal_{args.precision}"
    if args.mm_payload == "text-only":
        name += "_text_only"
    return str(Path(DEFAULT_BENCH_OUT_DIR) / f"{name}.json")


# ---------------------------------------------------------------------------
# Prompt rebuilders (collapse expanded mm-pad runs back to single pads — TRT
# re-expands using embed.shape[0]). Identical to phase5_bench_full_multimodal.
# ---------------------------------------------------------------------------

def build_unexpanded_prompt(input_ids, prompt_len, tokenizer,
                            image_pad_id, video_pad_id):
    """Same trick as the B.5'' phase5 script: collapse pad runs to single
    pads and masquerade video_pad as image_pad (TRT-LLM 1.3.0rc15
    Qwen3VLInputProcessorBase.get_prompt_token_ids only counts image_token_id;
    the LM doesn't care because fuse_input_embeds replaces with our embed)."""
    new_ids = []
    i = 0
    ids = input_ids[:prompt_len].tolist()
    while i < len(ids):
        t = ids[i]
        if t == image_pad_id:
            new_ids.append(image_pad_id)
            j = i
            while j < len(ids) and ids[j] == image_pad_id:
                j += 1
            i = j
        elif t == video_pad_id:
            new_ids.append(image_pad_id)
            j = i
            while j < len(ids) and ids[j] == video_pad_id:
                j += 1
            i = j
        else:
            new_ids.append(t)
            i += 1
    text = tokenizer.decode(new_ids, skip_special_tokens=False)
    return text, new_ids


def count_orig_mm_blocks(input_ids, prompt_len, image_pad_id, video_pad_id):
    """Count number of image / video CONTIGUOUS blocks in prompt."""
    orig_ids = input_ids[:prompt_len].tolist()
    n_img = 0; n_vid = 0
    j = 0
    while j < len(orig_ids):
        t = orig_ids[j]
        if t == image_pad_id:
            n_img += 1
            while j < len(orig_ids) and orig_ids[j] == image_pad_id:
                j += 1
        elif t == video_pad_id:
            n_vid += 1
            while j < len(orig_ids) and orig_ids[j] == video_pad_id:
                j += 1
        else:
            j += 1
    return n_img, n_vid


# ---------------------------------------------------------------------------
# F7: helpers reused from scripts/planning_eval_compress.py for video-token
# compression at the bench_trt layer. We import lazily (only when method != none)
# so the bench has zero compression cost in the baseline path.
# ---------------------------------------------------------------------------

def _factor_grid_simple(target: int, ms: int = 2):
    """Pick (1, h*ms, w*ms) with h*w == target and aspect closest to 1.
    Mirror of planning_eval_compress._factor_grid (kept local to avoid the
    extra import dependency in the bench path)."""
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


def _find_pad_runs(input_ids_1d, pad_id):
    """Return list of (start, end_exclusive) for each contiguous run of pad_id.
    Qwen3-VL interleaves timestamp/text tokens between temporal video frames, so
    a single camera's video appears as T separate <|video_pad|> runs."""
    ids = input_ids_1d.tolist()
    runs = []
    i, n = 0, len(ids)
    while i < n:
        if ids[i] == pad_id:
            j = i
            while j < n and ids[j] == pad_id:
                j += 1
            runs.append((i, j)); i = j
        else:
            i += 1
    return runs


def _trim_video_pad_runs_1d(input_ids_1d, video_pad_id, per_block_target):
    """Trim each contiguous <|video_pad|> run in a 1-D input_ids tensor to
    its target count. Returns (new_input_ids_1d, new_prompt_len).

    Mirrors planning_eval_compress._trim_video_pad_for_compression but on a
    SINGLE sample (no batch / padding) since bench_trt builds per-sample."""
    import torch
    ids = input_ids_1d.tolist()
    runs = []  # list of (start_idx, end_exclusive)
    i = 0
    n = len(ids)
    while i < n:
        if ids[i] == video_pad_id:
            j = i
            while j < n and ids[j] == video_pad_id:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    if not runs:
        return input_ids_1d, n
    if len(runs) != len(per_block_target):
        raise RuntimeError(
            f"video_pad runs={len(runs)} but per_block_target has "
            f"{len(per_block_target)} entries"
        )
    drop = set()
    for (s, e), tgt in zip(runs, per_block_target):
        run_len = e - s
        if tgt < run_len:
            for k in range(s + tgt, e):
                drop.add(k)
    keep_mask = torch.ones(n, dtype=torch.bool, device=input_ids_1d.device)
    if drop:
        keep_mask[torch.tensor(sorted(drop), device=input_ids_1d.device)] = False
    new_ids = input_ids_1d[keep_mask]
    return new_ids, int(new_ids.shape[0])


# ---------------------------------------------------------------------------
# Build per-sample TRT request payload (text prompt + disagg params)
# ---------------------------------------------------------------------------

def build_trt_request_for_sample(*, sample, hf_model, processor, image_pad_id,
                                 video_pad_id, device, dtype, mm_payload,
                                 compress_method="none", compress_ratio=1):
    """Returns (text_prompt, DisaggregatedParams | None, prompt_len).

    For mm_payload="text-only" we strip the vision blocks entirely and don't
    pass disagg params — pure LM timing.
    """
    import torch
    from phase1_5_vision_embeds import assemble_mm_embedding
    from phase2_mrope_config import build_mrope_config
    from tensorrt_llm._torch.shared_tensor import SharedTensorContainer
    from tensorrt_llm.disaggregated_params import DisaggregatedParams

    tok = processor.tokenizer
    prompt_len = int(sample["_meta_prompt_len"])

    if mm_payload == "text-only":
        # Strip ALL vision blocks (image_pad / video_pad runs) AND the
        # surrounding <|vision_start|> / <|vision_end|> markers — yields a
        # clean text-only prompt for LM-only TTFT measurement.
        vision_start_id = tok.convert_tokens_to_ids("<|vision_start|>")
        vision_end_id = tok.convert_tokens_to_ids("<|vision_end|>")
        ids = sample["input_ids"][:prompt_len].tolist()
        out = []
        skip = False
        for t in ids:
            if t == vision_start_id:
                skip = True
                continue
            if t == vision_end_id:
                skip = False
                continue
            if skip:
                continue
            if t in (image_pad_id, video_pad_id):
                continue
            out.append(t)
        text_prompt = tok.decode(out, skip_special_tokens=False)
        return text_prompt, None, len(out), 0, 0

    # ---- real multimodal payload ----
    embed_result = assemble_mm_embedding(
        hf_model=hf_model,
        pixel_values=sample.get("pixel_values"),
        image_grid_thw=sample.get("image_grid_thw"),
        pixel_values_videos=sample.get("pixel_values_videos"),
        video_grid_thw=sample.get("video_grid_thw"),
        modality_order=("video", "image"),  # B.5'' v2 = video first, image second
    )
    mm_embedding_full = embed_result["mm_embedding"].to(device, dtype=dtype)
    n_video = embed_result["video_mm_tokens"]
    n_image = embed_result["image_mm_tokens"]

    # ---- F5: sanity-check that video token count matches what cfg predicts.
    # 0.9 slack covers <8% chat-template variance + boundary patches.
    assert n_video >= int(EXPECTED_VIDEO_TOKENS * 0.9), (
        f"[F5 GATE] video tokens {n_video} << expected "
        f"{EXPECTED_VIDEO_TOKENS} (cfg-derived). The wrong number of cams or "
        f"wrong video_max_pixels is in play. Aborting before TRT injection."
    )
    video_embed = mm_embedding_full[:n_video].contiguous()
    image_embed = mm_embedding_full[n_video:].contiguous()

    # ---- F7: optional FasterVLM-style spatial compression on VIDEO only ----
    # HD-map (image) is left untouched (~121 tokens is not worth compressing).
    n_video_in = int(n_video)
    n_video_out = int(n_video)
    if compress_method != "none" and int(compress_ratio) > 1:
        import torch
        add_project_paths()
        from visual_compress import compress_visual_tokens  # noqa: E402

        vid_thw = sample.get("video_grid_thw")
        if vid_thw is None:
            raise RuntimeError("compress requested but video_grid_thw missing")
        vid_thw_local = vid_thw if vid_thw.dim() == 2 else vid_thw.unsqueeze(0)
        # Build per-frame post-merger H,W grid for compress_visual_tokens. It
        # takes pre-merger thw and divides internally — so feed the RAW grid
        # rebuilt onto the actual video_embed row count.
        # Each row = [T, H_pre, W_pre]; rows must sum (after // ms**2) to n_video.
        ms = int(getattr(hf_model.config.vision_config, "spatial_merge_size", 2))
        # Translate post-merger video_embed back into the per-item grid the
        # compressor expects. We rebuild a synthetic [T, H_post*ms, W_post*ms]
        # grid per video block using _factor_grid_simple so the compressor
        # internal reshape works on the n_post_per_block rows.
        # Qwen3-VL interleaves timestamp/text tokens between temporal frames, so
        # ONE camera's video spans T separate <|video_pad|> runs in the prompt
        # (NOT 1 contiguous run per video_grid_thw row). Segment the embed +
        # compress + trim PER RUN so the embed row count == prompt video_pad
        # count exactly, and emit one grid row (T=1) per run so the post-merge
        # sum still adds up. (Old per-grid-row logic assumed 1 run/cam and broke
        # with "video_pad runs=2 but per_block_target has 1 entries".)
        prompt_ids_1d = sample["input_ids"][:prompt_len]
        runs = _find_pad_runs(prompt_ids_1d, video_pad_id)
        run_lens = [e - s for (s, e) in runs]
        assert sum(run_lens) == n_video, (
            f"prompt video_pad total {sum(run_lens)} != n_video {n_video}; "
            f"cannot map embed rows to runs"
        )
        offset = 0
        comp_chunks = []
        per_run_comp = []
        for rlen in run_lens:
            seg = video_embed[offset:offset + rlen]
            offset += rlen
            # already POST-merger features → grid [1,1,rlen] (compressor only
            # uses the t*h*w product to pick the keep count = rlen // ratio).
            grid_synth = torch.tensor([[1, 1, rlen]], device=seg.device,
                                      dtype=torch.int64)
            comp, _ = compress_visual_tokens(
                seg, grid_synth, compress_method, int(compress_ratio)
            )
            comp_chunks.append(comp.contiguous())
            per_run_comp.append(int(comp.shape[0]))

        video_embed = torch.cat(comp_chunks, dim=0).contiguous()
        n_video_out = int(video_embed.shape[0])
        # One grid row per run (T=1), post-merge total = sum(per_run_comp).
        new_video_rows = [list(_factor_grid_simple(c, ms)) for c in per_run_comp]
        new_video_thw = torch.tensor(new_video_rows, dtype=vid_thw.dtype,
                                     device=vid_thw.device)
        # Trim each <|video_pad|> run to its compressed count (runs now match).
        new_prompt_ids, new_prompt_len = _trim_video_pad_runs_1d(
            prompt_ids_1d, video_pad_id, per_run_comp
        )
        # Splice the trimmed prompt back into a full input_ids vector
        # (preserves the post-prompt label region for safety).
        rest = sample["input_ids"][prompt_len:]
        full_new_ids = torch.cat([new_prompt_ids, rest], dim=0)
        # New attention_mask (left part = 1 for prompt, rest mirrors original)
        new_attn = torch.cat([
            torch.ones(new_prompt_len, dtype=sample["attention_mask"].dtype,
                       device=sample["attention_mask"].device),
            sample["attention_mask"][prompt_len:],
        ], dim=0)
        # Rebuild mm_token_type_ids in lockstep — recompute by scanning the
        # new prompt for video_pad / image_pad ids.
        new_mtt = torch.zeros_like(full_new_ids)
        for k, t in enumerate(full_new_ids[:new_prompt_len].tolist()):
            if t == video_pad_id:
                new_mtt[k] = 2  # video
            elif t == image_pad_id:
                new_mtt[k] = 1  # image
        # Stash a compressed-aware sample view for downstream rebuild paths.
        sample = dict(sample)
        sample["input_ids"] = full_new_ids
        sample["attention_mask"] = new_attn
        sample["mm_token_type_ids"] = new_mtt
        sample["video_grid_thw"] = new_video_thw
        sample["_meta_prompt_len"] = new_prompt_len
        prompt_len = new_prompt_len
        n_video = n_video_out
        # Rebuild mm_embedding_full with compressed video portion
        mm_embedding_full = torch.cat([video_embed, image_embed], dim=0).contiguous()

    # Per-cam video block split — Qwen3-VL inserts ONE <|video_pad|> per
    # temporal patch per video. Same logic as phase5.
    text_prompt, unexpanded_ids = build_unexpanded_prompt(
        sample["input_ids"], prompt_len, tok, image_pad_id, video_pad_id
    )
    n_img_blocks, n_vid_blocks = count_orig_mm_blocks(
        sample["input_ids"], prompt_len, image_pad_id, video_pad_id
    )
    if n_vid_blocks == 0 or n_video % n_vid_blocks != 0:
        raise RuntimeError(
            f"video block count {n_vid_blocks} does not divide n_video {n_video}"
        )
    if n_img_blocks == 0 or n_image % n_img_blocks != 0:
        raise RuntimeError(
            f"image block count {n_img_blocks} does not divide n_image {n_image}"
        )
    video_chunk = n_video // n_vid_blocks
    image_chunk = n_image // n_img_blocks
    video_chunks = [video_embed[i * video_chunk:(i + 1) * video_chunk].contiguous()
                    for i in range(n_vid_blocks)]
    image_chunks = [image_embed[i * image_chunk:(i + 1) * image_chunk].contiguous()
                    for i in range(n_img_blocks)]

    # Walk the ORIGINAL ids to figure out text-order handle list
    orig_ids = sample["input_ids"][:prompt_len].tolist()
    mm_handles_tensors = []
    vi = 0; ii = 0; k = 0
    while k < len(orig_ids):
        t = orig_ids[k]
        if t == video_pad_id:
            mm_handles_tensors.append(video_chunks[vi]); vi += 1
            while k < len(orig_ids) and orig_ids[k] == video_pad_id: k += 1
        elif t == image_pad_id:
            mm_handles_tensors.append(image_chunks[ii]); ii += 1
            while k < len(orig_ids) and orig_ids[k] == image_pad_id: k += 1
        else:
            k += 1
    # NOTE: build the shared-tensor handles from CPU tensors, NOT CUDA. The
    # CUDA path uses cudaIPC handles whose consumer-side restore calls
    # pidfd_getfd, blocked by this container's seccomp ("Operation not
    # permitted"). CPU tensors take SharedTensorContainer's cpu_handle_to_dict
    # serialize path (no pidfd); TRT-LLM moves them back to GPU internally.
    # CRITICAL: torch file_system sharing unlinks the /dev/shm segment when the
    # producer storage is GC'd. The same `disagg` handles are reused across
    # parity + warmup + timed runs, so we MUST keep the CPU tensors alive for
    # the whole bench — collect them in `_keepalive` and return it to the caller.
    _mm_cpu = [t.detach().cpu().contiguous() for t in mm_handles_tensors]

    # M-RoPE position ids (Phase 2)
    mrope_full = build_mrope_config(
        model_config=hf_model.config,
        input_ids=sample["input_ids"],
        mm_token_type_ids=sample["mm_token_type_ids"],
        image_grid_thw=sample["image_grid_thw"].clone() if sample.get("image_grid_thw") is not None else None,
        video_grid_thw=sample["video_grid_thw"].clone() if sample.get("video_grid_thw") is not None else None,
        attention_mask=sample["attention_mask"],
    )
    mrope_pos_ids = mrope_full["mrope_position_ids"][:, :, :prompt_len].to(
        device, dtype=torch.int32).contiguous()
    mrope_deltas = mrope_full["mrope_position_deltas"].view(-1).to(
        device, dtype=torch.int32).contiguous()
    _mpc = mrope_pos_ids.detach().cpu().contiguous()
    _mdc = mrope_deltas.detach().cpu().contiguous()

    def make_disagg():
        # FRESH shm per call. TRT-LLM mm-disagg shared-tensor handles are
        # ONE-SHOT (consumer unlinks the /dev/shm segment after restore), so a
        # reused handle dies on the 2nd+ generate ("No such file or directory").
        # We use CPU handles (not CUDA) because this container's seccomp blocks
        # pidfd_getfd. Clone the held CPU tensors to mint a fresh shm each call;
        # return (disagg, clones-to-hold) — the caller MUST keep `clones` alive
        # across the (blocking) generate so the shm isn't GC'd before the
        # executor reads it.
        clones = [c.clone() for c in _mm_cpu]
        cpc = _mpc.clone(); cdc = _mdc.clone()
        mm_h = [SharedTensorContainer.from_tensor(c).dump_to_dict() for c in clones]
        ph = SharedTensorContainer.from_tensor(cpc).dump_to_dict()
        dh = SharedTensorContainer.from_tensor(cdc).dump_to_dict()
        d = DisaggregatedParams(
            request_type="context_and_generation",
            multimodal_embedding_handles=mm_h,
            mrope_position_ids_handle=ph,
            mrope_position_deltas_handle=dh,
        )
        return d, (clones + [cpc, cdc])

    # Return a make_disagg() FACTORY (None for text-only) instead of a single
    # reusable disagg — callers mint a fresh handle-set per generate.
    return text_prompt, make_disagg, prompt_len, n_video_in, n_video_out


# ---------------------------------------------------------------------------
# HF parity gate: load HF, get top-1 first-generated token, then free
# ---------------------------------------------------------------------------

def hf_first_token_topk(hf_model, sample, device, k=5):
    """Run HF model.forward on the full prompt and return top-k token ids of
    the next-token logits at position prompt_len-1."""
    import torch
    prompt_len = int(sample["_meta_prompt_len"])
    batch = {}
    for key in ("input_ids", "attention_mask", "mm_token_type_ids"):
        if key in sample and sample[key] is not None:
            v = sample[key][:prompt_len].unsqueeze(0).to(device)
            batch[key] = v
    for key in ("pixel_values", "pixel_values_videos"):
        if key in sample and sample[key] is not None:
            batch[key] = sample[key].to(device, dtype=torch.bfloat16)
    for key in ("image_grid_thw", "video_grid_thw"):
        if key in sample and sample[key] is not None:
            v = sample[key].to(device)
            # PATCH 2026-05-25: transformers 5.x qwen3_vl expects 2D (N, 3)
            # tensor; dataset may yield 1D (3,) for single-modality samples.
            if v.ndim == 1:
                v = v.unsqueeze(0)
            batch[key] = v
    with torch.inference_mode():
        out = hf_model(**batch)
    logits = out.logits[0, -1]  # (vocab,)
    topk = torch.topk(logits, k=k)
    return topk.indices.tolist(), topk.values.tolist()


# ---------------------------------------------------------------------------
# L2 trajectory error evaluation
# ---------------------------------------------------------------------------

def eval_l2(*, llm, hf_model, processor, image_pad_id, video_pad_id,
            device, dtype, n_samples, max_new_tokens, mm_payload,
            config_yaml):
    """Greedy-decode `n_samples` val samples, decode trajectories, compute
    L2_avg (TemAvg). Returns dict of summary metrics.

    NOTE: only "real" mm_payload makes sense for L2 — text-only loses the
    visual grounding and will produce garbage trajectories.
    """
    import numpy as np
    import torch
    from tensorrt_llm import SamplingParams

    add_project_paths()
    from trajectory_tokenizer import (
        TrajectoryTokenizer, TrajectoryTokenizerConfig,
        BIN_BASE, TRAJ_START_ID, TRAJ_END_ID,
    )
    from planning_eval import decode_waypoints, l2_temavg

    # Build val dataset (cap at n_samples)
    val_ds = build_calib_dataset(
        processor=processor, n_samples=n_samples, split="val",
        config_yaml=config_yaml,
    )
    n_total = len(val_ds)
    print(f"[l2] evaluating L2 on {n_total} val samples (mm_payload={mm_payload})")

    traj_cfg = TrajectoryTokenizerConfig(num_waypoints=val_ds.num_future)
    traj_tok = TrajectoryTokenizer(traj_cfg)

    sp = SamplingParams(max_tokens=max_new_tokens, temperature=0.0)
    per_sample = []  # list of {"L2_avg", "L2_1s", ..., "pred", "gt", "sample_token"}
    t0 = time.time()

    for i in range(n_total):
        try:
            sample = val_ds[i]
        except Exception as e:
            print(f"[l2]   sample {i} build failed: {e}; skip")
            continue
        try:
            text_prompt, make_disagg, _, _, _ = build_trt_request_for_sample(
                sample=sample, hf_model=hf_model, processor=processor,
                image_pad_id=image_pad_id, video_pad_id=video_pad_id,
                device=device, dtype=dtype, mm_payload=mm_payload,
            )
        except Exception as e:
            print(f"[l2]   sample {i} prep failed: {e}; skip")
            continue

        try:
            if make_disagg is not None:
                disagg, _hold = make_disagg()  # fresh shm; _hold alive thru generate
                out = llm.generate([{"prompt": text_prompt}],
                                   sampling_params=sp,
                                   disaggregated_params=disagg)
                del _hold
            else:
                out = llm.generate([{"prompt": text_prompt}],
                                   sampling_params=sp)
        except Exception as e:
            print(f"[l2]   sample {i} generate failed: {e}; skip")
            continue

        token_ids = list(out[0].outputs[0].token_ids)
        pred_wp = decode_waypoints(token_ids, traj_tok, val_ds.num_future)
        gt_wp = sample["_meta_waypoints"].cpu().numpy()
        valid = sample["_meta_valid_mask"].cpu().numpy()
        metrics = l2_temavg(pred_wp, gt_wp, valid)
        per_sample.append({
            "i": i,
            "L2_avg": metrics.get("L2_avg", float("nan")),
            "L2_1s": metrics.get("L2_1s", float("nan")),
            "L2_2s": metrics.get("L2_2s", float("nan")),
            "L2_3s": metrics.get("L2_3s", float("nan")),
        })
        if (i + 1) % 10 == 0 or (i + 1) == n_total:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-6)
            eta = (n_total - i - 1) / max(rate, 1e-6)
            l2_now = np.nanmean([s["L2_avg"] for s in per_sample])
            print(f"[l2]   {i+1}/{n_total}  L2_avg(running)={l2_now:.4f}  "
                  f"rate={rate:.2f} samp/s  eta={eta:.0f}s")

    def _mean(key):
        vals = [s[key] for s in per_sample
                if not math.isnan(s.get(key, float("nan")))]
        return float(sum(vals) / len(vals)) if vals else float("nan")

    summary = {
        "n_samples_evaluated": len(per_sample),
        "L2_avg_mean": _mean("L2_avg"),
        "L2_1s_mean": _mean("L2_1s"),
        "L2_2s_mean": _mean("L2_2s"),
        "L2_3s_mean": _mean("L2_3s"),
        "elapsed_secs": time.time() - t0,
    }
    print(f"[l2] L2 summary: {summary}")
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    out_path = Path(_derive_out(args))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    src = Path(args.ckpt)
    if not src.is_dir():
        print(f"[bench] FATAL: ckpt not found: {src}", file=sys.stderr)
        return 2
    print(f"[bench] ckpt = {src}")
    print(f"[bench] precision = {args.precision}")
    print(f"[bench] mm_payload = {args.mm_payload}")
    print(f"[bench] out = {out_path}")

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    add_project_paths()

    # =====================================================================
    # STEP 1: load HF model — needed for vision pre-compute (always, for real
    # mm payload) AND parity gate (always). After bench, we DEL it to free
    # VRAM before TRT load.
    # =====================================================================
    print(f"[bench] === STEP 1: load HF model (bf16 vision pre-compute + parity) ===")
    t0 = time.perf_counter()
    hf_model = AutoModelForImageTextToText.from_pretrained(
        str(src), torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(args.device).eval()
    processor = AutoProcessor.from_pretrained(str(src))
    tok = processor.tokenizer
    print(f"[bench] HF model loaded in {time.perf_counter()-t0:.1f}s")

    # ---- F4: breadcrumb — log the actual processor caps loaded from the ckpt.
    # If these don't match what the cfg yaml declares, F2 GATE in
    # build_calib_dataset() will fail with a loud error. This prints them up
    # front so the bench log captures what was actually used.
    try:
        ip_max = getattr(processor.image_processor, "max_pixels", None)
        ip_min = getattr(processor.image_processor, "min_pixels", None)
        print(f"[bench][F4] processor.image_processor: max_pixels={ip_max} "
              f"min_pixels={ip_min}")
    except Exception as _e:
        print(f"[bench][F4] image_processor caps probe failed: {_e}")
    try:
        vp = getattr(processor, "video_processor", None)
        if vp is not None:
            print(f"[bench][F4] processor.video_processor.size = {vp.size}")
        else:
            print(f"[bench][F4] processor has no video_processor (unexpected for Qwen3-VL)")
    except Exception as _e:
        print(f"[bench][F4] video_processor caps probe failed: {_e}")

    image_pad_id = tok.convert_tokens_to_ids("<|image_pad|>")
    video_pad_id = tok.convert_tokens_to_ids("<|video_pad|>")
    print(f"[bench] image_pad={image_pad_id} video_pad={video_pad_id}")

    # =====================================================================
    # STEP 2: build bench sample (val[0])
    # =====================================================================
    print(f"[bench] === STEP 2: build val[0] bench sample ===")
    val_ds_one = build_calib_dataset(
        processor=processor, n_samples=2, split="val",
        config_yaml=args.config_yaml,
    )
    sample = val_ds_one[0]
    prompt_len = int(sample["_meta_prompt_len"])
    print(f"[bench] val[0] prompt_len = {prompt_len}")

    # =====================================================================
    # STEP 3: build TRT request (vision precompute + disagg params)
    # =====================================================================
    print(f"[bench] === STEP 3: build TRT request (mm-disagg pattern) ===")
    text_prompt, make_disagg, eff_prompt_len, visual_tokens_in, visual_tokens_out = \
        build_trt_request_for_sample(
            sample=sample, hf_model=hf_model, processor=processor,
            image_pad_id=image_pad_id, video_pad_id=video_pad_id,
            device=args.device, dtype=torch.bfloat16, mm_payload=args.mm_payload,
            compress_method=args.compress_method,
            compress_ratio=int(args.compress_ratio),
        )
    print(f"[bench] text_prompt length (chars): {len(text_prompt)}")
    print(f"[bench] disagg params: {'present' if make_disagg is not None else 'NONE (text-only)'}")
    if args.compress_method != "none":
        print(f"[bench] compression: {args.compress_method} x{args.compress_ratio} "
              f"video tokens {visual_tokens_in} -> {visual_tokens_out}")

    # =====================================================================
    # STEP 4: parity gate — HF top-1 vs TRT top-1
    # =====================================================================
    hf_top5_ids = None
    if not args.skip_parity_gate:
        print(f"[bench] === STEP 4: HF parity baseline (top-5 next-token) ===")
        hf_top5_ids, hf_top5_vals = hf_first_token_topk(hf_model, sample, args.device, k=5)
        print(f"[bench] HF baseline top-5: ids={hf_top5_ids} logits={[round(v, 3) for v in hf_top5_vals]}")

    # =====================================================================
    # STEP 5: free HF, load TRT (both 4B models don't fit on a single 32GB)
    # =====================================================================
    print(f"[bench] === STEP 5: free HF model, load TRT-LLM ===")
    del hf_model
    torch.cuda.empty_cache()
    import gc; gc.collect()

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import KvCacheConfig

    reset_peak_mem()
    t0 = time.perf_counter()
    llm = LLM(
        model=str(src),
        tensor_parallel_size=1,
        max_batch_size=int(args.max_batch_size),
        max_seq_len=int(args.max_seq_len),
        max_num_tokens=int(args.max_seq_len),
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=float(args.free_gpu_mem_frac)),
        trust_remote_code=True,
    )
    trt_load_secs = time.perf_counter() - t0
    trt_load_peak_gb = peak_mem_gb()
    print(f"[bench] TRT loaded in {trt_load_secs:.1f}s, peak GPU mem: {trt_load_peak_gb:.2f} GB")

    # Single generate with a FRESH per-call disagg handle (mm-disagg shared
    # tensors are one-shot; a reused handle dies on the 2nd+ call). For text-only
    # (make_disagg is None) this is a plain generate.
    def _gen(sp):
        if make_disagg is not None:
            d, _hold = make_disagg()
            r = llm.generate([{"prompt": text_prompt}], sampling_params=sp,
                             disaggregated_params=d)
            del _hold
            return r
        return llm.generate([{"prompt": text_prompt}], sampling_params=sp)

    # =====================================================================
    # STEP 6: parity check (max_tokens=1)
    # =====================================================================
    parity = {"checked": False, "hf_top5": hf_top5_ids, "trt_top1": None, "passed": None}
    if not args.skip_parity_gate:
        print(f"[bench] === STEP 6: TRT top-1 (max_tokens=1) ===")
        sp_one = SamplingParams(max_tokens=1, temperature=0.0)
        out = _gen(sp_one)
        trt_top1 = list(out[0].outputs[0].token_ids)[0] if out[0].outputs[0].token_ids else None
        passed = (trt_top1 in hf_top5_ids) if hf_top5_ids else False
        parity = {"checked": True, "hf_top5": hf_top5_ids, "trt_top1": trt_top1, "passed": bool(passed)}
        print(f"[bench] TRT top-1 = {trt_top1}, in HF top-5 = {passed}")
        if not passed:
            print(f"[bench] WARN: parity gate FAIL — TRT top-1 {trt_top1} not in HF top-5 "
                  f"{hf_top5_ids}. Likely embed/mrope misalignment OR quant outlier. "
                  f"Continuing bench but flagging.")

    # =====================================================================
    # STEP 7: bench warmup
    # =====================================================================
    print(f"[bench] === STEP 7: warmup ({args.n_warmup} runs) ===")
    sp_full = SamplingParams(max_tokens=int(args.max_new_tokens), temperature=0.0)
    for _ in range(int(args.n_warmup)):
        _ = _gen(sp_full)

    # =====================================================================
    # STEP 8: bench TTFT (max_tokens=1) and full traj (max_tokens=N)
    # =====================================================================
    print(f"[bench] === STEP 8: timing ({args.n_runs} runs each) ===")
    sp_one = SamplingParams(max_tokens=1, temperature=0.0)
    reset_peak_mem()
    ttft = []
    for _ in range(int(args.n_runs)):
        d_hold = make_disagg() if make_disagg is not None else None  # fresh handle OUTSIDE timer
        t = time.perf_counter()
        _ = (llm.generate([{"prompt": text_prompt}], sampling_params=sp_one,
                          disaggregated_params=d_hold[0])
             if d_hold is not None
             else llm.generate([{"prompt": text_prompt}], sampling_params=sp_one))
        ttft.append(time.perf_counter() - t)
        del d_hold

    full = []
    for _ in range(int(args.n_runs)):
        d_hold = make_disagg() if make_disagg is not None else None  # fresh handle OUTSIDE timer
        t = time.perf_counter()
        _ = (llm.generate([{"prompt": text_prompt}], sampling_params=sp_full,
                          disaggregated_params=d_hold[0])
             if d_hold is not None
             else llm.generate([{"prompt": text_prompt}], sampling_params=sp_full))
        full.append(time.perf_counter() - t)
        del d_hold
    bench_peak_gb = peak_mem_gb()

    n_dec = max(1, int(args.max_new_tokens) - 1)
    decode = [(f - t) / n_dec for f, t in zip(full, ttft)]
    throughput = [int(args.max_new_tokens) / f for f in full]

    print(f"[bench] TTFT mean={1000*sum(ttft)/len(ttft):.1f}ms  "
          f"full mean={1000*sum(full)/len(full):.1f}ms  "
          f"throughput mean={sum(throughput)/len(throughput):.1f} tok/s  "
          f"peak GPU mem={bench_peak_gb:.2f}GB")

    # =====================================================================
    # STEP 9: optional L2 eval
    # =====================================================================
    l2_summary = None
    # F6: in-process L2 eval requires reloading HF AFTER TRT load. On 32GB cards
    # the 4B HF + 4B TRT both live on GPU and OOM. Default to SKIP the in-proc
    # path and direct the user to run L2 via the standalone planning_eval
    # pipeline. Set --no-l2-skip-inproc to override (only safe on >40GB cards).
    if int(args.l2_n) > 0 and args.mm_payload == "real" and not args.l2_skip_inproc:
        print(f"[bench] === STEP 9: L2 trajectory eval on {args.l2_n} val samples (in-proc) ===")
        # Reload HF for vision precompute during L2 eval
        from transformers import AutoModelForImageTextToText as _AM
        hf_for_l2 = _AM.from_pretrained(
            str(src), torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        ).to(args.device).eval()
        try:
            l2_summary = eval_l2(
                llm=llm, hf_model=hf_for_l2, processor=processor,
                image_pad_id=image_pad_id, video_pad_id=video_pad_id,
                device=args.device, dtype=torch.bfloat16,
                n_samples=int(args.l2_n), max_new_tokens=int(args.max_new_tokens),
                mm_payload=args.mm_payload, config_yaml=args.config_yaml,
            )
        finally:
            del hf_for_l2
            torch.cuda.empty_cache()
            gc.collect()
    elif int(args.l2_n) > 0 and args.mm_payload == "real" and args.l2_skip_inproc:
        print(f"[bench] === STEP 9 SKIPPED: in-process L2 disabled (default) ===")
        print(f"[bench] To compute L2 for this ckpt, run separately:")
        print(f"[bench]   /usr/bin/python3 scripts/planning_eval.py \\")
        print(f"[bench]     --ckpt {src} --max-samples {int(args.l2_n)} \\")
        print(f"[bench]     --output deploy/trt_bench/L2_$(basename {src}).json")
        print(f"[bench] (override with --no-l2-skip-inproc on a >40GB card)")
    elif int(args.l2_n) > 0 and args.mm_payload != "real":
        print(f"[bench] Skipping L2 eval (mm_payload={args.mm_payload}; only 'real' is meaningful)")

    # =====================================================================
    # STEP 10: write JSON in same schema as B5pp_trt_qwen3vl_multimodal_bf16.json
    # =====================================================================
    results = {
        "ckpt": str(src),
        "backend": f"TRT-LLM 1.3.0rc15 PyTorch backend + mm-disagg "
                   f"(HF vision pre-compute + injected handles, precision={args.precision})",
        "model": "Qwen3-VL-4B B.5'' v2 VLA (1-cam + HD-map + bbox + ego)",
        "modality_real": (
            "video(1cam x 4f native) + image(HD-map BEV) + text(bbox/ego/prompt)"
            if args.mm_payload == "real"
            else "text-only (LM-only timing; no images/videos)"
        ),
        "dtype": args.precision,
        "precision": args.precision,
        "mm_payload": args.mm_payload,
        "video_mm_tokens": None,  # filled if real
        "image_mm_tokens": None,
        "embed_dim": None,
        # F7: compression bookkeeping (visual_tokens_in/out are video-only)
        "compress_method": args.compress_method,
        "compress_ratio": int(args.compress_ratio),
        "visual_tokens_in": int(visual_tokens_in),
        "visual_tokens_out": int(visual_tokens_out),
        "prompt_len_expanded": int(eff_prompt_len),
        "max_new_tokens": int(args.max_new_tokens),
        "n_warmup": int(args.n_warmup),
        "n_runs": int(args.n_runs),
        "parity_gate": parity,
        "TTFT_ms": {
            "mean": 1000 * sum(ttft) / len(ttft),
            "p50": 1000 * percentile(ttft, 50),
            "p99": 1000 * percentile(ttft, 99),
        },
        "per_token_decode_ms": {
            "mean": 1000 * sum(decode) / len(decode),
            "p50": 1000 * percentile(decode, 50),
            "p99": 1000 * percentile(decode, 99),
        },
        "full_traj_ms": {
            "mean": 1000 * sum(full) / len(full),
            "p50": 1000 * percentile(full, 50),
            "p99": 1000 * percentile(full, 99),
        },
        "throughput_toks_per_s": {
            "mean": sum(throughput) / len(throughput),
            "p50": percentile(throughput, 50),
        },
        "gpu_mem_gb": {
            "trt_load_peak": trt_load_peak_gb,
            "bench_peak": bench_peak_gb,
        },
        "l2_summary": l2_summary,
    }

    # Fill MM token counts if we computed embeds
    if args.mm_payload == "real":
        # Re-derive from sample (cheap — no model call)
        try:
            from phase1_5_vision_embeds import assemble_mm_embedding  # noqa
        except Exception:
            pass
        # Use shape inference from earlier was fine; cheaper to redo from sample meta.
        # The mm token counts equal the post-merger token counts; for B.5'' v2
        # (1-cam) these are 2800 (video) + 121 (image) = 2921.
        try:
            img_thw = sample.get("image_grid_thw")
            vid_thw = sample.get("video_grid_thw")
            spatial_merge = 2  # Qwen3-VL default
            def _mm_count(thw):
                if thw is None: return 0
                t = thw if thw.dim() == 2 else thw.unsqueeze(0)
                # post-merger count: T * (H // sm) * (W // sm)
                tot = 0
                for row in t.tolist():
                    T, H, W = row
                    tot += T * (H // spatial_merge) * (W // spatial_merge)
                return tot
            results["video_mm_tokens"] = _mm_count(vid_thw)
            results["image_mm_tokens"] = _mm_count(img_thw)
        except Exception as e:
            print(f"[bench] couldn't compute mm token counts: {e}")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[bench] === DONE === saved → {out_path}")
    print(f"  TTFT mean: {results['TTFT_ms']['mean']:.1f} ms")
    print(f"  decode mean: {results['per_token_decode_ms']['mean']:.2f} ms/tok")
    print(f"  full mean: {results['full_traj_ms']['mean']:.1f} ms / {args.max_new_tokens} tok")
    print(f"  throughput: {results['throughput_toks_per_s']['mean']:.1f} tok/s")
    print(f"  peak GPU mem: {results['gpu_mem_gb']['bench_peak']:.2f} GB")
    if l2_summary:
        print(f"  L2_avg: {l2_summary['L2_avg_mean']:.4f} ({l2_summary['n_samples_evaluated']} samples)")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
