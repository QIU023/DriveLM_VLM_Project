#!/usr/bin/env /venv/trt_llm/bin/python
"""B.5'' v2 FULL-PIPELINE quantized bench (end-to-end, per-stage timed).

Unlike bench_trt.py (which precomputes vision embeds ONCE on HF, untimed, and
only times the LM), this bench measures the COMPLETE deployed path per sample,
with each stage timed inside the end-to-end window:

    raw video frames + HD-map image
      -> [vision]   HF Qwen3-VL ViT, GPU (bf16 OR modelopt fp8/nvfp4) .. vision_ms
      -> [compress] FasterVLM token pruning on VIDEO tokens ......... compress_ms
      -> [prefill]  LM in TRT (bf16/fp8/nvfp4), embedding injection . prefill_ms
      -> [decode]   greedy-decode full 14-token trajectory in TRT ... decode_ms
      => full_traj_ms = vision + compress + prefill + decode

WHY this architecture (see FULL_PIPELINE_FEASIBILITY.md for source quotes):
  - TRT-LLM 1.3.0rc15 CAN run the Qwen3-VL vision tower in-engine, BUT
    Qwen3VisionModelBase.forward raises "Currently only support single modality
    per request" (modeling_qwen3vl.py:920). B.5'' sends video (camera) AND image
    (HD-map) in ONE request -> in-engine native path is blocked.
  - The ViT IS quantizable. It is Linear+attention, and modelopt quantizes it
    fine. It only stayed bf16 before because (a) our PTQ scripts quantized the
    language_model ONLY, never model.model.visual, and (b) TRT-LLM's bundled
    Qwen3-VL class STRIPS the ViT quant config on load (modeling_qwen3vl.py:823 —
    a framework integration choice, NOT a TRT capability limit). The deployable
    route is the standard automotive "vision-as-separate-engine" pattern:
    modelopt-quantize model.visual, run it as its own quantized module/engine,
    and feed pruned embeddings to the LM engine (exactly the embedding-injection
    seam this bench uses for the LM).
  - So for fp8/nvfp4 this bench NOW quantizes model.model.visual via modelopt
    (quant_vision.py). The fake-quant ViT is accuracy-faithful (reported L2
    reflects quantized vision+LM). LATENCY of the fake-quant ViT is NOT a vision
    speedup (extra de/quant ops over the same matmul); a real fp8 ViT engine is
    the deploy artifact and is labeled "pending" in the JSON. bf16 keeps the ViT
    bf16 (correct for bf16).

The JSON schema documents which stages are quantized vs bf16 (`stages` block).

Usage:
    /venv/trt_llm/bin/python bench_full_pipeline.py \
        --precision {bf16,fp8,nvfp4} \
        [--compress-method fastervlm --compress-ratio 4] \
        [--n-warmup 2] [--n-runs 3] [--max-new-tokens 14] \
        [--l2-n 30] \
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
    DEFAULT_PARENT,
    EXPECTED_VIDEO_TOKENS,
    add_project_paths,
    build_calib_dataset,
    peak_mem_gb,
    reset_peak_mem,
)

# Reuse the heavily-validated request-building helpers from bench_trt.py rather
# than re-implementing the mm-disagg block math. We re-split the work so the
# vision forward and the compress op are timed SEPARATELY.
import bench_trt as B  # noqa: E402


def percentile(values, p):
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    f = int(k); c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f) if f != c else s[f]


def _mean(xs):
    return float(sum(xs) / len(xs)) if xs else float("nan")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Full-pipeline quantized bench for B.5'' v2 Qwen3-VL-4B 1-cam",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt", default=None,
                   help="LM ckpt dir. Default: derived from --precision "
                        "(quant_fp8 / quant_nvfp4 / final).")
    p.add_argument("--vision-ckpt", default=DEFAULT_CKPT,
                   help="HF ckpt for the (bf16) vision tower. Default: final/.")
    p.add_argument("--precision", choices=["bf16", "fp8", "nvfp4"], required=True)
    p.add_argument("--n-warmup", type=int, default=2)
    p.add_argument("--n-runs", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=14,
                   help="6 waypoints x 2 dims + 2 boundary = 14 traj tokens")
    p.add_argument("--l2-n", type=int, default=30,
                   help="val samples for L2 vs HF bf16 reference (0 = skip)")
    p.add_argument("--compress-method",
                   choices=["none", "fastervlm", "prumerge", "pyramiddrop", "avg_pool"],
                   default="fastervlm")
    p.add_argument("--compress-ratio", type=int, default=4)
    p.add_argument("--quant-vision", choices=["auto", "yes", "no"], default="auto",
                   help="Quantize model.model.visual via modelopt at --precision. "
                        "auto = quantize iff precision != bf16 (default). "
                        "The ViT IS quantizable; bf16-only was a fixable PTQ/framework "
                        "choice, not a TRT limit (see quant_vision.py header).")
    p.add_argument("--vision-calib-n", type=int, default=8,
                   help="real multimodal samples for ViT activation calibration")
    p.add_argument("--config-yaml", default=DEFAULT_CONFIG_YAML)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-seq-len", type=int, default=12288)
    p.add_argument("--max-batch-size", type=int, default=1)
    p.add_argument("--free-gpu-mem-frac", type=float, default=0.35,
                   help="Lower because HF vision tower lives alongside TRT LM.")
    p.add_argument("--out", default=None)
    return p.parse_args()


def _derive_lm_ckpt(args) -> Path:
    if args.ckpt is not None:
        return Path(args.ckpt)
    parent = Path(DEFAULT_PARENT)
    if args.precision == "fp8":
        cand = parent / "quant_fp8"
    elif args.precision == "nvfp4":
        cand = parent / "quant_nvfp4"
    else:
        cand = parent / "final"
    if not cand.is_dir():
        print(f"[fullpipe] WARN: {cand} missing; falling back to final/")
        cand = parent / "final"
    return cand


def _derive_out(args) -> str:
    if args.out is not None:
        return args.out
    cm = args.compress_method
    cr = int(args.compress_ratio)
    suffix = f"_compress_{cm}{cr}" if cm != "none" else "_nocompress"
    name = f"B5pp_fullpipe_{args.precision}{suffix}.json"
    return str(Path(DEFAULT_BENCH_OUT_DIR) / name)


# ---------------------------------------------------------------------------
# Per-stage timed vision+compress. Mirrors bench_trt.build_trt_request_for_sample
# but splits the vision forward and the FasterVLM op into individually-timed
# sub-steps and returns the timings. The mm-disagg block math + handle minting
# is delegated back to bench_trt helpers to stay in lockstep with the validated
# path. We pass the embeddings we already computed so the vision forward isn't
# done twice.
# ---------------------------------------------------------------------------

def vision_and_compress_timed(*, sample, hf_model, processor, image_pad_id,
                              video_pad_id, device, dtype, compress_method,
                              compress_ratio):
    """Run vision tower (timed) then FasterVLM (timed), then build the TRT
    request payload (make_disagg factory). Returns:
        (text_prompt, make_disagg, eff_prompt_len, stats)
    where stats = {vision_ms, compress_ms, n_video_in, n_video_out, n_image}.
    """
    import torch
    from phase1_5_vision_embeds import assemble_mm_embedding

    # ---- STAGE 1: vision tower forward (TIMED) ----
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    embed_result = assemble_mm_embedding(
        hf_model=hf_model,
        pixel_values=sample.get("pixel_values"),
        image_grid_thw=sample.get("image_grid_thw"),
        pixel_values_videos=sample.get("pixel_values_videos"),
        video_grid_thw=sample.get("video_grid_thw"),
        modality_order=("video", "image"),
    )
    torch.cuda.synchronize()
    vision_ms = 1000.0 * (time.perf_counter() - t0)

    mm_embedding_full = embed_result["mm_embedding"].to(device, dtype=dtype)
    n_video = embed_result["video_mm_tokens"]
    n_image = embed_result["image_mm_tokens"]
    assert n_video >= int(EXPECTED_VIDEO_TOKENS * 0.9), (
        f"video tokens {n_video} << expected {EXPECTED_VIDEO_TOKENS}")

    # ---- STAGE 2: FasterVLM compress on VIDEO tokens (TIMED) ----
    # We measure ONLY the compress op here; the surrounding prompt/grid bookkeeping
    # is request-build cost, not a deployed inference stage, so it's excluded from
    # compress_ms. We replicate the exact run-segmented compression from
    # bench_trt and time the compress_visual_tokens calls.
    prompt_len = int(sample["_meta_prompt_len"])
    video_embed = mm_embedding_full[:n_video].contiguous()
    image_embed = mm_embedding_full[n_video:].contiguous()
    n_video_in = int(n_video)
    n_video_out = int(n_video)
    compress_ms = 0.0

    if compress_method != "none" and int(compress_ratio) > 1:
        add_project_paths()
        from visual_compress import compress_visual_tokens

        prompt_ids_1d = sample["input_ids"][:prompt_len]
        runs = B._find_pad_runs(prompt_ids_1d, video_pad_id)
        run_lens = [e - s for (s, e) in runs]
        assert sum(run_lens) == n_video, (
            f"prompt video_pad total {sum(run_lens)} != n_video {n_video}")
        ms = int(getattr(hf_model.config.vision_config, "spatial_merge_size", 2))

        offset = 0
        comp_chunks = []
        per_run_comp = []
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for rlen in run_lens:
            seg = video_embed[offset:offset + rlen]
            offset += rlen
            grid_synth = torch.tensor([[1, 1, rlen]], device=seg.device,
                                      dtype=torch.int64)
            comp, _ = compress_visual_tokens(
                seg, grid_synth, compress_method, int(compress_ratio))
            comp_chunks.append(comp.contiguous())
            per_run_comp.append(int(comp.shape[0]))
        video_embed = torch.cat(comp_chunks, dim=0).contiguous()
        torch.cuda.synchronize()
        compress_ms = 1000.0 * (time.perf_counter() - t0)

        n_video_out = int(video_embed.shape[0])
        # --- prompt / grid / mask rebuild (NOT timed; request-build bookkeeping) ---
        new_video_rows = [list(B._factor_grid_simple(c, ms)) for c in per_run_comp]
        vid_thw = sample.get("video_grid_thw")
        new_video_thw = torch.tensor(new_video_rows, dtype=vid_thw.dtype,
                                     device=vid_thw.device)
        new_prompt_ids, new_prompt_len = B._trim_video_pad_runs_1d(
            prompt_ids_1d, video_pad_id, per_run_comp)
        rest = sample["input_ids"][prompt_len:]
        full_new_ids = torch.cat([new_prompt_ids, rest], dim=0)
        new_attn = torch.cat([
            torch.ones(new_prompt_len, dtype=sample["attention_mask"].dtype,
                       device=sample["attention_mask"].device),
            sample["attention_mask"][prompt_len:],
        ], dim=0)
        new_mtt = torch.zeros_like(full_new_ids)
        for k, t in enumerate(full_new_ids[:new_prompt_len].tolist()):
            if t == video_pad_id:
                new_mtt[k] = 2
            elif t == image_pad_id:
                new_mtt[k] = 1
        sample = dict(sample)
        sample["input_ids"] = full_new_ids
        sample["attention_mask"] = new_attn
        sample["mm_token_type_ids"] = new_mtt
        sample["video_grid_thw"] = new_video_thw
        sample["_meta_prompt_len"] = new_prompt_len
        prompt_len = new_prompt_len
        n_video = n_video_out
        mm_embedding_full = torch.cat([video_embed, image_embed], dim=0).contiguous()

    # ---- build TRT request payload (handle minting / mrope) from the embeds ----
    tok = processor.tokenizer
    text_prompt, unexpanded_ids = B.build_unexpanded_prompt(
        sample["input_ids"], prompt_len, tok, image_pad_id, video_pad_id)
    n_img_blocks, n_vid_blocks = B.count_orig_mm_blocks(
        sample["input_ids"], prompt_len, image_pad_id, video_pad_id)
    if n_vid_blocks == 0 or n_video % n_vid_blocks != 0:
        raise RuntimeError(f"video blocks {n_vid_blocks} !| n_video {n_video}")
    if n_img_blocks == 0 or n_image % n_img_blocks != 0:
        raise RuntimeError(f"image blocks {n_img_blocks} !| n_image {n_image}")
    video_chunk = n_video // n_vid_blocks
    image_chunk = n_image // n_img_blocks
    video_chunks = [video_embed[i * video_chunk:(i + 1) * video_chunk].contiguous()
                    for i in range(n_vid_blocks)]
    image_chunks = [image_embed[i * image_chunk:(i + 1) * image_chunk].contiguous()
                    for i in range(n_img_blocks)]

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
    _mm_cpu = [t.detach().cpu().contiguous() for t in mm_handles_tensors]

    from phase2_mrope_config import build_mrope_config
    from tensorrt_llm._torch.shared_tensor import SharedTensorContainer
    from tensorrt_llm.disaggregated_params import DisaggregatedParams

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

    stats = {
        "vision_ms": vision_ms,
        "compress_ms": compress_ms,
        "n_video_in": n_video_in,
        "n_video_out": n_video_out,
        "n_image": int(n_image),
    }
    return text_prompt, make_disagg, prompt_len, stats


# ---------------------------------------------------------------------------
# L2 vs HF bf16 reference, full quantized pipeline (vision bf16 + LM TRT).
# ---------------------------------------------------------------------------

_HORIZONS = (1.0, 2.0, 3.0)
_HORIZON_IDX = (1, 3, 5)  # 0-indexed waypoint at t = 1/2/3 s (mirrors planning_eval)


def _find_traj_block(token_ids, traj_start_id, traj_end_id):
    try:
        i0 = token_ids.index(traj_start_id)
    except ValueError:
        return []
    try:
        i1 = token_ids.index(traj_end_id, i0 + 1)
    except ValueError:
        i1 = len(token_ids)
    return token_ids[i0:i1 + 1]


def _decode_waypoints(generated_ids, traj_tok, num_waypoints):
    """Mirror of planning_eval.decode_waypoints (inlined to avoid importing
    planning_eval, which pulls in _planning_metric -> skimage, absent in the
    trt_llm venv)."""
    import numpy as np
    block = _find_traj_block(generated_ids, traj_tok.cfg.traj_start_id,
                             traj_tok.cfg.traj_end_id)
    wp = traj_tok.decode(block) if block else traj_tok.decode(generated_ids)
    out = np.zeros((num_waypoints, 2), dtype=np.float32)
    n = min(num_waypoints, wp.shape[0])
    if n > 0:
        out[:n] = wp[:n]
    return out


def _l2_temavg(pred, gt, valid):
    """Mirror of planning_eval.l2_temavg (VAD-style TemAvg)."""
    import numpy as np
    out = {}
    for hi, hs in zip(_HORIZON_IDX, _HORIZONS):
        sl = slice(0, hi + 1)
        l2 = np.sqrt(((pred[sl] - gt[sl]) ** 2).sum(axis=-1))
        m = valid[sl]
        out[f"L2_{int(hs)}s"] = (float((l2 * m).sum() / m.sum())
                                 if m.sum() >= 1e-6 else float("nan"))
    vals = [out[k] for k in ("L2_1s", "L2_2s", "L2_3s") if not math.isnan(out[k])]
    out["L2_avg"] = float(np.mean(vals)) if vals else float("nan")
    return out


def eval_l2_fullpipe(*, llm, hf_model, processor, image_pad_id, video_pad_id,
                     device, dtype, n_samples, max_new_tokens, config_yaml,
                     compress_method, compress_ratio):
    import numpy as np
    import torch
    from tensorrt_llm import SamplingParams

    add_project_paths()
    decode_waypoints = _decode_waypoints
    l2_temavg = _l2_temavg
    from trajectory_tokenizer import TrajectoryTokenizer, TrajectoryTokenizerConfig

    val_ds = build_calib_dataset(processor=processor, n_samples=n_samples,
                                 split="val", config_yaml=config_yaml)
    n_total = len(val_ds)
    print(f"[l2] evaluating full-pipeline L2 on {n_total} val samples")
    traj_cfg = TrajectoryTokenizerConfig(num_waypoints=val_ds.num_future)
    traj_tok = TrajectoryTokenizer(traj_cfg)
    sp = SamplingParams(max_tokens=max_new_tokens, temperature=0.0)

    per_sample = []
    for i in range(n_total):
        try:
            sample = val_ds[i]
            text_prompt, make_disagg, _, _ = vision_and_compress_timed(
                sample=sample, hf_model=hf_model, processor=processor,
                image_pad_id=image_pad_id, video_pad_id=video_pad_id,
                device=device, dtype=dtype, compress_method=compress_method,
                compress_ratio=compress_ratio)
            disagg, _hold = make_disagg()
            out = llm.generate([{"prompt": text_prompt}], sampling_params=sp,
                               disaggregated_params=disagg)
            del _hold
        except Exception as e:
            print(f"[l2]   sample {i} failed: {e}; skip")
            continue
        token_ids = list(out[0].outputs[0].token_ids)
        pred_wp = decode_waypoints(token_ids, traj_tok, val_ds.num_future)
        gt_wp = sample["_meta_waypoints"].cpu().numpy()
        valid = sample["_meta_valid_mask"].cpu().numpy()
        m = l2_temavg(pred_wp, gt_wp, valid)
        per_sample.append({k: m.get(k, float("nan"))
                           for k in ("L2_avg", "L2_1s", "L2_2s", "L2_3s")})
        if (i + 1) % 10 == 0 or (i + 1) == n_total:
            run = np.nanmean([s["L2_avg"] for s in per_sample])
            print(f"[l2]   {i+1}/{n_total}  L2_avg(running)={run:.4f}")

    def _m(key):
        vals = [s[key] for s in per_sample if not math.isnan(s.get(key, float("nan")))]
        return float(sum(vals) / len(vals)) if vals else float("nan")

    return {
        "n_samples_evaluated": len(per_sample),
        "L2_avg_mean": _m("L2_avg"), "L2_1s_mean": _m("L2_1s"),
        "L2_2s_mean": _m("L2_2s"), "L2_3s_mean": _m("L2_3s"),
    }


def main() -> int:
    args = parse_args()
    lm_ckpt = _derive_lm_ckpt(args)
    vision_ckpt = Path(args.vision_ckpt)
    out_path = Path(_derive_out(args))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not lm_ckpt.is_dir():
        print(f"[fullpipe] FATAL: LM ckpt not found: {lm_ckpt}", file=sys.stderr)
        return 2
    if not vision_ckpt.is_dir():
        print(f"[fullpipe] FATAL: vision ckpt not found: {vision_ckpt}", file=sys.stderr)
        return 2

    print(f"[fullpipe] LM ckpt     = {lm_ckpt}  (precision={args.precision})")
    _vq_planned = (args.quant_vision == "yes"
                   or (args.quant_vision == "auto" and args.precision != "bf16"))
    print(f"[fullpipe] vision ckpt = {vision_ckpt}  "
          f"({'modelopt ' + args.precision + ' (fake-quant)' if _vq_planned else 'bf16'})")
    print(f"[fullpipe] compress    = {args.compress_method} x{args.compress_ratio}")
    print(f"[fullpipe] out         = {out_path}")

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    add_project_paths()

    # === Load HF vision tower (bf16). Stays resident the whole bench (it IS
    #     part of the deployed pipeline now, not a one-shot precompute). ===
    print(f"[fullpipe] loading HF (vision tower, bf16) ...")
    hf_model = AutoModelForImageTextToText.from_pretrained(
        str(vision_ckpt), torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(args.device).eval()
    processor = AutoProcessor.from_pretrained(str(vision_ckpt))
    tok = processor.tokenizer
    image_pad_id = tok.convert_tokens_to_ids("<|image_pad|>")
    video_pad_id = tok.convert_tokens_to_ids("<|video_pad|>")
    print(f"[fullpipe] image_pad={image_pad_id} video_pad={video_pad_id}")

    # === Build a bench sample (val[0]) ===
    val_ds_one = build_calib_dataset(processor=processor, n_samples=2,
                                     split="val", config_yaml=args.config_yaml)
    sample0 = val_ds_one[0]

    # === QUANTIZE THE VISION TOWER (model.model.visual) via modelopt ===
    # The ViT IS quantizable. For fp8/nvfp4 we modelopt-quantize it in-place
    # (fake-quant, accuracy-faithful). bf16 keeps it bf16. See quant_vision.py.
    do_vis_quant = (args.quant_vision == "yes"
                    or (args.quant_vision == "auto" and args.precision != "bf16"))
    vision_quant_meta = {"quantized": False, "precision": "bf16", "cfg": None,
                         "n_calib_done": 0}
    if do_vis_quant:
        from quant_vision import quantize_vision_inplace
        # Calibrate the ViT on train samples (same split PTQ uses), but only a
        # few — ViT activation stats converge fast and we want the bench to boot
        # quickly. Build a small train calib set.
        vis_calib = build_calib_dataset(
            processor=processor, n_samples=int(args.vision_calib_n),
            split="train", config_yaml=args.config_yaml)
        vision_quant_meta = quantize_vision_inplace(
            hf_model=hf_model, calib_ds=vis_calib, precision=args.precision,
            n_calib=int(args.vision_calib_n), device=args.device, verbose=True)
        print(f"[fullpipe] vision quant: {vision_quant_meta}")
    else:
        print(f"[fullpipe] vision tower NOT quantized (precision={args.precision}, "
              f"quant_vision={args.quant_vision}) -> stays bf16")

    # === Load TRT LM engine (alongside the bf16 vision tower) ===
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import KvCacheConfig
    reset_peak_mem()
    print(f"[fullpipe] loading TRT LM ...")
    t0 = time.perf_counter()
    llm = LLM(
        model=str(lm_ckpt), tensor_parallel_size=1,
        max_batch_size=int(args.max_batch_size),
        max_seq_len=int(args.max_seq_len), max_num_tokens=int(args.max_seq_len),
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=float(args.free_gpu_mem_frac)),
        trust_remote_code=True,
    )
    trt_load_secs = time.perf_counter() - t0
    print(f"[fullpipe] TRT loaded in {trt_load_secs:.1f}s")

    sp_full = SamplingParams(max_tokens=int(args.max_new_tokens), temperature=0.0)
    sp_one = SamplingParams(max_tokens=1, temperature=0.0)

    # === Warmup. CRITICAL: warm up BOTH the max_tokens=1 (prefill) AND the
    #     max_tokens=N (decode) generate paths — each triggers its own CUDA
    #     graph capture on first call, and an uncaptured path pollutes run 1
    #     of the timed loop with a multi-hundred-ms capture spike. ===
    print(f"[fullpipe] warmup ({args.n_warmup} x {{prefill,decode}}) ...")
    for _ in range(int(args.n_warmup)):
        _tp, _md, _pl, _st = vision_and_compress_timed(
            sample=sample0, hf_model=hf_model, processor=processor,
            image_pad_id=image_pad_id, video_pad_id=video_pad_id,
            device=args.device, dtype=torch.bfloat16,
            compress_method=args.compress_method,
            compress_ratio=int(args.compress_ratio))
        d1, h1 = _md()
        llm.generate([{"prompt": _tp}], sampling_params=sp_one,
                     disaggregated_params=d1)
        del h1
        d2, h2 = _md()
        llm.generate([{"prompt": _tp}], sampling_params=sp_full,
                     disaggregated_params=d2)
        del h2

    # === Timed runs: per-stage breakdown + end-to-end ===
    print(f"[fullpipe] timing ({args.n_runs}) ...")
    reset_peak_mem()
    vision_t, compress_t, prefill_t, decode_t, full_t = [], [], [], [], []
    last_stats = None
    n_dec = max(1, int(args.max_new_tokens) - 1)
    for _ in range(int(args.n_runs)):
        torch.cuda.synchronize(); e2e0 = time.perf_counter()
        # vision + compress (timed internally)
        _tp, make_disagg, _pl, st = vision_and_compress_timed(
            sample=sample0, hf_model=hf_model, processor=processor,
            image_pad_id=image_pad_id, video_pad_id=video_pad_id,
            device=args.device, dtype=torch.bfloat16,
            compress_method=args.compress_method,
            compress_ratio=int(args.compress_ratio))
        last_stats = st
        # prefill (TTFT, max_tokens=1)
        d1, h1 = make_disagg()
        torch.cuda.synchronize(); tpf = time.perf_counter()
        llm.generate([{"prompt": _tp}], sampling_params=sp_one,
                     disaggregated_params=d1)
        torch.cuda.synchronize(); prefill_ms = 1000.0 * (time.perf_counter() - tpf)
        del h1
        # full decode (max_tokens=N)
        d2, h2 = make_disagg()
        torch.cuda.synchronize(); tfd = time.perf_counter()
        llm.generate([{"prompt": _tp}], sampling_params=sp_full,
                     disaggregated_params=d2)
        torch.cuda.synchronize(); lm_full_ms = 1000.0 * (time.perf_counter() - tfd)
        del h2
        torch.cuda.synchronize()
        decode_ms = max(0.0, lm_full_ms - prefill_ms)
        # end-to-end = vision + compress + (LM prefill + decode)
        full_ms = st["vision_ms"] + st["compress_ms"] + lm_full_ms
        vision_t.append(st["vision_ms"]); compress_t.append(st["compress_ms"])
        prefill_t.append(prefill_ms); decode_t.append(decode_ms); full_t.append(full_ms)

    bench_peak_gb = peak_mem_gb()
    # decode_t holds TOTAL decode time (all n_dec tokens) per run, already.
    print(f"[fullpipe] vision={_mean(vision_t):.1f}ms compress={_mean(compress_t):.2f}ms "
          f"prefill={_mean(prefill_t):.1f}ms decode_total={_mean(decode_t):.1f}ms "
          f"full={_mean(full_t):.1f}ms peak={bench_peak_gb:.2f}GB")

    # === L2 vs HF bf16 reference (full quantized pipeline) ===
    l2_summary = None
    if int(args.l2_n) > 0:
        print(f"[fullpipe] L2 eval on {args.l2_n} val samples ...")
        try:
            l2_summary = eval_l2_fullpipe(
                llm=llm, hf_model=hf_model, processor=processor,
                image_pad_id=image_pad_id, video_pad_id=video_pad_id,
                device=args.device, dtype=torch.bfloat16,
                n_samples=int(args.l2_n), max_new_tokens=int(args.max_new_tokens),
                config_yaml=args.config_yaml,
                compress_method=args.compress_method,
                compress_ratio=int(args.compress_ratio))
        except Exception as e:
            print(f"[fullpipe] L2 eval failed: {e}")
            l2_summary = {"error": str(e)}

    # === JSON output with explicit per-stage quantization labels ===
    lm_quantized = args.precision != "bf16"
    vis_quantized = bool(vision_quant_meta.get("quantized"))
    vis_dtype = args.precision if vis_quantized else "bf16"
    if vis_quantized:
        vision_stage = {
            "runtime": "HF torch GPU (modelopt fake-quant)",
            "dtype": vis_dtype,
            "quantized": True,
            "quant_method": f"modelopt {vision_quant_meta.get('cfg')}",
            "quant_target": "model.model.visual (ViT: Linear + attention)",
            "quant_calib_samples": int(vision_quant_meta.get("n_calib_done", 0)),
            "latency_basis": ("fake-quant (accuracy-faithful); real fp8/nvfp4 ViT "
                              "engine pending. vision_ms here is NOT a quantized-ViT "
                              "speedup — fake-quant runs the same matmuls plus extra "
                              "de/quant ops, so it is ~bf16-equivalent latency. The "
                              "deployable speedup needs modelopt-export + TRT ViT "
                              "engine (vision-as-separate-engine pattern)."),
            "accuracy_faithful": True,
            "note": ("The ViT IS quantizable. It stayed bf16 in the prior run only "
                     "because (a) quant_fp8.py/quant_nvfp4.py quantized the "
                     "language_model ONLY, never model.model.visual, and (b) "
                     "TRT-LLM's bundled Qwen3-VL class strips the ViT quant config "
                     "on load (modeling_qwen3vl.py:823) — a framework integration "
                     "choice, NOT a TRT capability limit. Here we modelopt-quantize "
                     "model.model.visual directly (vision-as-separate-engine route)."),
            "timed": True,
        }
    else:
        vision_stage = {
            "runtime": "HF torch GPU",
            "dtype": "bf16",
            "quantized": False,
            "reason_not_quantized": ("precision=bf16 -> vision correctly stays bf16. "
                                     "(For fp8/nvfp4 the ViT IS quantized via modelopt; "
                                     "see quant_vision.py.)"),
            "timed": True,
        }
    results = {
        "ckpt_lm": str(lm_ckpt),
        "ckpt_vision": str(vision_ckpt),
        "model": "Qwen3-VL-4B B.5'' v2 VLA (1-cam + HD-map + bbox + ego)",
        "precision": args.precision,
        "backend": (f"FULL PIPELINE: vision(HF {vis_dtype}"
                    f"{', modelopt fake-quant' if vis_quantized else ''}, GPU, timed) -> "
                    f"{args.compress_method}x{args.compress_ratio}(timed) -> "
                    f"LM(TRT-LLM 1.3.0rc15, {args.precision}, embedding-injection)"),
        "architecture": (f"vision_{vis_dtype}_GPU"
                         f"{'_modelopt_quant' if vis_quantized else ''} -> "
                         f"fastervlm_compress -> LM_TRT_embedding_injection"),
        "vision_quant": vision_quant_meta,
        "native_vision_in_trt": False,
        "native_vision_in_trt_reason": (
            "Two SEPARATE facts. (1) The native in-engine multimodal path is "
            "blocked for THIS model because TRT-LLM 1.3.0rc15 "
            "Qwen3VisionModelBase.forward (modeling_qwen3vl.py:920) raises "
            "ValueError('Currently only support single modality per request') and "
            "B.5'' sends video+image in one request. (2) The ViT being bf16 in the "
            "in-engine path is NOT a TRT capability limit and NOT 'unquantizable by "
            "design' — the bundled Qwen3-VL class STRIPS the ViT quant config on "
            "load (modeling_qwen3vl.py:823), a framework integration choice. The "
            "ViT IS quantizable; the deployable route is the automotive "
            "vision-as-separate-engine pattern: modelopt-quantize model.visual, run "
            "it as its own quantized module/engine, feed pruned embeddings to the LM "
            "engine. That is what this bench's vision stage does for fp8/nvfp4. See "
            "FULL_PIPELINE_FEASIBILITY.md."),
        "stages": {
            "vision":   vision_stage,
            "compress": {"runtime": "torch GPU", "method": args.compress_method,
                         "ratio": int(args.compress_ratio), "quantized": False,
                         "timed": True},
            "prefill":  {"runtime": "TRT-LLM", "dtype": args.precision,
                         "quantized": lm_quantized, "timed": True},
            "decode":   {"runtime": "TRT-LLM", "dtype": args.precision,
                         "quantized": lm_quantized, "timed": True},
        },
        "compress_method": args.compress_method,
        "compress_ratio": int(args.compress_ratio),
        "visual_tokens_in": last_stats["n_video_in"] if last_stats else None,
        "visual_tokens_out": last_stats["n_video_out"] if last_stats else None,
        "image_mm_tokens": last_stats["n_image"] if last_stats else None,
        "max_new_tokens": int(args.max_new_tokens),
        "n_warmup": int(args.n_warmup),
        "n_runs": int(args.n_runs),
        "stage_ms": {
            "vision":  {"mean": _mean(vision_t),  "p50": percentile(vision_t, 50)},
            "compress": {"mean": _mean(compress_t), "p50": percentile(compress_t, 50)},
            "prefill": {"mean": _mean(prefill_t), "p50": percentile(prefill_t, 50)},
            "decode_total": {"mean": _mean(decode_t),
                             "per_token_mean": _mean(decode_t) / n_dec,
                             "note": f"total decode time for {n_dec} tokens "
                                     f"(= LM full generate - prefill TTFT)"},
        },
        "full_traj_ms": {
            "mean": _mean(full_t), "p50": percentile(full_t, 50),
            "p99": percentile(full_t, 99),
            "note": "end-to-end = vision + compress + LM(prefill+decode)",
        },
        "gpu_mem_gb": {
            "bench_peak": bench_peak_gb,
            "note": (f"peak across vision tower ({vis_dtype}"
                     f"{', modelopt fake-quant' if vis_quantized else ''}) + TRT LM "
                     f"engine resident together. NOTE: fake-quant vision holds extra "
                     f"scale/observer tensors, so peak is not a quantized-ViT memory "
                     f"saving; a real fp8 ViT engine would reduce ViT resident mem."),
        },
        "l2_summary": l2_summary,
        "l2_note": (
            f"Planning L2 (TemAvg, metres) of the FULL quantized pipeline "
            f"({vis_dtype}{' modelopt-quantized' if vis_quantized else ''} vision + "
            f"{args.compress_method}x{args.compress_ratio} compress "
            f"+ {args.precision} LM) decoded trajectories vs nuScenes ground-truth "
            f"waypoints. The HF bf16 baseline for the SAME N samples (run "
            f"scripts/planning_eval.py on final/) is the apples-to-apples accuracy "
            f"reference; a near-equal L2 here means quantization+compression did not "
            f"degrade trajectory quality."),
    }
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[fullpipe] === DONE === saved -> {out_path}")
    print(f"  full_traj_ms mean: {results['full_traj_ms']['mean']:.1f} ms")
    print(f"  stage breakdown: vision {_mean(vision_t):.1f} / compress {_mean(compress_t):.2f} "
          f"/ prefill {_mean(prefill_t):.1f} / decode_total {_mean(decode_t):.1f} (ms)")
    print(f"  peak GPU mem: {bench_peak_gb:.2f} GB")
    if l2_summary and "L2_avg_mean" in l2_summary:
        print(f"  L2_avg: {l2_summary['L2_avg_mean']:.4f} "
              f"({l2_summary['n_samples_evaluated']} samples)")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
