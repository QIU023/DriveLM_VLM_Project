#!/usr/bin/env /venv/trt_llm/bin/python
"""UNIFIED end-to-end deploy bench: REAL TRT ViT engine -> FasterVLM -> TRT-LLM.

This is the #175 deliverable. Unlike bench_full_pipeline.py (which runs the
vision tower on HF torch, optionally modelopt fake-quant), THIS bench feeds the
1-cam video through the REAL serialized TensorRT ViT engine
(engines/vit_{bf16,fp16,fp8,fp4}/vit.engine), so the vision latency is a real
TRT-engine number, not HF.

Chain (per the task spec):
    raw 1-cam video pixel_values_videos [11200,1536] (fixed grid [2,56,100])
      -> [TRT ViT engine]  pooler [2800,2560] + 3 deepstack [2800,2560]
      -> concat dim=1 -> video mm_embedding [2800, 10240]      (= HF format)
      -> [FasterVLM]       prune video 2800 -> 700 (HD-map 121 untouched)
      -> [TRT-LLM Qwen3 LM] embedding injection (mm-disagg) prefill
      -> [decode]          full 14-token trajectory
    => full_traj_ms = vit_engine + compress + LM-prefill + LM-decode

The HD-map IMAGE branch (grid [1,22,22], 121 tokens) stays on the HF vision
tower: the ViT engine has the video grid [2,56,100] baked in as constants, so it
ONLY runs the video. This matches the deployed automotive design (one
fixed-shape camera-video engine; the small HD-map raster goes through the
generic vision path). The image vision_ms is tiny (484 patches) and is folded
into vision_ms with a separate `image_vision_ms` field for transparency.

LM path (prefill/decode, mm-disagg embedding injection, mrope, L2 decode) is
REUSED verbatim from bench_full_pipeline.py / bench_trt.py — we only swap the
video vision stage. JSON per precision -> B5pp_e2e_{prec}_fastervlm4.json.

Precision pairing (task spec):
    bf16: ViT-bf16 + LM bf16(final)
    fp16: ViT-fp16 + LM bf16(final)      (fp16 ViT recommended)
    fp8 : ViT-fp8  + LM fp8(quant_fp8)
    fp4 : ViT-fp4  + LM nvfp4(quant_nvfp4)
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

import bench_trt as B  # noqa: E402
import bench_full_pipeline as FP  # noqa: E402

# Precision -> (vit engine subdir, LM ckpt subdir)
PRECISION_PAIRING = {
    "bf16": ("vit_bf16", "final"),
    "fp16": ("vit_fp16", "final"),
    "fp8":  ("vit_fp8",  "quant_fp8"),
    "fp4":  ("vit_fp4",  "quant_nvfp4"),
}


# ---------------------------------------------------------------------------
# TRT ViT engine wrapper (video only, fixed grid [2,56,100]).
# ---------------------------------------------------------------------------

class TrtVitEngine:
    """Loads a serialized ViT TRT engine. forward(pixel_values_videos[11200,1536])
    -> video mm_embedding [2800, hidden*(1+n_deepstack)] = cat([pooler, ds0,
    ds1, ds2], dim=1) (HF assemble_mm_embedding video format)."""

    def __init__(self, engine_path: str, device: str = "cuda:0"):
        import tensorrt as trt
        import torch
        self.torch = torch
        self.trt = trt
        self._TRT_TO_TORCH = {
            trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16,
            trt.DataType.BF16: torch.bfloat16, trt.DataType.INT32: torch.int32,
            trt.DataType.INT8: torch.int8,
        }
        self.device = device
        logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.in_names, self.out_names = [], []
        for i in range(self.engine.num_io_tensors):
            n = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(n)
            (self.in_names if mode == trt.TensorIOMode.INPUT
             else self.out_names).append(n)
        assert len(self.in_names) == 1, self.in_names
        # outputs ordered: pooler, deepstack0, deepstack1, deepstack2
        assert len(self.out_names) == 4, self.out_names
        self.in_name = self.in_names[0]
        self.in_dtype = self._TRT_TO_TORCH[self.engine.get_tensor_dtype(self.in_name)]
        self.stream = torch.cuda.Stream()

    def forward(self, pixel_values_videos):
        """pixel_values_videos: [11200,1536] tensor. Returns video mm_embedding
        [2800, 10240] (fp32 on `device` to match HF assemble dtype handling)."""
        torch = self.torch
        d_in = pixel_values_videos.to(self.device, self.in_dtype).contiguous()
        self.ctx.set_input_shape(self.in_name, tuple(d_in.shape))
        self.ctx.set_tensor_address(self.in_name, d_in.data_ptr())
        out_bufs = {}
        for n in self.out_names:
            shape = tuple(self.ctx.get_tensor_shape(n))
            odt = self._TRT_TO_TORCH[self.engine.get_tensor_dtype(n)]
            buf = torch.empty(shape, dtype=odt, device=self.device)
            out_bufs[n] = buf
            self.ctx.set_tensor_address(n, buf.data_ptr())
        self.ctx.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        # cat([pooler, ds0, ds1, ds2], dim=1) in float (matches HF assemble path)
        parts = [out_bufs[self.out_names[0]].float()] + [
            out_bufs[self.out_names[k]].float() for k in range(1, 4)]
        return torch.cat(parts, dim=1).contiguous()


# ---------------------------------------------------------------------------
# Vision (TRT ViT engine for video + HF for HD-map image) + compress, timed.
# Mirrors bench_full_pipeline.vision_and_compress_timed but the VIDEO vision
# forward is the TRT engine. The rest (FasterVLM, prompt/grid/mrope rebuild,
# mm-disagg handle minting) is identical and delegated to bench_trt / FP.
# ---------------------------------------------------------------------------

def vision_and_compress_timed_trt(*, sample, vit_engine, hf_model, processor,
                                  image_pad_id, video_pad_id, device, dtype,
                                  compress_method, compress_ratio):
    import torch
    from phase1_5_vision_embeds import assemble_mm_embedding

    pv_videos = sample.get("pixel_values_videos")
    video_grid_thw = sample.get("video_grid_thw")
    pv_image = sample.get("pixel_values")
    image_grid_thw = sample.get("image_grid_thw")
    assert pv_videos is not None and video_grid_thw is not None

    # ---- STAGE 1a: VIDEO vision via REAL TRT ViT engine (TIMED) ----
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    video_concat = vit_engine.forward(pv_videos).to(device)  # [2800, 10240]
    torch.cuda.synchronize()
    video_vision_ms = 1000.0 * (time.perf_counter() - t0)

    # ---- STAGE 1b: HD-map IMAGE vision via HF (TIMED; tiny: 484 patches) ----
    image_vision_ms = 0.0
    image_concat = None
    n_image = 0
    if pv_image is not None and image_grid_thw is not None:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        img_res = assemble_mm_embedding(
            hf_model=hf_model, pixel_values=pv_image,
            image_grid_thw=image_grid_thw, pixel_values_videos=None,
            video_grid_thw=None, modality_order=("image",))
        torch.cuda.synchronize()
        image_vision_ms = 1000.0 * (time.perf_counter() - t0)
        image_concat = img_res["mm_embedding"].to(device, dtype=dtype)
        n_image = int(img_res["image_mm_tokens"])

    vision_ms = video_vision_ms + image_vision_ms

    # assemble full mm_embedding in INPUT-IDS order (B.5'' = video first, image second)
    video_concat = video_concat.to(device, dtype=dtype)
    n_video = int(video_concat.shape[0])
    assert n_video >= int(EXPECTED_VIDEO_TOKENS * 0.9), (
        f"video tokens {n_video} << expected {EXPECTED_VIDEO_TOKENS}")
    if image_concat is not None:
        mm_embedding_full = torch.cat([video_concat, image_concat], dim=0).contiguous()
    else:
        mm_embedding_full = video_concat

    # ---- STAGE 2: FasterVLM compress on VIDEO tokens (TIMED) ----
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

    # ---- build TRT request payload (handle minting / mrope) ----
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
        "video_vision_ms": video_vision_ms,
        "image_vision_ms": image_vision_ms,
        "compress_ms": compress_ms,
        "n_video_in": n_video_in,
        "n_video_out": n_video_out,
        "n_image": int(n_image),
    }
    return text_prompt, make_disagg, prompt_len, stats


# ---------------------------------------------------------------------------
# L2 eval (full pipeline, TRT ViT video vision). Mirror of FP.eval_l2_fullpipe.
# ---------------------------------------------------------------------------

def eval_l2_e2e(*, llm, vit_engine, hf_model, processor, image_pad_id,
                video_pad_id, device, dtype, n_samples, max_new_tokens,
                config_yaml, compress_method, compress_ratio):
    import numpy as np
    import torch
    from tensorrt_llm import SamplingParams

    add_project_paths()
    from trajectory_tokenizer import TrajectoryTokenizer, TrajectoryTokenizerConfig

    val_ds = build_calib_dataset(processor=processor, n_samples=n_samples,
                                 split="val", config_yaml=config_yaml)
    n_total = len(val_ds)
    print(f"[l2] evaluating e2e L2 on {n_total} val samples (TRT ViT video vision)")
    traj_cfg = TrajectoryTokenizerConfig(num_waypoints=val_ds.num_future)
    traj_tok = TrajectoryTokenizer(traj_cfg)
    sp = SamplingParams(max_tokens=max_new_tokens, temperature=0.0)

    per_sample = []
    n_failed = 0
    first_errs = []
    for i in range(n_total):
        try:
            sample = val_ds[i]
            text_prompt, make_disagg, _, _ = vision_and_compress_timed_trt(
                sample=sample, vit_engine=vit_engine, hf_model=hf_model,
                processor=processor, image_pad_id=image_pad_id,
                video_pad_id=video_pad_id, device=device, dtype=dtype,
                compress_method=compress_method, compress_ratio=compress_ratio)
            disagg, _hold = make_disagg()
            out = llm.generate([{"prompt": text_prompt}], sampling_params=sp,
                               disaggregated_params=disagg)
            del _hold
        except Exception as e:
            n_failed += 1
            if len(first_errs) < 3:
                first_errs.append(f"sample {i}: {e}")
            print(f"[l2]   sample {i} failed: {e}; skip")
            continue
        token_ids = list(out[0].outputs[0].token_ids)
        pred_wp = FP._decode_waypoints(token_ids, traj_tok, val_ds.num_future)
        gt_wp = sample["_meta_waypoints"].cpu().numpy()
        valid = sample["_meta_valid_mask"].cpu().numpy()
        m = FP._l2_temavg(pred_wp, gt_wp, valid)
        per_sample.append({k: m.get(k, float("nan"))
                           for k in ("L2_avg", "L2_1s", "L2_2s", "L2_3s")})
        if (i + 1) % 10 == 0 or (i + 1) == n_total:
            run = np.nanmean([s["L2_avg"] for s in per_sample]) if per_sample else float("nan")
            print(f"[l2]   {i+1}/{n_total}  L2_avg(running)={run:.4f}")

    def _m(key):
        vals = [s[key] for s in per_sample if not math.isnan(s.get(key, float("nan")))]
        return float(sum(vals) / len(vals)) if vals else float("nan")

    return {
        "n_samples_evaluated": len(per_sample),
        "n_failed": n_failed,
        "first_errors": first_errs,
        "L2_avg_mean": _m("L2_avg"), "L2_1s_mean": _m("L2_1s"),
        "L2_2s_mean": _m("L2_2s"), "L2_3s_mean": _m("L2_3s"),
    }


# ---------------------------------------------------------------------------
# Parity check: first generated token = traj_start (151934) and HF top-5 sanity.
# ---------------------------------------------------------------------------

def parity_check(*, llm, vit_engine, hf_model, processor, image_pad_id,
                 video_pad_id, device, dtype, sample, compress_method,
                 compress_ratio):
    from tensorrt_llm import SamplingParams
    add_project_paths()
    from trajectory_tokenizer import TrajectoryTokenizerConfig

    traj_start_id = TrajectoryTokenizerConfig().traj_start_id
    text_prompt, make_disagg, _, _ = vision_and_compress_timed_trt(
        sample=sample, vit_engine=vit_engine, hf_model=hf_model,
        processor=processor, image_pad_id=image_pad_id, video_pad_id=video_pad_id,
        device=device, dtype=dtype, compress_method=compress_method,
        compress_ratio=compress_ratio)
    disagg, _hold = make_disagg()
    sp = SamplingParams(max_tokens=1, temperature=0.0)
    out = llm.generate([{"prompt": text_prompt}], sampling_params=sp,
                       disaggregated_params=disagg)
    del _hold
    first_tok = int(out[0].outputs[0].token_ids[0])
    return {"first_token": first_tok, "traj_start_id": int(traj_start_id),
            "match": first_tok == int(traj_start_id)}


def parse_args():
    p = argparse.ArgumentParser(description="UNIFIED e2e bench: TRT ViT -> FasterVLM -> TRT-LLM")
    p.add_argument("--precision", choices=list(PRECISION_PAIRING), required=True)
    p.add_argument("--vision-ckpt", default=DEFAULT_CKPT)
    p.add_argument("--n-warmup", type=int, default=2)
    p.add_argument("--n-runs", type=int, default=5)
    p.add_argument("--max-new-tokens", type=int, default=14)
    p.add_argument("--l2-n", type=int, default=50)
    p.add_argument("--compress-method", default="fastervlm",
                   choices=["none", "fastervlm", "prumerge", "pyramiddrop", "avg_pool"])
    p.add_argument("--compress-ratio", type=int, default=4)
    p.add_argument("--config-yaml", default=DEFAULT_CONFIG_YAML)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-seq-len", type=int, default=12288)
    p.add_argument("--free-gpu-mem-frac", type=float, default=0.35)
    p.add_argument("--out", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    vit_sub, lm_sub = PRECISION_PAIRING[args.precision]
    vit_engine_path = _HERE / "engines" / vit_sub / "vit.engine"
    lm_ckpt = Path(DEFAULT_PARENT) / lm_sub
    vision_ckpt = Path(args.vision_ckpt)
    out_path = Path(args.out) if args.out else Path(DEFAULT_BENCH_OUT_DIR) / f"B5pp_e2e_{args.precision}_fastervlm{args.compress_ratio}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for pth, label in [(vit_engine_path, "ViT engine"), (lm_ckpt, "LM ckpt"),
                       (vision_ckpt, "vision ckpt (HD-map HF)")]:
        if not Path(pth).exists():
            print(f"[e2e] FATAL: {label} not found: {pth}", file=sys.stderr)
            return 2

    print(f"[e2e] precision = {args.precision}")
    print(f"[e2e] ViT engine = {vit_engine_path}")
    print(f"[e2e] LM ckpt    = {lm_ckpt}")
    print(f"[e2e] HD-map vision (HF) = {vision_ckpt}")
    print(f"[e2e] compress   = {args.compress_method} x{args.compress_ratio}")
    print(f"[e2e] out        = {out_path}")

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    add_project_paths()

    # HF model: ONLY used for the HD-map image branch + config (mrope). bf16.
    print("[e2e] loading HF (HD-map vision tower + config, bf16) ...")
    hf_model = AutoModelForImageTextToText.from_pretrained(
        str(vision_ckpt), torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(args.device).eval()
    processor = AutoProcessor.from_pretrained(str(vision_ckpt))
    tok = processor.tokenizer
    image_pad_id = tok.convert_tokens_to_ids("<|image_pad|>")
    video_pad_id = tok.convert_tokens_to_ids("<|video_pad|>")
    print(f"[e2e] image_pad={image_pad_id} video_pad={video_pad_id}")

    print("[e2e] loading TRT ViT engine ...")
    vit_engine = TrtVitEngine(str(vit_engine_path), device=args.device)

    val_ds_one = build_calib_dataset(processor=processor, n_samples=2,
                                     split="val", config_yaml=args.config_yaml)
    sample0 = val_ds_one[0]

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import KvCacheConfig
    reset_peak_mem()
    print("[e2e] loading TRT LM engine ...")
    t0 = time.perf_counter()
    llm = LLM(
        model=str(lm_ckpt), tensor_parallel_size=1, max_batch_size=1,
        max_seq_len=int(args.max_seq_len), max_num_tokens=int(args.max_seq_len),
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=float(args.free_gpu_mem_frac)),
        trust_remote_code=True,
    )
    print(f"[e2e] TRT LM loaded in {time.perf_counter()-t0:.1f}s")

    sp_full = SamplingParams(max_tokens=int(args.max_new_tokens), temperature=0.0)
    sp_one = SamplingParams(max_tokens=1, temperature=0.0)

    # === Parity check FIRST (before trusting L2) ===
    print("[e2e] parity check (first decoded token == traj_start?) ...")
    parity = parity_check(
        llm=llm, vit_engine=vit_engine, hf_model=hf_model, processor=processor,
        image_pad_id=image_pad_id, video_pad_id=video_pad_id, device=args.device,
        dtype=torch.bfloat16, sample=sample0,
        compress_method=args.compress_method, compress_ratio=int(args.compress_ratio))
    print(f"[e2e] parity: first_token={parity['first_token']} "
          f"traj_start={parity['traj_start_id']} match={parity['match']}")

    # === Warmup (prefill + decode each capture own CUDA graph) ===
    print(f"[e2e] warmup ({args.n_warmup} x {{prefill,decode}}) ...")
    for _ in range(int(args.n_warmup)):
        _tp, _md, _pl, _st = vision_and_compress_timed_trt(
            sample=sample0, vit_engine=vit_engine, hf_model=hf_model,
            processor=processor, image_pad_id=image_pad_id,
            video_pad_id=video_pad_id, device=args.device, dtype=torch.bfloat16,
            compress_method=args.compress_method, compress_ratio=int(args.compress_ratio))
        d1, h1 = _md()
        llm.generate([{"prompt": _tp}], sampling_params=sp_one, disaggregated_params=d1)
        del h1
        d2, h2 = _md()
        llm.generate([{"prompt": _tp}], sampling_params=sp_full, disaggregated_params=d2)
        del h2

    # === Timed runs ===
    print(f"[e2e] timing ({args.n_runs}) ...")
    reset_peak_mem()
    video_vis_t, image_vis_t, vision_t, compress_t, prefill_t, decode_t, full_t = (
        [], [], [], [], [], [], [])
    last_stats = None
    n_dec = max(1, int(args.max_new_tokens) - 1)
    for _ in range(int(args.n_runs)):
        torch.cuda.synchronize()
        _tp, make_disagg, _pl, st = vision_and_compress_timed_trt(
            sample=sample0, vit_engine=vit_engine, hf_model=hf_model,
            processor=processor, image_pad_id=image_pad_id,
            video_pad_id=video_pad_id, device=args.device, dtype=torch.bfloat16,
            compress_method=args.compress_method, compress_ratio=int(args.compress_ratio))
        last_stats = st
        d1, h1 = make_disagg()
        torch.cuda.synchronize(); tpf = time.perf_counter()
        llm.generate([{"prompt": _tp}], sampling_params=sp_one, disaggregated_params=d1)
        torch.cuda.synchronize(); prefill_ms = 1000.0 * (time.perf_counter() - tpf)
        del h1
        d2, h2 = make_disagg()
        torch.cuda.synchronize(); tfd = time.perf_counter()
        llm.generate([{"prompt": _tp}], sampling_params=sp_full, disaggregated_params=d2)
        torch.cuda.synchronize(); lm_full_ms = 1000.0 * (time.perf_counter() - tfd)
        del h2
        decode_ms = max(0.0, lm_full_ms - prefill_ms)
        full_ms = st["vision_ms"] + st["compress_ms"] + lm_full_ms
        video_vis_t.append(st["video_vision_ms"]); image_vis_t.append(st["image_vision_ms"])
        vision_t.append(st["vision_ms"]); compress_t.append(st["compress_ms"])
        prefill_t.append(prefill_ms); decode_t.append(decode_ms); full_t.append(full_ms)

    bench_peak_gb = peak_mem_gb()
    print(f"[e2e] vit_engine={FP._mean(video_vis_t):.1f}ms hdmap_hf={FP._mean(image_vis_t):.2f}ms "
          f"compress={FP._mean(compress_t):.2f}ms prefill={FP._mean(prefill_t):.1f}ms "
          f"decode_total={FP._mean(decode_t):.1f}ms full={FP._mean(full_t):.1f}ms "
          f"peak={bench_peak_gb:.2f}GB")

    # === L2 eval (>=50 samples) ===
    l2_summary = None
    if int(args.l2_n) > 0:
        print(f"[e2e] L2 eval on {args.l2_n} val samples ...")
        try:
            l2_summary = eval_l2_e2e(
                llm=llm, vit_engine=vit_engine, hf_model=hf_model,
                processor=processor, image_pad_id=image_pad_id,
                video_pad_id=video_pad_id, device=args.device,
                dtype=torch.bfloat16, n_samples=int(args.l2_n),
                max_new_tokens=int(args.max_new_tokens),
                config_yaml=args.config_yaml,
                compress_method=args.compress_method,
                compress_ratio=int(args.compress_ratio))
        except Exception as e:
            print(f"[e2e] L2 eval failed: {e}")
            l2_summary = {"error": str(e)}

    lm_quantized = args.precision in ("fp8", "fp4")
    results = {
        "ckpt_lm": str(lm_ckpt),
        "ckpt_vision_hdmap": str(vision_ckpt),
        "vit_engine": str(vit_engine_path),
        "model": "Qwen3-VL-4B B.5'' 1-cam full-modal (video + HD-map + bbox + ego)",
        "precision": args.precision,
        "precision_pairing": {"vit_engine": vit_sub, "lm_ckpt": lm_sub},
        "backend": (f"UNIFIED E2E: video[TRT ViT engine {vit_sub}] + "
                    f"HD-map[HF bf16] -> {args.compress_method}x{args.compress_ratio} "
                    f"-> LM[TRT-LLM 1.3.0rc15 {args.precision}, embedding-injection] "
                    f"-> decode {args.max_new_tokens}-tok trajectory"),
        "real_trt_vision": True,
        "vision_note": ("VIDEO (1-cam, grid [2,56,100], 11200 patches) runs through "
                        "the REAL serialized TRT ViT engine (pooler [2800,2560] + 3 "
                        "deepstack -> mm_embedding [2800,10240]). The HD-map IMAGE "
                        "branch (grid [1,22,22], 484 patches -> 121 tokens) stays on "
                        "the HF vision tower because the ViT engine has the video grid "
                        "baked in as constants. image_vision_ms is the (tiny) HF HD-map "
                        "cost; video_vision_ms is the real TRT engine latency."),
        "parity": parity,
        "stages": {
            "video_vision": {"runtime": f"TRT engine ({vit_sub})", "dtype": args.precision,
                             "real_engine": True, "timed": True},
            "hdmap_vision": {"runtime": "HF torch bf16", "dtype": "bf16", "timed": True},
            "compress": {"runtime": "torch GPU", "method": args.compress_method,
                         "ratio": int(args.compress_ratio), "timed": True},
            "prefill": {"runtime": "TRT-LLM", "dtype": args.precision,
                        "quantized": lm_quantized, "timed": True},
            "decode": {"runtime": "TRT-LLM", "dtype": args.precision,
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
            "video_vision_trt": {"mean": FP._mean(video_vis_t), "p50": FP.percentile(video_vis_t, 50)},
            "hdmap_vision_hf": {"mean": FP._mean(image_vis_t), "p50": FP.percentile(image_vis_t, 50)},
            "vision_total": {"mean": FP._mean(vision_t), "p50": FP.percentile(vision_t, 50)},
            "compress": {"mean": FP._mean(compress_t), "p50": FP.percentile(compress_t, 50)},
            "prefill": {"mean": FP._mean(prefill_t), "p50": FP.percentile(prefill_t, 50)},
            "decode_total": {"mean": FP._mean(decode_t),
                             "per_token_mean": FP._mean(decode_t) / n_dec,
                             "note": f"total decode time for {n_dec} tokens"},
        },
        "full_traj_ms": {
            "mean": FP._mean(full_t), "p50": FP.percentile(full_t, 50),
            "p99": FP.percentile(full_t, 99),
            "note": "end-to-end = TRT-ViT(video) + HF(HD-map) + compress + LM(prefill+decode)",
        },
        "gpu_mem_gb": {
            "bench_peak": bench_peak_gb,
            "note": ("peak across TRT ViT engine + HF HD-map vision tower + TRT LM "
                     "engine resident together."),
        },
        "l2_summary": l2_summary,
        "l2_note": (f"Planning L2 (TemAvg, metres) of the FULL e2e pipeline (TRT ViT "
                    f"{args.precision} video vision + HD-map HF + "
                    f"{args.compress_method}x{args.compress_ratio} + {args.precision} "
                    f"LM) decoded trajectories vs nuScenes GT waypoints."),
    }
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[e2e] === DONE === saved -> {out_path}")
    print(f"  full_traj_ms mean: {results['full_traj_ms']['mean']:.1f} ms")
    print(f"  stages: vit {FP._mean(video_vis_t):.1f} / hdmap {FP._mean(image_vis_t):.2f} "
          f"/ compress {FP._mean(compress_t):.2f} / prefill {FP._mean(prefill_t):.1f} "
          f"/ decode_total {FP._mean(decode_t):.1f} (ms)")
    print(f"  peak GPU mem: {bench_peak_gb:.2f} GB")
    if l2_summary and "L2_avg_mean" in l2_summary:
        print(f"  L2_avg: {l2_summary['L2_avg_mean']:.4f} ({l2_summary['n_samples_evaluated']} samples)")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
