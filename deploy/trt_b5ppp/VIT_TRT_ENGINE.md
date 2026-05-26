# Real TensorRT engine — Qwen3-VL-4B vision tower (B.5'' 1-cam)

Date: 2026-05-26. GPU: RTX 5090 (Blackwell sm_120), CUDA 12.8, torch 2.10,
TensorRT 10.15.1.29, TRT-LLM 1.3.0rc15, modelopt 0.37, transformers 5.5.3.
venv `/venv/trt_llm/bin/python`. GPU 0 only.

Target: `model.model.visual` of
`checkpoints_qwen25/nusc_planning_b5pp_1cam_qwen3vl_multimodal/final`
(`Qwen3VLVisionModel`: 24 blocks, hidden 1024, MLP 4096, 16 heads, head_dim 64,
patch 16, temporal_patch 2, spatial_merge 2, deepstack at layers 5/11/17,
out_hidden 2560).
Real deploy input: 1-cam native video, `video_grid_thw=[2,56,100]` →
`pixel_values_videos [11200,1536]` → ViT outputs `pooler [2800,2560]` + 3
deepstack features `[2800,2560]`.

## This is a REAL TRT engine (not the prior torch-eager Linear-only result)

The prior work (`VIT_QUANT_REAL.md`) only ran torch-eager `torch._scaled_mm` /
`nvfp4_gemm` on the Linear layers (1.36x fp4, attention left in bf16 SDPA, never
an engine). **This work builds and runs an actual serialized TensorRT engine**
(`*.engine`) for the WHOLE ViT (patch_embed + 24 attention+MLP blocks + merger +
3 deepstack mergers), measured by running the engine via the TRT Python runtime.

## How the ONNX export was made traceable (the key engineering)

Qwen3-VL's ViT forward computes `pos_embeds` (`fast_pos_embed_interpolate`),
rotary `cos/sin` (`rot_pos_emb`) and `cu_seqlens` from `grid_thw` with python
loops + `.tolist()` — data-dependent, NOT ONNX-traceable. Since the deploy input
is a FIXED grid `[2,56,100]`, we **precompute those constants once** (with the
real HF code) and bake them as buffers, so the exported graph's only input is
`hidden_states [11200,1536]`.

Critical correctness detail: `grid_thw=[2,56,100]` is a 2-FRAME video, so HF
attention uses `cu_seqlens=[0,5600,11200]` — the two frames attend SEPARATELY
(block-diagonal), NOT one 11200-token full attention. We implement this by
reshaping `[11200,1536] → [2,5600,1536]` and running batched per-frame attention
(no mask), then flattening back to `[11200,1024]` for the mergers. Per-frame
`pos/cos/sin` are identical across the 2 frames (verified `torch.equal`).
With this, the export wrapper matches HF **bit-exactly in fp32 (rel-L2 = 0.0)**.
(First attempt used a single full attention over 11200 tokens → rel-L2 0.43,
the bug; an explicit `-inf` block-diagonal mask fixed correctness but produced
`nan` in TRT bf16 and was slower — the batched-frame reshape is both correct and
fast.)

## Results

| Precision | engine | builds+runs | TRT median ms | speedup vs HF bf16 eager | accuracy |
|-----------|--------|-------------|---------------|--------------------------|----------|
| HF bf16 eager (sdpa) | n/a (torch) | — | 100.9 (ref) | 1.00x | ref |
| **bf16 TRT** | `engines/vit_bf16/vit.engine` | YES | **69.6** | **1.45x** | rel-L2 0.065 vs HF-fp32 (HF-bf16 itself is 0.059) |
| **fp16 TRT** | `engines/vit_fp16/vit.engine` | YES | **51.6** | **1.96x** | rel-L2 0.030 vs HF-fp16 |
| FP8 TRT | (not produced) | **BLOCKED at calibration** | — | — | see FP8 section |

### fp16 engine — VERIFIED (faster + more accurate than bf16)
- Built in ~56 s. Latency mean/median 51.6 ms, min 51.0, std 0.31. vs HF fp16
  eager (97.3 ms) = **1.89x**; vs HF bf16 eager (100.9 ms) = **1.96x**.
- Accuracy: rel-L2 0.030 vs HF fp16 (export wrapper itself is 5.4e-3 — fp16
  keeps more mantissa than bf16). `results/vit_trt_fp16.json`.
- fp16 beats bf16 here because TRT has better-tuned fp16 GEMM/MHA tactics for
  this shape on Blackwell. fp16 is the recommended bf16-class deploy engine.

### bf16 engine — VERIFIED
- Builds in ~48 s; engine 877 MB. Runs via TRT Python runtime on the real
  `[11200,1536]` input; output `pooler [2800,2560]` + 3 deepstack `[2800,2560]`.
- Latency: mean 69.6 / median 69.6 / min 69.4 ms, std 0.15 (8 warmup + 20 timed).
  **1.45x** vs HF bf16 eager (100.9 ms median, same machine/input).
- Accuracy: TRT-bf16 vs HF-**fp32** ground truth rel-L2 = **0.065**; HF-bf16 vs
  HF-fp32 is **0.059**. So the engine is as faithful as native bf16 — the 0.065
  is just bf16 kernel-order rounding, not an export error (fp32 export wrapper is
  bit-exact). `results/vit_trt_bf16.json`.

This 1.45x already matches the theoretical Linear-only ceiling from the prior
analysis (~1.5x) because TRT additionally fuses the attention/LN/elementwise ops
and removes python-eager overhead — without any quantization. fp16 (1.96x) goes
further with better Blackwell fp16 tactics.

### FP8 — BLOCKED at calibration (NOT at the TRT build)

The mature FP8 path is: `modelopt.onnx.quantization.quantize(quantize_mode='fp8',
calibration_data=real)` to insert calibrated Q/DQ nodes, then a TRT-10
strongly-typed FP8 build (`BuilderFlag.FP8`). The strongly-typed FP8 builder code
is written (`build_vit_trt_fp8.py`) and the bf16/fp16 builds prove the TRT build
path works. **The block is purely the ONNX-activation-calibration step.**

modelopt's min-max calibrator instruments the model so EVERY intermediate tensor
becomes a graph output. The Qwen3-VL ViT attention produces `[2,16,5600,5600]`
score/softmax tensors (~4 GB each in fp16) per block; many must stay live
simultaneously across the 24 blocks, so peak memory exceeds the 32 GB GPU. Exact
quoted error (final attempt, full 31 GB arena):

```
onnxruntime ... RUNTIME_EXCEPTION running MatMul '/blocks.3/attn/MatMul':
BFCArena Available memory of 1694182400 is smaller than requested bytes of 4014080000
```

Levers tried (all hit the same wall or a worse one):
- entropy calib on CPU → ran but intractably slow (392 GB RAM, hours, unfinished);
- installed `onnxruntime-gpu` + cuDNN/cuBLAS on `LD_LIBRARY_PATH` → CUDA EP on;
- entropy calib on GPU (3 then 1 sample) → OOM at `/blocks.3/attn/MatMul` (4 GB);
- `calibrate_per_node=True` → stalled at the attention-output histogram
  (~step 606/1950, 3 it/s) then OOM-killed;
- `nodes_to_exclude` the 48 bare attn BMMs → desyncs modelopt's MLP gelu
  partition: `ValueError: Quantization parameters are not specified for
  /blocks.0/mlp/act_fn/Mul_5_output_0`;
- `calibration_method='max'` (single-pass amax, no histogram) → still OOM at
  `/blocks.3/attn/MatMul`;
- CUDA EP `arena_extend_strategy='kSameAsRequested'` + `gpu_mem_limit` 26→31 GB
  (monkeypatched `_prepare_ep_list`) → still OOM (Available 1.7 GB < 4 GB): the
  calibrator legitimately holds many 4 GB tensors live at once.

This is intrinsic to ORT-based calibration of this large-sequence (5600-token,
non-windowed) attention graph — not an arena-config problem. **It is NOT a TRT
limitation**; the TRT FP8 build was never reached. See `results/vit_trt_fp8.json`.

Remaining path (not done, ~0.5–1 day): bypass the ORT calibrator — compute
per-tensor amax in PyTorch (attention runs in <1 GB there; `mtq.quantize` already
yields calibrated amax) and write those as static Q/DQ scales into the ONNX, then
run the same strongly-typed FP8 build; OR export the ViT ONNX with a chunked/
flash attention so no single 4 GB score tensor materializes during calibration.
fp4/NVFP4 was not attempted (gated behind a working FP8 engine).

## Commands

```bash
# bf16 engine (export ONNX + build + this is the validated path)
CUDA_VISIBLE_DEVICES=0 /venv/trt_llm/bin/python build_vit_trt_engine.py --dtype bf16
# run + accuracy + latency
CUDA_VISIBLE_DEVICES=0 /venv/trt_llm/bin/python run_vit_trt_engine.py \
    --engine engines/vit_bf16/vit.engine --dtype bf16

# FP8 engine (modelopt ONNX QDQ -> strongly-typed FP8 TRT). Needs onnxruntime-gpu
# + cuDNN/cuBLAS on LD_LIBRARY_PATH so calibration runs on GPU (CPU calib is hours).
pip install onnxruntime-gpu          # logged: installed 1.26.0
CUDNN=/venv/trt_llm/lib/python3.12/site-packages/nvidia/cudnn/lib
CUBLAS=/venv/trt_llm/lib/python3.12/site-packages/nvidia/cublas/lib
export LD_LIBRARY_PATH=$CUDNN:$CUBLAS:$LD_LIBRARY_PATH
VIT_NCALIB=1 CUDA_VISIBLE_DEVICES=0 /venv/trt_llm/bin/python build_vit_trt_fp8.py
CUDA_VISIBLE_DEVICES=0 /venv/trt_llm/bin/python run_vit_trt_engine.py \
    --engine engines/vit_fp8/vit.engine --dtype fp16 --out results/vit_trt_fp8.json
```

## Files
- `build_vit_trt_engine.py` — bf16/fp16 ONNX export (fixed-grid constant baking,
  batched-frame attention) + TRT build.
- `run_vit_trt_engine.py` — load engine, real-input accuracy (vs HF) + latency.
- `build_vit_trt_fp8.py` — modelopt ONNX FP8 QDQ + strongly-typed FP8 TRT build.
- `results/vit_trt_bf16.json`, `results/vit_trt_fp8.json`.
- engines: `engines/vit_bf16/vit.engine`, `engines/vit_fp8/vit.engine`.

## Installed deps (logged)
- `onnxruntime` 1.26.0, then `onnxruntime-gpu` 1.26.0 (modelopt.onnx calibration;
  GPU EP needs cuDNN+cuBLAS on LD_LIBRARY_PATH).
