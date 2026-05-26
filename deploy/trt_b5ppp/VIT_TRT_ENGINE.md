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

| Precision | engine | builds+runs | TRT median ms | speedup vs HF bf16 eager | rel-L2 vs HF-fp32 |
|-----------|--------|-------------|---------------|--------------------------|-------------------|
| HF bf16 eager (sdpa) | n/a (torch) | — | ~100.7 (ref) | 1.00x | ref |
| **bf16 TRT** | `engines/vit_bf16/vit.engine` | YES | **69.1** | **1.46x** | **0.065** (HF-bf16 itself 0.059) |
| **fp16 TRT** | `engines/vit_fp16/vit.engine` | YES | **51.4** | **1.96x** | **0.031** |
| **FP8 TRT (W+A, e4m3)** | `engines/vit_fp8/vit.engine` | **YES (real engine)** | **51.6** | **1.94x** | **0.174** |
| **NVFP4 TRT (weight-only, e2m1 + fp8 block scale)** | `engines/vit_fp4/vit.engine` | **YES (real engine)** | **51.9** | **1.92x** | **0.320** |
| NVFP4 TRT (W+A) | `engines/vit_fp4_wa/vit.engine` | builds+runs (1.96x) | 52.1 | 1.96x | **2.25 — accuracy BROKEN** (static per-tensor fp4 activation too coarse; do not deploy) |

All five engines above are REAL serialized `.engine` files, each run on the real
`[11200,1536]` input and verified rel-L2 vs the HF **fp32** ground-truth pooler
(8 warmup + 25 timed, median). The FP8 and NVFP4 engines were produced by the
STATIC-AMAX bypass (below); the ORT-calibration block (kept further down for
record) was sidestepped entirely.

### Key finding: ~52 ms is a HARD FLOOR for fp16/fp8/fp4 at this ViT shape
fp16, fp8(W+A), and fp4(weight-only) all land at ~51.5–51.9 ms = ~1.9x. The
Linear GEMMs (qkv/proj/fc1/fc2 ×24 + mergers) are only ~36 ms of the bf16 ViT;
quantizing them to fp8/fp4 shaves GEMM time but the UNQUANTIZED, fp16 remainder
(the 5600-token attention BMMs + softmax + LayerNorms + patch-embed conv) is the
bottleneck, so all low-precision GEMM variants converge to the fp16-graph floor.
This matches the prior torch-eager analysis (`VIT_QUANT_REAL.md`: Linear-only
ceiling ~1.5x; the engine's extra fusion pushes the fp16 floor to 1.9x). To beat
~52 ms you must quantize attention itself (needs an fp8/fp4 fused-MHA kernel),
not the Linears. Accuracy degrades monotonically with precision (fp16 0.031 ->
fp8 0.174 -> fp4-weight 0.320) exactly as expected; weight-only fp4 is the
faithful fp4 artifact, W+A fp4 needs per-block calibrated activation scales
(dynamic-quant) to be usable.

## STATIC-AMAX method (how FP8 + NVFP4 engines were actually built)

The bypass that unblocked FP8/FP4 (the previously-blocked ORT path is preserved
below for record):

1. **AMAX in PyTorch, never ORT** (`collect_vit_amax.py`). Run the SAME
   `VisualExportWrapper` over the 6 real calib samples with a forward hook on
   every `nn.Linear`, keeping the running per-tensor max of `|input|` (this is
   exactly modelopt's `algorithm="max"`, done by hand). Attention scores live in
   torch (<1 GB) so the 4 GB-per-block ORT OOM never happens. -> 104 input AMAX,
   `results/vit_input_amax.json`.
2. **Export an fp16 ONNX** (`export_vit_onnx_fp16.py`). CRITICAL: STRONGLY_TYPED
   fp8/fp4 builds take precision verbatim from the ONNX and IGNORE the
   BF16/FP16 builder flags that recast the fp32 graph for the bf16/fp16 engines.
   So the non-quantized ops (attention/LN/softmax/elementwise) must ALREADY be
   fp16 in the ONNX, else they run in fp32 and the engine is ~6x slower than
   bf16 (observed: an fp8 engine on the fp32 graph ran at 294 ms / 0.34x). The
   fp16 ONNX makes the fp8/fp4 GEMMs sit in an fp16 graph -> the ~52 ms floor.
3. **Inject static Q/DQ initializers** (no calibration pass):
   - **FP8** (`inject_vit_qdq.py --mode fp8`): per-tensor e4m3 `QuantizeLinear ->
     DequantizeLinear` on BOTH the activation input (scale = calibrated
     `in_amax/448`) and the weight (scale = `|W|.max/448`, weight amax read
     straight from the ONNX initializer) of all 104 weight-MatMul/Gemm nodes.
     DQ output dtype = fp16 to match the graph. The 48 attention BMMs (the
     [2,16,5600,5600] tensors) are LEFT fp16 — quantizing them is what OOM'd ORT
     and is unnecessary for the GEMM win.
   - **NVFP4** (`inject_vit_nvfp4.py`): the canonical TRT 2-DequantizeLinear
     weight pattern (mirrors modelopt `replace_fp4qdq_with_2dq`): per-block
     (block=16 on the K axis) fp8-e4m3 scales DQ'd by ONE fp32-per-tensor global
     scale (`(|W|.max/6)/448`) -> fp16 block scale, then the e2m1 weight DQ'd by
     that block scale (`axis=-1, block_size=16`). The per-tensor global scale is
     emitted in fp16 so the whole weight chain outputs fp16 (an fp32 scale made
     TRT reject the MatMul: "A is Half, B is Float"). `--weight-only` keeps
     activations fp16 (the FAITHFUL fp4 artifact, rel-L2 0.32); the default also
     fp4-quantizes activations but static per-tensor fp4 activation scales are
     too coarse (rel-L2 2.25).
   - ml_dtypes `float4_e2m1fn` / `float8_e4m3fn` arrays carry the low-precision
     initializers; gs's exporter mis-sizes packed fp4 (expects 2 vals/byte), so
     the fp4/fp8 initializers are injected post-export via `onnx.numpy_helper`.
4. **STRONGLY_TYPED build** (`build_vit_trt_static.py`), reusing the proven
   bf16/fp16 builder. With STRONGLY_TYPED you must NOT set `BuilderFlag.FP8/FP16`
   (rejected: "condition: !config.getFlag(kFP8)"); precision comes only from the
   graph QDQ types. FP8 builds in ~25 s (456 MB engine), NVFP4 in ~45 s (310 MB).

### Files (static-amax path)
- `collect_vit_amax.py` — PyTorch per-Linear input AMAX (max calib, no ORT).
- `export_vit_onnx_fp16.py` — fp16 ViT ONNX export (so the non-quant graph is fp16).
- `inject_vit_qdq.py` — static fp8 (or per-tensor fp4) Q/DQ injection.
- `inject_vit_nvfp4.py` — canonical NVFP4 2-DQ weight injection (+ optional act).
- `build_vit_trt_static.py` — STRONGLY_TYPED fp8/fp4 TRT build.
- `run_vit_trt_static.py` — run + rel-L2 vs HF-fp32 + latency.
- engines: `engines/vit_fp8/vit.engine`, `engines/vit_fp4/vit.engine` (weight-only,
  the deploy fp4 artifact), `engines/vit_fp4_wa/vit.engine` (W+A, broken accuracy).
- results: `results/vit_trt_fp8.json`, `vit_trt_fp4.json`, `vit_trt_fp4_wa.json`,
  `results/vit_input_amax.json`.

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

### FP8 — the OLD ORT-calibration block (HISTORICAL; now bypassed, see above)

NOTE: this section records the ORT-calibration dead-end. It is SUPERSEDED by the
STATIC-AMAX method above, which produced real FP8 + NVFP4 engines. Kept for record.

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
