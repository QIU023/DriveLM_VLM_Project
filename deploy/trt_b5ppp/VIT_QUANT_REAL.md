# Real (not fake-quant) ViT quantization — B.5'' Qwen3-VL-4B vision tower

Date: 2026-05-26. GPU: RTX 5090 (Blackwell, sm_120), CUDA 12.8, torch 2.10,
TRT-LLM 1.3.0rc15, modelopt 0.37. venv `/venv/trt_llm/bin/python`.

Target: `model.model.visual` of
`checkpoints_qwen25/nusc_planning_b5pp_1cam_qwen3vl_multimodal/final`.
Input: 1-cam native video, `video_grid_thw=[2,56,100]` (`pixel_values_videos`
`[11200,1536]`, 2800 post-merge tokens). ViT = 24 blocks, hidden 1024,
intermediate 4096, 16 heads. Bench: 8 warmup + 10-20 timed forwards of
`visual(pixel_values_videos, video_grid_thw)`, GPU 0, mean ms.

## TL;DR — there IS a real, measured speedup, but it is modest and Linear-only

| Precision | path | ViT mean ms | speedup vs bf16 | real GEMM? |
|-----------|------|-------------|-----------------|------------|
| bf16      | baseline (sdpa) | ~104 | 1.00x | n/a |
| fp8 Linear | `torch._scaled_mm` (fp8×fp8→bf16) | ~92–96 | **~1.09–1.13x** | YES (real) |
| fp4 / NVFP4 Linear | trtllm `fp4_quantize`+`nvfp4_gemm` | ~76 | **~1.36x** | YES (real) |

Numbers from `results/vit_real_quant_speedup.json`. fp8 mean jitters 90–96ms
(system noise); p50 ≈ 91.7ms → 1.13x. fp4 is stable at ~76ms.

These are GENUINE low-precision GEMM speedups (verified: weights stored as
`float8_e4m3fn` / uint8-packed fp4; matmul is `torch._scaled_mm` / the trtllm
fp4 kernel — NOT bf16 matmul + dequant). They are **Linear-ONLY**: attention is
SDPA bf16 (untouched). patch_embed is a Conv3d, also untouched.

## Why the speedup is small (the honest ceiling)

Micro-bench of the bf16 ViT at this shape:
- All Linear layers (qkv 1024→3072, proj 1024→1024, mlp fc1 1024→4096, fc2
  4096→1024) ×24 blocks = **~36 ms** of the ~104 ms ViT.
- Attention (SDPA over the full 11200-token sequence — Qwen3-VL ViT is NOT
  windowed) + LayerNorms + patch_embed conv = **~68 ms**, all bf16.

So even a zero-cost Linear caps Linear-only speedup at ~104/68 ≈ **1.5x**. fp4
GEMM (faster than fp8 on Blackwell) reaches 1.36x; fp8 GEMM only ~1.1x. To beat
~1.5x you MUST quantize attention too, which needs an fp8 attention kernel
(flash-attn is broken in this env: no `flash_attn_2_cuda` ext) or a full TRT
engine with fp8/fp4 fused MHA.

## Routes tried, in order (honest outcomes)

### Route 1 — modelopt real compress (`mtq.quantize` → `mtq.compress`)
- `mtq.quantize(visual, FP8_DEFAULT_CFG)` + `mtq.compress(visual)` DOES produce
  a real path: 104 layers become `RealQuantLinear`, weights become
  `float8_e4m3fn`, `is_real_quant_gemm_enabled=True`, forward dispatches to
  `torch._scaled_mm` (`backends/fp8_per_tensor_gemm.py`). This is real, not fake.
- **BUT it was SLOWER: 0.76x (104 → 138 ms), consistently across all runs.** The
  modelopt `RealQuantLinear` wrapper adds per-call overhead — dynamic input
  quant under `@torch.compile(dynamic=True)` (re-guard/recompile on varying
  seq), `QTensorWrapper` indirection, output_quantizer — that exceeds the small
  fp8 GEMM win at hidden=1024. NOT a usable speedup as-is.
- Fix needed first: `patch_embed.proj` is a `QuantConv3d`. `compress` only
  converts Linears to RealQuant; the conv kept fp8 weight + fake-quant input and
  crashed forward (`NotImplementedError: "fake_e4m3fy" not implemented for
  'Float8_e4m3fn'`). Resolved by disabling conv + patch_embed quantizers in the
  cfg (`bench_vit_real_quant.py`), quantizing Linears only.

### Route 1' — minimal-overhead manual static fp8 Linear (THE WIN)
- Replacing the modelopt wrapper with a bare module (weight pre-quantized to
  e4m3 once, static per-tensor scales, plain `torch._scaled_mm`, no
  torch.compile, no QTensorWrapper) recovers the GEMM win: **~1.09–1.13x real**.
  See `vit_real_quant_linear.py::StaticFp8Linear`. This isolates that the
  negative Route-1 result was wrapper overhead, not the fp8 GEMM itself.

### Route 4 — NVFP4 / fp4 (Blackwell FP4 GEMM)
- modelopt `mtq.compress(visual, NVFP4_DEFAULT_CFG)` compresses (104
  RealQuantLinear) but its forward is **BROKEN against TRT-LLM 1.3.0rc15**:
  `RuntimeError: mat2Scale dtype is Float8_e4m3fn, while Byte is expected`
  (`backends/nvfp4_gemm.py:90` passes the modelopt e4m3 `_scale` where the
  trtllm op wants a uint8 block scale — a modelopt 0.37 ↔ trtllm 1.3 version
  skew).
- The underlying trtllm fp4 ops work fine directly: `torch.ops.trtllm.
  fp4_quantize` (→ uint8 data + uint8 block scale) and `torch.ops.trtllm.
  nvfp4_gemm` verified standalone. So a manual `StaticFp4Linear` using these
  ops directly gives **~1.36x real (104 → 76 ms)** — the best result. See
  `vit_real_quant_linear.py::StaticFp4Linear`.

### Route 2 — torch-tensorrt: NOT AVAILABLE
`import torch_tensorrt` → `ModuleNotFoundError`. Not installed in this venv. Skipped.

### Route 3 — ONNX → TRT fp8 engine: NOT ATTEMPTED (no tooling, high risk)
`trtexec` is not present anywhere on the box, and ONNX export of the Qwen3-VL
ViT (rotary pos-emb on patches, deepstack merger, dynamic grid_thw) is high-risk
within the time-box. Per instructions, not pursued once Routes 1'/4 gave a real
measured number. This remains the production target.

## Accuracy honesty
The manual static modules use crude per-tensor input scales from ONE calib
forward. Raw-hidden-state relative-L2 vs bf16: **fp8 ≈ 0.20, fp4 ≈ 0.40** —
these prove LATENCY, not accuracy. The modelopt fake-quant path (existing
`quant_vision.py`) is the accuracy-faithful one (calibrated AMAX) but has no
speedup. A deploy artifact needs calibrated AMAX scales fed through the
minimal-overhead GEMM (not the modelopt RealQuantLinear wrapper).

## Remaining work + time estimates

**Full fp8 ViT (accurate + fast), ~0.5–1 day:** take modelopt's calibrated
per-tensor AMAX (already produced by `mtq.quantize`) and apply it as the static
scale in `StaticFp8Linear` instead of the 1-shot hook amax; verify rel-L2 and
planning-L2 parity; wire into `bench_full_pipeline.py` vision stage (replace the
fake-quant `quantize_vision_inplace` call with the static-fp8 swap). Expect the
same ~1.1x ViT win (Linear-only). Low risk — all pieces exist.

**fp8 ViT with quantized attention (>1.5x), ~2–4 days, blocked:** needs an fp8
attention kernel. flash-attn is broken here (no `flash_attn_2_cuda`). Options:
build a TRT engine with fused fp8 MHA (Route 3) or fix flash-attn — both are
multi-day and uncertain in this container.

**Full NVFP4 ViT TRT engine (production, ~2–3 days):** the manual
`StaticFp4Linear` already gives 1.36x in torch eager. A real TRT engine (ONNX
export + modelopt ONNX QDQ + trtexec `--fp4`/`--stronglyTyped`) would also fp4
the attention GEMMs and remove eager overhead — but requires installing trtexec
+ surviving ViT ONNX export (rope/deepstack/dynamic-grid), which is the main
risk. Estimate 2–3 days incl. export debugging.

## Files
- `bench_vit_real_quant.py` — Route 1 modelopt compress bench (shows the
  real-but-slower wrapper result + the conv-disable fix).
- `vit_real_quant_linear.py` — `StaticFp8Linear` / `StaticFp4Linear` (the
  measured-speedup real-GEMM modules) + swap helpers.
- `results/vit_real_quant_speedup.json` — the consolidated measured numbers.
- `results/vit_real_quant.json`, `results/vit_real_quant_nvfp4.json` — Route 1
  modelopt runs (fp8 slower; nvfp4 version-skew crash).
