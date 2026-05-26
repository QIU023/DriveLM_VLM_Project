# TRT ViT Engine — Summary (B.5'' Qwen3-VL-4B vision tower, 2026-05-26)

Real serialized TensorRT engines for `model.model.visual` (Qwen3-VL-4B ViT: 24 blocks,
hidden 1024, 16 heads, patch 16, deepstack 5/11/17, out 2560) at the fixed deploy input
1-cam native video `video_grid_thw=[2,56,100]` → `pixel_values_videos [11200,1536]` →
`pooler [2800,2560]` + 3 deepstack. Measured via TRT Python runtime (8 warmup + 20 timed)
on RTX 5090 (Blackwell sm_120), TensorRT 10.15.1.29, venv `/venv/trt_llm`.

## Measured table (all REAL engines, not torch-eager / not fake-quant)

| engine | median ms | speedup vs HF bf16 eager (~100ms) | rel-L2 vs HF fp32 | size | verdict |
|---|---|---|---|---|---|
| bf16 TRT  | 69.1 | 1.46× | 0.065 | 876 MB | baseline-class |
| **fp16 TRT** | **51.4** | **1.96×** | **0.031** | 833 MB | **RECOMMENDED (fastest + most accurate)** |
| fp8 TRT   | 51.6 | 1.94× | 0.174 | **434 MB** | footprint −50%, accuracy moderate |
| fp4 TRT   | 51.9 | 1.92× | 0.320 | **295 MB** | footprint −66%, accuracy marginal |
| fp4_wa    | 52.1 | 1.96× | **2.25** | 293 MB | **BROKEN (weight-amax-only) — drop** |

results: `deploy/trt_b5ppp/results/vit_trt_{vit_bf16_vs_fp32,vit_fp16_vs_fp32,fp8,fp4,fp4_wa}.json`.

## Key findings (honest)
1. **All 4 precisions build + run as real TRT engines.** fp4 included (hard requirement met).
2. **fp8/fp4 do NOT beat fp16 on latency** (all ~51 ms). The ViT is **attention-bound**: TRT runs the
   MHA in fp16/bf16 regardless; the QDQ only quantizes the Linear GEMMs (qkv/proj/mlp). Quantizing
   Linears below fp16 yields no further speedup once attention dominates. (Matches the earlier
   torch-eager Linear-only ceiling ~1.5–2×.)
3. **Quantization's real win is FOOTPRINT, not latency:** fp8 434 MB / fp4 295 MB vs bf16 876 MB.
4. **Accuracy degrades with precision:** rel-L2 0.031 (fp16) → 0.174 (fp8) → 0.320 (fp4). fp4_wa
   (weight-amax-only, no per-tensor input scales) is numerically broken (2.25) — excluded.
5. **Deploy recommendation: fp16 TRT engine** (1.96×, rel-L2 0.031). Use fp8 only if model footprint
   matters; fp4 only if footprint is critical and 0.320 rel-L2 is tolerable.

## Engineering (how it was done with mature tooling)
- ONNX export of the data-dependent Qwen3-VL ViT: baked the grid-derived constants
  (pos-embed / rope cos-sin / cu_seqlens) for the fixed deploy grid [2,56,100] so the only graph
  input is `hidden_states`. Block-diagonal 2-frame attention (cu_seqlens=[0,5600,11200]) implemented
  as batched per-frame attention → fp32 export wrapper bit-exact vs HF (rel-L2 0.0).
- bf16/fp16: direct TensorRT 10 build (`build_vit_trt_engine.py`).
- fp8/fp4: ORT min-max calibration OOMs on the [2,16,5600,5600] attention scores (32 GB) — bypassed
  by computing per-tensor amax in PyTorch (`collect_vit_amax.py`) and injecting STATIC Q/DQ into the
  ONNX (`inject_vit_qdq.py` fp8 / `inject_vit_nvfp4.py` fp4) → strongly-typed TRT build
  (`build_vit_trt_static.py`), run via `run_vit_trt_static.py`.

## Status of the larger deploy chain
- ViT engines (this doc): DONE bf16/fp16/fp8/fp4.
- LM TRT engine (Qwen3-VL): quant_fp8/quant_nvfp4 ckpts exist.
- **#175 DONE:** UNIFIED end-to-end bench = TRT-ViT engine → FasterVLM (2800→700) → TRT-LLM Qwen3 →
  full-trajectory decode, per precision (latency + planning L2 on 50 val samples). All 4 precisions
  pass parity (first token = traj_start 151934) and run with the REAL TRT ViT engine + REAL TRT-LLM.
  Results + deploy recommendation: **`docs/TRT_E2E_SUMMARY.md`**; per-row JSON
  `deploy/trt_bench/B5pp_e2e_{bf16,fp16,fp8,fp4}_fastervlm4.json`; runner
  `deploy/trt_b5ppp/bench_e2e_trt_vit.py`. Headline: full_traj ~243-254 ms (LM-bound; the real TRT ViT
  is only 57-72 ms of it), L2 0.80-0.82 across all precisions; recommend fp16-ViT + bf16-LM.
- **#174:** Qwen3 3-cam AutoVLA-res retrain (240 tok/cam, LBS=4/no-AC/no-offload).
