# TRT-LLM 1.3 vs HF transformers — apples-to-apples on B.5'' (Qwen3-VL-4B)

**Date**: 2026-05-23
**Ckpt**: `checkpoints_qwen25/nusc_planning_b5pp_1cam_qwen3vl_multimodal/final` (B.5'', Qwen3-VL-4B-Instruct, 1-cam camera + HD-map + bbox/ego SFT, L2=0.5557)
**Hardware**: RTX 5090 (compute cap 12.0), 32 GB HBM
**dtype**: bf16, batch=1, tensor_parallel=1
**Prompt**: 49 tokens, text-only (identical string across both benches)
**Decode**: 14 trajectory tokens, greedy
**n_warmup=3, n_runs=20**

| Metric | TRT-LLM 1.3.0rc15 (PyTorch backend) | HF transformers 5.6 SDPA | TRT speedup |
|---|---|---|---|
| TTFT mean | 34.79 ms | 28.08 ms | 0.81× (HF marginally faster) |
| TTFT p99 | 43.65 ms | 41.99 ms | — |
| per-token decode mean | 6.05 ms | 24.17 ms | **4.0×** |
| per-token decode p99 | 6.60 ms | 27.78 ms | **4.2×** |
| Full 14-tok mean | 113.5 ms | 342.3 ms | **3.0×** |
| Full 14-tok p99 | 117.7 ms | 391.8 ms | **3.3×** |
| Throughput mean | 123.4 tok/s | 41.0 tok/s | **3.0×** |
| Load seconds | 654.9 s | 2.75 s | (HF wins — TRT compiles kernels first time) |

## Honest read

**The speedup is real where it matters: the decode loop.**
- For a 14-tok trajectory output (the actual production payload size for a planning
  call), TRT cuts wall-clock from 342 → 113 ms, a 3.0× reduction. That maps to
  ~6 Hz vs ~2 Hz planning rate at single-shot batch=1.
- TTFT is a wash on a 49-token prompt: prefill is small enough that TRT's heavier
  PyTorch-backend dispatch overhead nets out slightly behind HF SDPA. This flips
  on long prompts (cf. multimodal prefill with thousands of visual tokens).
- Per-token decode is the headline 4× — TRT-LLM's paged KV cache + fused
  attention kernels dominate HF's per-step PyTorch overhead.

**What this bench does NOT cover (honesty):**
1. **Text-only**: both benches skip the visual encoder + projector forward. For
   B.5'' the multimodal prefill is ~6k visual tokens (1 cam × 4 frames) plus HD
   map + bbox text — that prefill is NOT free, and the TRT path here cannot serve
   Qwen3-VL multimodal inputs through the `LLM.generate()` API today. Real
   production deploy needs either:
   - A custom TRT engine that includes the vision tower, OR
   - HF vision encoder upfront + TRT for text continuation.
2. **Batch=1**: TRT's advantage compounds at higher batch sizes (paged attention
   amortizes KV cache better). Not measured here.
3. **No INT8/FP8 quant**: bf16 is the floor. INT8 SmoothQuant on TRT-LLM
   typically gives another 1.5-2× on RTX 5090; not attempted (would need
   calibration set).
4. **Cold load**: TRT first-time engine compile is 655 s — only relevant if
   you redeploy frequently. Warm reload from cache is ~5 s.

## Files
- `B5pp_trt_qwen3vl_bf16.json` — raw TRT measurements
- `B5pp_hf_qwen3vl_bf16.json` — raw HF measurements
- `B5prime_hf_bf16.json` — older HF bench on B.5' Qwen2.5-VL 3-cam (different model — NOT comparable to TRT row above; kept for reference)
