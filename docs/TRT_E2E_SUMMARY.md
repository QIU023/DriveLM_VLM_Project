# TRT End-to-End Deploy Bench — B.5'' Qwen3-VL-4B 1-cam full-modal (2026-05-26)

UNIFIED end-to-end measurement (#175): the FULL deployed planning path, with the
camera-video vision stage running through the **real serialized TensorRT ViT
engine** (not HF, not fake-quant). This supersedes the prior
`bench_full_pipeline.py`, whose vision stage ran on HF torch.

## Pipeline measured (per sample, every stage timed)

```
raw 1-cam video pixel_values_videos [11200,1536]  (fixed grid [2,56,100])
  -> [TRT ViT engine]   pooler [2800,2560] + 3 deepstack [2800,2560]
                        -> concat dim=1 -> video mm_embedding [2800,10240]
  +  HD-map image [484,1536] grid [1,22,22] -> [HF ViT bf16] -> 121 image tokens
  -> [FasterVLM]        prune VIDEO 2800 -> 700 (HD-map 121 untouched)
  -> [TRT-LLM Qwen3 LM] mm-disagg embedding injection -> prefill
  -> [decode]           greedy full 14-token trajectory
  => full_traj_ms = TRT-ViT(video) + HF(HD-map) + compress + LM(prefill+decode)
```

The HD-map image branch stays on the HF ViT because the TRT ViT engine has the
video grid `[2,56,100]` baked in as constants (it only runs the camera video).
`video_vision_ms` is the real TRT engine latency; `hdmap_vision_ms` is the small
(484-patch) HF cost — both reported separately and summed into `vision_total`.

Runner: `deploy/trt_b5ppp/bench_e2e_trt_vit.py` (RTX 5090 GPU 0, venv
`/venv/trt_llm`, TRT-LLM 1.3.0rc15, TensorRT 10.15.1.29). 2 warmup + 5 timed runs
for latency; L2 on **50 val samples** (TemAvg metres vs nuScenes GT waypoints).
JSON per row: `deploy/trt_bench/B5pp_e2e_{bf16,fp16,fp8,fp4}_fastervlm4.json`.

## Precision pairings (vision engine + LM engine)

| precision | ViT engine | LM ckpt | LM quant |
|---|---|---|---|
| bf16 | `engines/vit_bf16` | `final` | bf16 |
| fp16 | `engines/vit_fp16` | `final` | bf16 (fp16 ViT recommended vision) |
| fp8  | `engines/vit_fp8`  | `quant_fp8` | fp8 |
| fp4  | `engines/vit_fp4`  | `quant_nvfp4` | nvfp4 |

## Results — REAL TRT ViT engine + REAL TRT-LLM (50-sample L2)

All four rows: parity PASSED (first decoded token = traj_start 151934), video
compressed 2800 -> 700, HD-map 121 untouched, 50/50 L2 samples evaluated, 0
failures.

| precision | full_traj_ms | ViT(video) | HD-map(HF) | compress | LM prefill | LM decode(13 tok) | peak mem | L2_avg (50) | L2 1s/2s/3s |
|---|---|---|---|---|---|---|---|---|---|
| bf16 | 253.5 | 72.2 | 15.4 | 0.47 | 83.7 | 81.9 | 8.61 GB | 0.799 | 0.303 / 0.746 / 1.347 |
| **fp16** | 249.2 | **57.2** | 20.0 | 0.47 | 85.9 | 85.6 | 8.61 GB | **0.810** | 0.304 / 0.760 / 1.365 |
| fp8  | 243.7 | 61.7 | 15.8 | 0.47 | 85.5 | 80.3 | 8.63 GB | 0.823 | 0.298 / 0.768 / 1.404 |
| fp4  | 242.8 | 59.0 | 16.9 | 0.50 | 84.4 | 82.0 | 8.63 GB | 0.808 | 0.297 / 0.752 / 1.374 |

(ms; latencies are means over 5 timed runs. L2 is TemAvg-avg metres over the
first 50 val samples — same 50 across all rows, so directly comparable.)

## Key findings

1. **The unified chain works end-to-end with the real TRT ViT engine.** All
   precisions pass the parity gate (first token = traj_start 151934), confirming
   the engine's pooler+deepstack output, concatenated to `[2800,10240]`, feeds the
   FasterVLM + mm-disagg LM path correctly. The engine mm_embedding matches HF
   bf16 at rel-L2 0.047 (bf16) / 0.045 (fp16) / 0.142 (fp8) / 0.270 (fp4).

2. **Vision is NOT the bottleneck of the full trajectory.** The TRT ViT video
   stage is 57-72 ms of a ~250 ms end-to-end; the LM prefill (~85 ms) + decode
   (~80-85 ms for 13 tokens) dominate. So vision-engine precision moves
   `full_traj_ms` by only a few %.

3. **fp16 ViT is the fastest vision stage (57 ms) and the most accurate
   low-precision option.** fp8 ViT (61.7) and fp4 ViT do NOT beat fp16 — the ViT
   is attention-bound and all sub-fp16 GEMM quant converges to the fp16 graph
   floor (documented in TRT_VIT_ENGINE_SUMMARY.md). Quant's win on the ViT is
   footprint, not latency.

4. **L2 holds up across ALL precisions on these 50 samples** — bf16 0.799, fp16
   0.810, fp8 0.823, fp4 0.808 (all within ~2.4 cm of each other). Notably the
   fp8 LM does NOT collapse to the ~0.90 seen on a different earlier full-val run,
   and fp4 (fp4 ViT rel-L2 0.32 + nvfp4 LM) is statistically indistinguishable
   from bf16 on this set. The differences are within run-to-run noise at 50
   samples. (50-sample L2 is higher than the prior 20-sample 0.615 numbers because
   the first 20 val samples are easier — exactly why ≥50 was required.)

## Deploy recommendation

**fp16 ViT engine + bf16 LM** (the `fp16` row). Rationale:
- fp16 ViT is the fastest vision stage (57 ms, 1.96× vs HF) AND the most accurate
  low-precision ViT (engine rel-L2 0.031 pooler / 0.045 full mm-embedding).
- bf16 LM keeps trajectory quality at the bf16 reference; fp8/fp4 LM quant buys
  almost nothing on full_traj_ms (LM decode is memory-latency bound at batch 1,
  14 tokens) while adding accuracy risk on the full val set.
- fp8 is a reasonable footprint-driven alternative (ViT 434 MB vs 876 MB; L2 here
  0.823 ≈ bf16). fp4 (ViT 295 MB + nvfp4 LM) is also viable on this set (L2 0.808,
  full_traj 242.8 ms — the fastest end-to-end), but the larger ViT rel-L2 (0.32)
  means its accuracy margin should be re-checked on the FULL val benchmark before
  shipping. All four precisions land within ~11 ms of each other on full_traj_ms
  because the LM (not vision) dominates — so the choice is driven by accuracy +
  footprint, not end-to-end latency.

## Honest caveats

- The autotune/load of each TRT-LLM engine is ~16 min (reported in JSON as
  `TRT LM loaded in ~988s`); this is one-time engine build/profile cost, NOT part
  of `full_traj_ms`.
- `vision_total` includes the HD-map HF cost (~15-20 ms). A production deploy could
  also engine-ify the small HD-map ViT, but at 121 tokens it is a minor stage.
- L2 is on the first 50 val samples (fixed set, comparable across rows). It is NOT
  the full-val planning benchmark; treat it as a quantization-fidelity check, not
  the headline planning score.
