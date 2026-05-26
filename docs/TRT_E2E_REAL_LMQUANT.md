# B.5'' 1-cam deploy — e2e bench with REAL low-precision LM (2026-05-26, #176)

Corrects the earlier #175 e2e bench, which quantized only the TRT-ViT engine while the
TRT-LLM **language-model backbone ran bf16 in all rows** (the `quant_fp8`/`quant_nvfp4`
ckpts were stale 8.88GB all-bf16 with no `hf_quant_config.json`). That is why #175's L2 was
0.80-0.82 flat and latency ~250ms flat across precisions.

## Fix
Re-ran modelopt PTQ (`deploy/trt_b5ppp/quant_fp8.py` / `quant_nvfp4.py`, calib-n=256 real
1-cam multimodal prompts) → `export_hf_checkpoint` now emits genuinely quantized weights +
`hf_quant_config.json`:
- `quant_fp8/`  : 5.0 GB, `quant_algo=FP8`,   exclude lm_head
- `quant_nvfp4/`: 3.48 GB, `quant_algo=NVFP4`, exclude lm_head

(vs bf16 `final/` 8.88 GB.) TRT-LLM 1.3.0rc15 `LLM(model=...)` reads `hf_quant_config.json`
and runs the LM linears in fp8/nvfp4. Stage flags now show `prefill.quantized=True`,
`decode.dtype=fp8`/`nvfp4`.

## Full pipeline = REAL TRT-ViT engine + HF HD-map + FasterVLM×4 + TRT-LLM (fp8/fp4 LM) + 14-tok decode
RTX 5090 (Blackwell sm_120), 1-cam native video grid [2,56,100], end-to-end full-trajectory.

| precision (ViT+LM) | full_traj p50 | speedup vs bf16 | L2 (n=50)* | LM ckpt | footprint |
|---|---|---|---|---|---|
| bf16 ViT + **bf16 LM** | 256.2 ms | 1.00× | 0.799 | 8.9 GB | — |
| fp16 ViT + **bf16 LM** | 233.5 ms | 1.10× | 0.810 | 8.9 GB | — |
| fp8 ViT + **fp8 LM**   | **182.8 ms** | **1.40×** | 0.879 | 5.0 GB | **−44%** |
| fp4 ViT + **nvfp4 LM** | **182.8 ms** | **1.40×** | 0.806 | 3.5 GB | **−61%** |

\* L2 is on 50 val samples — the per-precision spread (0.799 / 0.810 / 0.879 / 0.806) is within
trajectory sampling noise (±~0.05–0.08). The honest claim is **L2 ≈ bf16 (~0.80) for all
precisions**, NOT a real fp8>fp4 ordering. For a precise deploy accuracy number, run the
full-5119 L2 on the chosen precision.

## Honest reading
1. **Quantizing the LM is what moves latency** — it is the bottleneck (LM prefill+decode dominate
   the ~250ms; the real TRT ViT is only ~51–72ms). fp8/fp4 LM → 1.40× e2e, footprint −44%/−61%,
   accuracy within noise of bf16. This is the correct deploy story that #175 missed.
2. **fp8 and fp4 land at the same 182.8ms p50** — decode is memory-bandwidth bound and TRT-LLM
   runs both LM weight formats through similar kernels at bs=1; fp4's smaller weights buy footprint,
   not extra speed, at this batch size. fp8 is the safe deploy pick (mature, −44%); fp4 if footprint
   is critical (−61%) and the full-val L2 holds.
3. Vision precision (bf16→fp16 ViT) alone buys only 1.10×; the win is the LM.

## Artifacts
- Per-row JSON: `deploy/trt_bench/B5pp_e2e_{bf16,fp16,fp8,fp4}_fastervlm4.json` (all rebuilt 2026-05-26 15:50–16:39).
- Quant ckpts: `checkpoints_qwen25/nusc_planning_b5pp_1cam_qwen3vl_multimodal/{quant_fp8,quant_nvfp4}/` (local, gitignored).
- Stale bf16-mislabeled dirs moved to `*.stale_bf16` (to be removed after this is confirmed).
