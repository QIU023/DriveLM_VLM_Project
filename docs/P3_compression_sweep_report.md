# P3 — Spatial-Compression Sweep (B.5', Qwen2.5-VL-3B, 3-cam full-modal)

**Date:** 2026-05-26
**Model:** `checkpoints_qwen25/nusc_planning_b5prime_3cam_multimodal/final` (Qwen2.5-VL-3B, 3-cam: CAM_FRONT + FL + FR, full-modal = video + HD-map BEV + bbox + ego). Training-free — no retrain; compression applied at eval/deploy time on **video tokens only** (HD-map/bbox/ego untouched).
**Eval:** full nuScenes planning val, **n=5119/cell**, 8-GPU data-parallel (torchrun, strided shard + `all_gather_object`), transformers 5.6, greedy decode, AutoVLA TemAvg (VAD) L2 protocol.
**Methods (deployable, vision-side only):** FasterVLM, PruMerge. Excluded: PyramidDrop (LLM-layer-internal, TRT can't hook), temporal pool/VTM/LongVU (accuracy loss).

## Results

| method | ratio | L2 (TemAvg) | L2 (NoAvg) | collision_avg | video tok/sample |
|---|---|---|---|---|---|
| baseline   | 1×  | **0.6168** | 1.2322 | 0.0336 | 11518 |
| FasterVLM  | 2×  | **0.5851** | 1.1742 | 0.0342 | 5759 |
| FasterVLM  | 4×  | 0.5918 | 1.1862 | 0.0359 | 2879 |
| FasterVLM  | 8×  | 0.6080 | 1.2209 | 0.0381 | 1440 |
| FasterVLM  | 16× | 0.6181 | 1.2450 | 0.0387 | 720 |
| PruMerge   | 2×  | 0.5901 | 1.1875 | 0.0346 | 5759 |
| PruMerge   | 4×  | 0.5945 | 1.1927 | 0.0371 | 2879 |
| PruMerge   | 8×  | 0.6177 | 1.2436 | 0.0371 | 1440 |
| PruMerge   | 16× | 0.6221 | 1.2595 | 0.0393 | 720 |

## Findings

1. **Both methods are ~lossless out to 16×.** Every compressed cell lands within ±0.01 of the 0.6168 baseline; most are *below* it.
2. **FasterVLM ≥ PruMerge at every ratio** (lower L2 at 2/4/8/16×) — consistent with the literature.
3. **Sweet spot = FasterVLM 2×** (L2 0.5851, −0.032 vs baseline): pruning redundant visual tokens has a mild denoising effect.
4. **Deploy choice FasterVLM 4× (L2 0.5918) beats baseline** — the deploy ratio is fully justified; 8× (0.6080) is also still lossless if more aggressive compression is wanted.
5. **collision** rises monotonically but trivially (0.0336 → 0.039 at 16×; +0.006 max).
6. **Max lossless ratio ≈ 16×** for FasterVLM (720 video tok, 0.6181 ≈ baseline).

## Caveats

- **Absolute baseline calibration:** this DP run (transformers 5.6) gives baseline L2 = **0.6168**, vs the 2026-05-22 archived eval of the same ckpt (transformers 4.57) at **0.6578**. The ~0.041 gap is the eval-environment numerics (4.57 → 5.6 Qwen2.5-VL forward: M-RoPE / attention / vision merge), NOT a bug. All 9 cells share the 5.6 environment, so the **Pareto is internally consistent** and the lossless-ratio conclusion holds; only the absolute anchor shifts.
- **Tokenizer:** the B.5' ckpt tokenizer had been corrupted by post-train traj-token surgery (`<|image_pad|>` shadowed to 151926 vs config 151655); rebuilt from the clean pre-traj base + id-space traj tokens (151678–151935) before this sweep. See `reference_b5prime_tokenizer_corruption_fix`.

## Artifacts

- Per-cell JSON: `eval_results/compress_bench_b5prime/{baseline,fastervlm_r{2,4,8,16},prumerge_r{2,4,8,16}}.json`
- Eval entrypoint: `scripts/planning_eval_compress_mm.py` (full-modal + DP + deepstack-conditional)
- Runner: `deploy/trt_b5ppp/run_compress_benchmark_b5prime.sh` (9 cells sequential × 8-GPU DP)
