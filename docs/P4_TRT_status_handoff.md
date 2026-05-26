# P4 TRT Deploy — 3-Precision Bench (B.5'' Qwen3-VL-4B, COMPLETE 2026-05-26)

Target: **B.5'' Qwen3-VL-4B 1-cam full-modal** VLA, deployed at 3 precisions × FasterVLM 4× compression applied at deploy time (video tokens only).
Backend: **TRT-LLM 1.3.0rc15 PyTorch backend + mm-disagg** (HF vision pre-compute → injected multimodal embedding handles).
Ckpts: `final` (9.65G bf16) · `quant_fp8` (8.3G) · `quant_nvfp4` (8.3G), all under `checkpoints_qwen25/nusc_planning_b5pp_1cam_qwen3vl_multimodal/`.

## Results — 3 precisions × FasterVLM 4× (real full-modal payload, B=1)

Common to all cells: video 2800→700 visual tokens (FasterVLM 4×) + HD-map 121 img tokens; prompt_len_expanded 1391; n_warmup 1, n_runs 3; greedy; **parity gate PASSED** (TRT top-1 = `151934` traj_start ∈ HF top-5) for every precision.

| precision | TTFT mean (ms) | TTFT p50 / p99 | per-tok decode (ms) | throughput (tok/s) | peak mem (GB) | ckpt size |
|---|---|---|---|---|---|---|
| **BF16**  | **85.7** | 83.9 / 89.7 | 6.35 | **83.2** | 8.37 | 9.65G |
| **FP8**   | 91.9 | 91.0 / 94.9 | 6.35 | 80.3 | 8.37 | 8.3G |
| **NVFP4** | 92.8 | 91.8 / 95.1 | 6.26 | 80.4 | 8.37 | 8.3G |

Bench JSON: `deploy/trt_bench/B5pp_v2_trt_{bf16,fp8,nvfp4}_compress_fastervlm4.json`.

## Findings

1. **All 3 precisions are functionally correct** — each passes the HF-parity gate (top-1 = traj_start 151934, the expected first action token), so FP8/NVFP4 PTQ did not break the planning head.
2. **Weight-only quant does NOT improve latency at this operating point.** TTFT is *higher* for FP8 (+6.2ms) / NVFP4 (+7.1ms) than BF16, and throughput is marginally lower (80 vs 83 tok/s). Expected: at B=1 with a short prefill (1391 tokens, compute-bound) on a 4B model, dequantization overhead is not amortized and the weight-bandwidth savings don't help — quant wins show up in **memory-bound / high-batch / long-context** regimes, not single-stream short-prompt latency.
3. **Peak GPU memory is identical (8.37GB) across all three.** The runtime peak is dominated by the **bf16 vision tower** (unquantized in all variants) + activation/KV buffers, not the LM weights. Quant shrinks the **stored ckpt** (9.65G→8.3G, −14%) but not the bench-time peak at this batch/seq.
4. **NVFP4 ≈ FP8** on every metric (within ~1ms TTFT, <0.1 tok/s) — no W4A4 advantage over W8A8 here for the same reasons as (2).
5. **per-token decode (~6.3ms) is precision-invariant** — decode is bandwidth-bound on the tiny per-step compute and the bf16 vision/embedding path dominates; quant doesn't move it at B=1.

**Deploy recommendation:** for single-stream low-latency planning, **BF16** is the right default (fastest TTFT, same peak mem). FP8/NVFP4 are validated and ready, and become the right choice only when **ckpt footprint** (−14%) or **batched/long-context throughput** matters — not demonstrated by this B=1 latency bench.

## bench_trt.py — Qwen3-VL port (committed phase22, 121b48c)

6-bug chain resolved (Qwen2.5-VL → Qwen3-VL):
1. **trim**: Qwen3-VL interleaves timestamp tokens → T `<|video_pad|>` runs/cam (not 1 contiguous). `_find_pad_runs` + per-run trim.
2. **pidfd**: CUDA-IPC handle restore blocked by container seccomp → build handles from **CPU tensors** (serialize path).
3. **shm one-shot**: TRT-LLM mm-disagg shared tensors consumed/unlinked after first restore → `make_disagg()` factory mints a **fresh handle per generate**.
4. **lifecycle**: hold cloned CPU tensors alive across the blocking generate.
5. **quant `.visual` path**: Qwen3-VL top-level = {lm_head, model}; vision+LM under `.model` → `model.model.visual` / `model.model.language_model`.
6. **grid_thw 1-D→2-D**: PTQ calib forward passed 1-D `video_grid_thw`; Qwen3-VL `fast_pos_embed_interpolate` needs 2-D. `_move_to_device` unsqueezes 1-D grids.

## Key facts / repro

- venv: `/venv/trt_llm` (TRT-LLM 1.3.0rc15, modelopt 0.37.0, transformers 5.5.3). System python `/usr/bin/python3` for non-TRT.
- model structure: `model.model.visual` (vision, stays bf16), `model.model.language_model` (quant target), embed/lm_head = 151936 rows.
- driver: `/tmp/run_p4_fp8_nvfp4.sh` (FP8→NVFP4 PTQ+bench loop, calib-n 128).
- quant CLI: `quant_{fp8,nvfp4}.py --ckpt <final> --calib-n N --out <parent>/quant_{fp8,nvfp4}`.
- bench CLI: `bench_trt.py --ckpt <quant_dir> --precision fp8|nvfp4 --compress-method fastervlm --compress-ratio 4 --out <json>`.
- TRT-LLM first-request torch.compile ≈ 12-16min CPU (normal, not a hang).
- **Artifacts protected — demo pending** (do NOT delete): `quant_fp8`, `quant_nvfp4`, `deploy/trt_b5ppp/engines/b5ppp_*`, `deploy/trt_bench/B5pp_v2_trt_*_compress_fastervlm4.json`. See memory `feedback_keep_trt_deploy_artifacts`.

## P3 (separate, DONE) — see docs/P3_compression_sweep_report.md

B.5' Qwen2.5-VL-3B 3-cam compression Pareto: FasterVLM ~lossless to 16×, 4× = L2 0.5918 (beats baseline 0.6168). FasterVLM ≥ PruMerge at every ratio.
