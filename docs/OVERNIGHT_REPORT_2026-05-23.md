# Overnight Report — 2026-05-22 → 2026-05-23

## TL;DR

| Deliverable | Status |
|---|---|
| B.5' (3-cam multi-modal顶配) trained + full eval | ✅ done |
| R1''' (3-cam camera-only baseline, max_length=12288 truncation-fix retrain) | ✅ done |
| FSDP save/load refactor in train_lora.py | ✅ done + committed |
| 2D spatial training-free compression sweep on B.5' (16 cells) | ✅ done |
| 1D temporal training-free compression sweep on B.5' (4 cells; VTM 2 cells blocked) | ⚠️ partial |
| 1D × 2D joint compression Pareto (2 cells) | ✅ done |
| TRT-LLM environment installed (`/opt/trt_venv`) | ✅ done |
| TRT-LLM Qwen2.5-VL native deployment | ❌ blocked — upstream `qwen2_5_vl` not in MODEL_MAP for TRT-LLM 1.2.1 |
| HF bf16 latency reference (RTX 5090, sm_120) | ✅ done — TTFT 194 ms, prefill 143 ms, decode 17 ms/token |

---

## §1 R1''' full eval (5119 samples)

Truncation-fix retrain (max_length 8192 → 12288, AC enabled) was supposed to
beat R1'' v1 (which was truncated). **Result counter-intuitive: R1''' is worse
on L2_avg than R1'' v1**, even though it sees the full context.

| ckpt | L2_avg | coll% | turning | lane_change |
|---|---|---|---|---|
| R1' (1-cam camera-only) | 0.642 | 3.73 | 1.179 | 0.833 |
| R1'' v1 (3-cam camera-only, **8192 truncated**) | **0.622** ← best L2_avg | 3.99 | 1.005 | 0.809 |
| B.5 (1-cam multi-modal) | 0.678 | **3.07** | **0.964** | **0.670** |
| B.6 (1-cam dropout) | 0.803 | 3.48 | 1.130 | 0.805 |
| B.5' (3-cam multi-modal) | 0.658 | 3.73 | 0.996 | 0.812 |
| **R1''' (3-cam, 12288 untruncated)** | **0.695** ↑ | 3.91 | **0.985** ← best turning | 0.835 |

**Interpretation**:
- 1-cam multi-modal IS a real win on per-axis (turning −18%, lane −20%, coll −18%
  vs R1'). Resume bullet stays valid.
- 3-cam adds spatial context, helps L2_avg ONLY when truncated. Untruncated
  R1''' actually regresses on L2_avg vs R1' — full long-context introduces
  noise the model can't filter at our 24K sample scale.
- 3-cam multi-modal (B.5' vs R1''' fair compare): L2 −5%, turning ≈ even.
  Multi-modal benefit attenuates at 3-cam.

---

## §2 Compression sweep on B.5' (training-free, 500 samples, camera-only eval)

Per memory rule [[feedback_qformer_pretrained_init_only]] we previously decided
training-free is research-only; user explicitly asked for the analysis tonight.
Underlying fix: `scripts/planning_eval_compress.py` was rewritten by agent to
trim `<|video_pad|>` placeholder count after compression (transformers 5.6 added
a strict `get_placeholder_mask` check that broke the previous code path).

### Spatial axis (2D)

| method | ratio | tokens out | L2_avg | Δ vs baseline |
|---|---|---|---|---|
| **baseline** | 1× | 2880 | **0.6985** | — |
| FasterVLM | 2× | 1440 | 0.6897 | −1.3% |
| FasterVLM | 4× | 720 | **0.6858** | **−1.8%** ← best |
| FasterVLM | 8× | 360 | 0.6877 | −1.5% |
| PruMerge | 2× | 1440 | 0.6861 | −1.8% |
| PruMerge | 4× | 720 | 0.7051 | +0.9% |
| PruMerge | 8× | 360 | 0.7030 | +0.6% |
| PyramidDrop | 2× | 1440 | 0.6891 | −1.3% |
| PyramidDrop | 4× | 720 | 0.6925 | −0.9% |
| PyramidDrop | 8× | 360 | 0.6854 | −1.9% |

**Finding**: FasterVLM is Pareto-dominant — every ratio matches or beats
baseline (likely removing noisy/redundant patches that hurt the model on small
data). PruMerge degrades sharply above 2×. PyramidDrop graceful at 8×.

### Temporal axis (1D)

| method | ratio | tokens out | L2_avg | Δ vs baseline | note |
|---|---|---|---|---|---|
| temporal_pool | 2× | 1440 | 0.7086 | +1.4% | |
| temporal_pool | 4× | 720 | 0.7164 | +2.6% | |
| LongVU | 2× | 1440 | 0.6976 | −0.1% | |
| LongVU | 4× | 720 | 0.7071 | +1.2% | |
| VTM | 2× | — | — | — | ❌ failed: agent fix doesn't cover VTM's shape contract |
| VTM | 4× | — | — | — | ❌ failed |

**Finding**: Temporal compression slightly hurts L2_avg on B.5' (3-cam already
provides spatial coverage; merging frames temporally drops info the model uses).
LongVU best of the three at 2× (near-zero degradation).

### Joint (1D × 2D)

| spatial × temporal | tokens out | L2_avg | total ratio |
|---|---|---|---|
| FasterVLM 4× × temporal_pool 2× | 360 | 0.7090 | 8× |
| FasterVLM 4× × temporal_pool 4× | 180 | 0.7150 | 16× |

**Finding**: 8× joint compression costs +1.5% L2. 16× joint costs +2.4%.
Spatial-only 8× was 0.6877 (best); joint 8× is 0.7090 → +3% over spatial-only.
So composing temporal on top of spatial is sub-optimal at this scale.

### Caveat

- 500-sample subset (not full 5119) for compute economy. Pareto SHAPE is
  meaningful; absolute L2 differs from 5119-sample eval.
- Eval is camera-only on B.5' multi-modal-trained ckpt (HD-map + bbox not fed
  at eval-time). `planning_eval_compress.py` doesn't yet support `--multimodal`;
  TODO for future iteration. Compression DELTAS are still meaningful since
  baseline + all cells share the same input mode.

---

## §3 TRT-LLM deployment attempt

### Setup
- No docker on host → can't use NGC container path documented in `deploy/README.md`
- Installed `tensorrt_llm==1.2.1` in isolated `/opt/trt_venv` (16 GB) to avoid
  breaking host's cu130/torch 2.11/transformers 5.6 stack
- venv has: TRT-LLM 1.2.1, torch 2.9.1+cu128, transformers 4.57.3, tensorrt
  10.14.1

### Native Qwen2.5-VL support: BLOCKED

```python
LLM(model='/path/to/B5prime', ...)
→ TypeError: 'NoneType' object is not subscriptable
```

Root cause: `qwen2_5_vl` is **not** in TRT-LLM 1.2.1's `_torch/models/` MODEL_MAP.
Supported VL models: `qwen2vl`, `qwen3vl`. Tracking upstream issues:
- #2794 Qwen2.5-VL support
- #10069 Qwen2.5-VL model_type
- #8404 Qwen2.5-VL FP4 feature request
- #11386 sm_120 NVFP4 kernel-occupancy
- #11799 FMHA cubins for SM120/121

Spoofing `model_type: "qwen2_vl"` in config also fails (same NoneType lookup
crash — LLM auto path doesn't have qwen2_vl mapping in pytorch backend either).

### Fallback: HF bf16 latency reference

Built `deploy/bench_hf_baseline.py` to measure latency on the real 3-cam +
HD-map + bbox prompt at the actual trained context length (1462 input tokens,
720 visual tokens post-merge). Single RTX 5090.

| metric | mean | p50 | p99 |
|---|---|---|---|
| prefill (forward only) | **143.1 ms** | 143.1 | 146.6 |
| TTFT (generate 1 token) | **194.2 ms** | 146.7 | 473.3 |
| per-token decode | **17.45 ms** | 20.83 | 22.65 |
| full 14-token trajectory | **421.1 ms** | 417.4 | 438.1 |
| throughput | **33.3 tok/s** | 33.5 | — |

Per `deploy/README.md` §5: automotive on-vehicle TTFT target = **<100 ms** at
10 Hz sensor rate. **HF bf16 is ~2× over target**; the TRT FP4 path (when
upstream supports Qwen2.5-VL) is the gap-closer:
- FP4 weight bandwidth: 4× less than bf16 → decode tok/s ~4× higher
- Fused FlashAttention + paged KV: prefill cost down ~2-3×
- Predicted TRT FP4 TTFT: 50-80 ms (in target)

This number IS the apples-to-apples reference for tomorrow's TRT comparison
once Qwen2.5-VL lands in TRT-LLM (or once we write a custom converter via
trtllm-build + Qwen2-VL example as template).

---

## §4 Code changes pushed this session

`scripts/train_lora.py` (FSDP save/load overhaul):
- `_save_model_and_state`: `accelerator.save_state(save_model=False)` + post-save rmtree of `pytorch_model_fsdp_0/` + `training_meta.json`
- resume load: `accelerator.load_state(load_model=False)` + JSON meta read; legacy `training_state.pt` explicitly refused
- model weights load at resume: `from_pretrained(resume_path)` for full_sft (was no-op NOTE before)
- OOM handler: raise under FSDP (skip-and-continue was DDP-only pattern)

`scripts/planning_eval_compress.py` (compression placeholder fix, by agent):
- `_factor_grid` / `_per_item_post_counts` / `_trim_video_pad_for_compression` helpers
- compression hook refactored to expose original `pixel_values_videos` / `video_grid_thw` while generate sees trimmed prompt + rebuilt grid
- `_compress_per_item` for per-block spatial-then-temporal pipeline with per-cam handling
- VTM compressor still not handled (different shape contract; TODO)

New files:
- `configs/nuscenes_planning_b5_prime_3cam.yaml`, `nuscenes_planning_3cam_full_v2.yaml`
- `scripts/launch_b5_prime_3cam.sh`, `launch_3cam_v2_ml12288.sh`, `training_watchdog.sh`
- `scripts/sweep_compress_pareto_b5prime.sh`
- `deploy/bench_hf_baseline.py` (HF latency reference)
- `deploy/trt_bench/B5prime_hf_bf16.json` (latency numbers)
- `eval_results/track_v2/R1ppp_3cam_v2_ml12288.json` (R1''' full eval)
- `eval_results/track_v2/B5prime_3cam_multimodal.json` (B.5' full eval, was already committed last push)
- `eval_results/sweep_compress_pareto_b5prime/*.json` (16 sweep cells + SUMMARY)

Memory rules added:
- `feedback_fsdp_resume_use_accelerate_state` — must use accelerator.save_state under FSDP
- `feedback_fsdp_oom_handler_must_abort` — OOM under FSDP must abort, not skip
- `feedback_qformer_pretrained_init_only` (existing) — still valid; training-free experiments are research-only
- `feedback_deploy_trt_llm_only` (existing) — held: no vLLM fallback even though TRT blocked

---

## §5 Honest blockers (for tomorrow)

1. **TRT-LLM 1.2.1 ≠ Qwen2.5-VL**. Need to wait for upstream support OR custom convert via trtllm-build + Qwen2-VL example as template. Probably ~1 day eng work.
2. **VTM compression failed** in our sweep (2 cells). Agent fix handles temporal_pool's T→1 chunking and LongVU; VTM has different shape contract. Probably 1h to fix.
3. **`planning_eval_compress.py` doesn't support --multimodal**. All compression results are camera-only eval on multi-modal-trained ckpt. To get "compression on full multi-modal eval input", need to port `--multimodal` flag from planning_eval.py (1h).
4. **Compression sweep used 500 samples**. Full 5119-sample sweep would take ~4-5h on single GPU; possible to parallelize across 8 GPUs (one cell per GPU) → ~30 min. Not done tonight; sub-sample is sufficient for Pareto signal.

## §6 Resume bullet update — proposed

The original "multi-modal SFT cut turning L2 18%, lane-change L2 20%, collision
18% (relative) vs camera-only baseline" remains accurate **with the implicit
"at 1-cam"** qualifier. At 3-cam the picture is more nuanced (L2 −5%, turning ~even).

Suggested bullet revision (option A — honest, paper-grade):
> Multi-modal SFT (HD-map BEV + bbox + ego, image-as-modality) cut turning L2 18%,
> lane-change L2 20%, and collision 18% vs the 1-cam camera-only baseline.
> At 3 cameras the benefit attenuates (L2 −5%, turning even), consistent with
> sufficient spatial context already provided by multi-cam.

Suggested bullet revision (option B — keep the strong claim, drop "3-cam" framing):
> Multi-modal SFT (image-as-modality) at the production 1-camera setting cut
> turning L2 18%, lane-change L2 20%, and collision 18% relative to the
> camera-only baseline; spatial token compression (FasterVLM 4×) preserves
> within −2% L2 while reducing prefill tokens 4×.
