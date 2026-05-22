# DriveLM VLA — TensorRT-LLM Deployment

FP4 / INT4 deployment of the fine-tuned Qwen2.5-VL VLA on Blackwell consumer
hardware (RTX 5090, sm_120), used as a **local preview of the on-vehicle
DRIVE Thor (also Blackwell, FP4-native) production path**.

> Why FP4, not INT4-AWQ: NVFP4 is Blackwell-native (5090 + Thor). Deploying on
> 5090 in FP4 exercises the **same datatype** the production chip uses. Ada
> cards (4070Ti) do NOT have FP4 tensor cores — that's why deployment moved to
> the 5090.

---

## 0. Hardware / precision matrix

| Target | Arch | FP4? | Role |
|---|---|---|---|
| RTX 4070Ti | Ada `sm_89` | ✗ | (retired) old llama.cpp GGUF Q4 story |
| RTX 5090 | Blackwell `sm_120` | ✓ NVFP4 | **dev + FP4 engine + benchmark** |
| DRIVE Orin-X | Ampere | ✗ (INT8) | prod today — small models only (see §5) |
| DRIVE Thor | Blackwell | ✓ FP4 | prod target for 7B+ VLA |

**Engines are arch-specific.** An engine built on sm_120 will NOT run on sm_89
or on Orin. The *quantized checkpoint* is portable; the *engine* is not — rebuild
per target GPU.

---

## 1. Container (Blackwell needs CUDA 12.8 + TRT-LLM ≥ 1.0)

```bash
docker run --rm -it --gpus '"device=0"' --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /path/to/models:/models -v $PWD:/work \
  nvcr.io/nvidia/pytorch:25.01-py3 bash

pip install --upgrade pip && pip install tensorrt_llm
python -c "import tensorrt_llm as t; print('TRT-LLM', t.__version__)"
python -c "import torch; print('cap', torch.cuda.get_device_capability())"  # (12,0) on 5090
```

⚠️ **VERIFY before quantize/build**: clone the examples at the *exact* installed
version tag, then locate the Qwen2-VL multimodal example (its path moved across
versions):

```bash
git clone https://github.com/NVIDIA/TensorRT-LLM.git /work/TRT-LLM
cd /work/TRT-LLM && git checkout v$(python -c "import tensorrt_llm;print(tensorrt_llm.__version__)")
ls examples/ | grep -iE "qwen|multimodal"     # paste this back to lock exact flags
```

---

## 2. Pipeline (scripts in this folder are scaffolds — fill exact flags per §1)

```
merged HF model ──quantize_fp4.sh──> NVFP4 ckpt ──build_engine.sh──> .engine ──benchmark.py──> tok/s + TTFT
   (LLM part)        (modelopt)        (portable)    (per-arch)
       │
   vision encoder ──> separate TRT engine (built by the multimodal example script)
```

- `quantize_fp4.sh` — modelopt NVFP4 quantization of the **language_model
  submodule** (vision tower stays bf16/fp16).
- `build_engine.sh` — `trtllm-build` the LLM engine + the vision engine.
- `benchmark.py` — TTFT + decode throughput harness (measurement logic is
  version-independent; wire the actual generate call per your TRT-LLM version).

See each script's header for the VERIFY markers.

---

## 3. What gets deployed

- **VQA model**: the LoRA-merged DriveLM 3B (run `scripts/merge_lora_to_base.py`
  first → standard HF dir). This is the throughput/TTFT benchmark target
  (replaces the old llama.cpp 170 tok/s / 142 ms story).
- **VLA model**: the full-SFT planning 3B (already a plain HF dir). Generates
  short trajectory-token sequences; useful for an end-to-end latency number.

---

## 4. TRT engine + C++ runtime (mental model)

- **Engine** = AOT-compiled, arch-locked artifact. Build-time passes: kernel
  fusion, **per-GPU tactic auto-tuning** (why it's arch-specific), per-layer
  precision (FP4/INT8/...), static activation-memory planning (zero malloc at
  runtime).
- **TRT C++ runtime**: `IRuntime → deserializeCudaEngine → ICudaEngine →
  IExecutionContext`; bind device buffers via `setTensorAddress`, pick dynamic
  shape via `setInputShape`, launch async with `enqueueV3(stream)`. Links
  `libnvinfer` + `libnvinfer_plugin` + CUDA runtime.
- **TRT-LLM runtime** (on top of TRT): `libtensorrt_llm` adds paged KV cache,
  in-flight batching, sampling, the decode loop. Modern C++ entry point is
  `tensorrt_llm::executor::Executor` (predecessors: `GptManager`/`GptSession`);
  Triton's trtllm backend wraps it. Core plugins: `gpt_attention_plugin`
  (fused attn + KV), `gemm_plugin`.

---

## 5. Production context — Orin/Thor (the "why" behind the design)

Automotive inference is **batch=1, latency-bound** (one ego vehicle, fixed
sensor rate, can't batch across time) → pure **memory-bandwidth-bound** decode,
the worst case for GPU utilization.

**30B dense on dual Orin-X (254 TOPS, 64GB, ~204 GB/s each) — does NOT fit
real-time:**
- decode tok/s ≈ BW / weight_bytes; INT4 30B = 15GB → ~13 theoretical, ~8–10
  realistic → a 50-token plan ≈ 5 s vs the ~100 ms (10 Hz) budget → ~50× too slow
- prefill of ~1k tokens alone ≈ 0.5 s on Orin's dense compute
- two Orins **don't share memory** and have **no NVLink** → can't TP; arrange as
  **functional pipeline** (Orin-A: vision encoder + perception; Orin-B:
  LLM/planning), passing only **compressed visual tokens** across the chip link

**Levers to make a VLA fit on-vehicle** (and how this repo's work maps). The
action path emits a short trajectory → **prefill-bound, not decode-bound**, so
optimize prefill first:
- **Small DENSE model** — the standard for real-time driving VLA (OpenVLA, AutoVLA,
  π0, EMMA, DriveVLM are all dense). FFN-MoE is the *wrong* tool: it's a decode-
  throughput lever on a prefill-bound workload, all experts stay resident (no edge
  memory saving), and routing adds latency variance (bad for hard real-time). MoE
  in driving is 2025 research only, and uses *structured* experts (DriveMoE
  scene/skill, AutoMoT fast-slow MoT), not generic token routing.
- **Visual-token compression** (this repo) — prefill ∝ tokens, VLA context is
  vision-heavy → biggest prefill win + minimizes inter-chip transfer
- **W4A4 / FP4 + KV-cache quant** — fewer weight bytes over LPDDR
- **Distillation** (this repo's SATS-CRP KD) — large teacher → small dense student

**Orin → Thor**: 30B-class VLA targets **Thor (Blackwell, FP4-native, ~1000
TOPS INT8 / ~2000 TFLOPS FP4, much higher BW)**. The 5090 FP4 deployment here is
a faithful local preview of that path.

---

## 6. Running the full pipeline tomorrow

Single end-to-end command, run **inside the NGC container** (do NOT run on
the training host — see §1 for bring-up):

```bash
# 0. Container bring-up (host)
docker run --rm -it --gpus '"device=0"' --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /workspace/DriveLM_VLM_Project:/work \
  -v /workspace/models:/models \
  -v /workspace/DriveLM_VLM_Project/checkpoints_qwen25:/ckpts \
  nvcr.io/nvidia/pytorch:25.01-py3 bash

# 1. Inside container — install TRT-LLM (~5 min first time)
pip install --upgrade pip && pip install tensorrt_llm
python -c "import tensorrt_llm as t; print('TRT-LLM', t.__version__)"
python -c "import torch; print('cap', torch.cuda.get_device_capability())"  # expect (12, 0)

# 2. Full pipeline (per checkpoint) — quant + LM engine + vision engine + benchmark CSV
cd /work
bash deploy/run_full_pipeline.sh \
    --hf-dir /ckpts/nusc_planning_b5prime_3cam_multimodal/final \
    --out    /models/engine_b5prime_5090 \
    --calib-n 512 \
    --bench-runs 50

# 3. Parity check (5 samples vs HF bf16) — both halves emit JSON the user combines
#    (a) on the training host, in a separate shell, when training is done:
#        /usr/bin/python3 deploy/parity_check.py --hf-only \
#            --hf-dir checkpoints_qwen25/nusc_planning_b5prime_3cam_multimodal/final \
#            --tokenizer-dir /workspace/models/Qwen2.5-VL-3B-Instruct
#    (b) in the container:
bash deploy/run_full_pipeline.sh ...   # already done above
python deploy/parity_check.py --trt-only \
    --engine-dir /models/engine_b5prime_5090/engine \
    --vision-engine-dir /models/engine_b5prime_5090/vision \
    --tokenizer-dir /ckpts/nusc_planning_b5prime_3cam_multimodal/final
```

### Time budget (3B Qwen2.5-VL on a single 5090, TRT-LLM 1.x as of May 2026)

| Stage | Wall time | Notes |
|---|---|---|
| TRT-LLM container first-time install | 5-15 min | only once per box |
| `quantize_fp4.sh` (modelopt NVFP4, calib=512) | 5-10 min | calib activations are the long pole; smaller = faster but lower fidelity |
| `build_engine.sh` (LM trtllm-build) | 3-8 min | sm_120 has *known* slow template compile per [#11386](https://github.com/NVIDIA/TensorRT-LLM/issues/11386); first build of a new shape combo can hit ~8 min |
| Vision engine build | 1-2 min | small ViT relative to LM |
| `benchmark.py` (50 runs) | 3-5 min | 1 s/run on hot KV |
| `parity_check.py` (5 samples × both engines) | 5-10 min | HF bf16 forward is the slow side |
| **Total per checkpoint, cold** | **~30-50 min** | once installed |

### What "good" looks like in `benchmark.py` output

| Metric | Pass | Comment |
|---|---|---|
| p99 TTFT | **< 100 ms** | automotive 10 Hz gate (§5). PASS line is printed automatically. |
| mean TTFT | < 60 ms | for 3B + ~2k vision tokens this is what FP4 + the gemm/attn plugin should hit |
| decode | > 200 tok/s | mem-BW bound at batch=1: ~1.8 TB/s 5090 BW / ~1.5 GB FP4 weights ≈ 1200 tok/s ceiling; 200+ is healthy after plugin overhead |
| total request (prompt + 14 traj tokens) | < 150 ms | end-to-end planning cycle latency |

If TTFT is way over budget, the next levers in order are: (a) ensure
`--gemm_plugin auto` was passed to `trtllm-build` (it is by default in our
`build_engine.sh`); (b) shrink the visual-token count via the compression
work in this repo (the real point of the project); (c) move
`--max_input_len`, `--max_seq_len` down to the actual prompt sizes the val
set uses (we leave headroom at 4096/4608).

### How to read `parity_check.py` output

The script writes three files under `deploy/parity_out/`:
- `parity_hf.json` — HF bf16 reference per-sample (generated tokens, decoded
  waypoints, GT L2, top-20 logits at first 3 trajectory positions)
- `parity_trt.json` — TRT FP4 engine per-sample (same schema; logits N/A on
  TRT-LLM 1.x streaming runner — see comment in `parity_check.py`)
- `parity_combined.json` — only written when both halves ran in the same
  process; per-sample exact-match flag, first divergence position, delta-L2

| Token exact-match rate | Verdict |
|---|---|
| **5/5 (100 %)** | green; FP4 changed nothing the trajectory tokenizer can see |
| **4/5** | acceptable; one bin-flip on a noisy sample is well within FP4 quantization noise (each bin is ~0.4 m so a flip = ~0.2 m typical delta) |
| **3/5 or less** | red — re-run quantize with a larger calib set (`--calib-n 1024`), or fall back to FP8 mixed (`--qformat fp8`) and re-benchmark |
| **0/5 + delta_L2 ≫ HF L2 to GT** | FP4 collapse; abort and use the vLLM baseline (next section). |

`delta_l2` (TRT − HF, in metres) is the per-sample driving-relevant signal:
the L2 between predicted and GT trajectory should not move more than ~0.05 m
on average compared to the bf16 reference. If it does, the FP4 engine is
producing trajectories that drift from the eval numbers we measured during
training — do NOT ship that engine to the demo, regardless of token-match
rate.

### Fallback: vLLM baseline if TRT-LLM 1.x blocks

TRT-LLM Blackwell-consumer support is still flaky as of May 2026 —
specifically:
- sm_120 NVFP4 builds slow + emit kernel-occupancy warnings ([#11386](https://github.com/NVIDIA/TensorRT-LLM/issues/11386))
- Qwen2.5-VL multimodal isn't fully in the MODEL_MAP yet (Qwen2-VL is;
  Qwen2.5-VL is a community PR — see [#2794](https://github.com/NVIDIA/TensorRT-LLM/issues/2794),
  [#10069](https://github.com/NVIDIA/TensorRT-LLM/issues/10069))
- trtllm-gen FMHA cubins for SM120/121 are still being filled in ([#11799](https://github.com/NVIDIA/TensorRT-LLM/issues/11799))

If `run_full_pipeline.sh` errors at step 2 or 3 with something that looks
like a missing kernel or unsupported model_type, fall back to vLLM — it's
~70-80 % of the TRT FP4 perf but installs cleanly on host cu130:

```bash
# Host — no container needed
/usr/bin/python3 -m pip install vllm --pre   # check Blackwell release notes
bash deploy/vllm_baseline.sh \
    /workspace/DriveLM_VLM_Project/checkpoints_qwen25/nusc_planning_b5prime_3cam_multimodal/final \
    --quantization awq --max-model-len 4608
# Then point benchmark.py at the vLLM OpenAI-compatible endpoint (port 8000).
```

The demo deck table can quote both: "TRT-LLM FP4 (target)" + "vLLM AWQ
(fallback)" — both are valid Blackwell-FP4-era numbers.

