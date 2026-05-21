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
