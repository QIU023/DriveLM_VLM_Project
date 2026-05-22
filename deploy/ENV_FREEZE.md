# VLA env freeze — 2026-05-22

Snapshot of the training + inference stack used for the v2 eval matrix
(R1' / R1'' / B.5 / B.6) and the upcoming TRT-LLM deployment. Reproduce from
`requirements.core.txt`; full 356-package pin in `requirements.full.txt`.

## Hardware

| Component | Value |
|---|---|
| GPU | RTX 5090 × 8 |
| Architecture | Blackwell, `sm_120`, capability `(12, 0)` |
| FP4 tensor cores | ✓ native NVFP4 |
| FP8 tensor cores | ✓ E4M3 / E5M2 |
| NVLink | ✗ (consumer Blackwell — no peer-to-peer, TP penalized) |

## System

| Layer | Version |
|---|---|
| NVIDIA driver | 580.105.08 |
| CUDA toolkit | 13.0.88 |
| Python | system `/usr/bin/python3` (per [[reference_torchrun_uses_system_python]]) |
| OS | Linux 6.8.0-87-generic |

> **Why system python, not venv**: torchrun child workers `exec /usr/bin/python3`
> directly, ignoring the venv interpreter. Install training deps with
> `/usr/bin/python3 -m pip install`, never into a venv.

## Critical deps

| Library | Version | Notes |
|---|---|---|
| torch | 2.11.0+**cu130** | matched to CUDA 13.0 / sm_120 |
| transformers | 5.6.0 | Qwen2.5-VL native support |
| accelerate | 1.13.0 | used in `train_lora.py` 8-GPU FSDP |
| peft | 0.19.1 | LoRA (track A); not used in full-SFT B.5/B.6 |
| flash-attn-4 | 4.0.0b11 | FA4 Blackwell beta — fallback to SDPA if loaded fail |
| bitsandbytes | 0.49.2 | 4-bit optional; not used in current full-SFT path |
| nuscenes-devkit | 1.2.0 | dataset loader |
| torchao | 0.17.0+cu130 | quantization (post-train) |

## NOT installed (deferred to deployment container)

| Library | Why deferred |
|---|---|
| tensorrt | host conflict with PyTorch CUDA 13.0; install inside NGC container |
| tensorrt_llm | same — see `deploy/README.md` §1 for `nvcr.io/nvidia/pytorch:25.01-py3` bring-up |
| nvidia-modelopt | inside TRT-LLM container only |
| vllm | optional alternative deployment (faster setup, ~70-80% TRT perf) |
| sglang | already linked at `/workspace/sgl-workspace-link` for fork dev |

## How to bring up a fresh box

```bash
# 0. Verify hardware
nvidia-smi  # expect sm_120 / Blackwell
nvcc --version  # expect 13.0+

# 1. Reinstall pinned deps (system python, not venv)
/usr/bin/python3 -m pip install -r deploy/requirements.core.txt

# 2. Verify torch sees Blackwell
/usr/bin/python3 -c "import torch; print(torch.cuda.get_device_capability(0))"  # (12, 0)

# 3. Verify HF transformers Qwen2.5-VL
/usr/bin/python3 -c "from transformers import Qwen2_5_VLForConditionalGeneration; print('ok')"

# 4. (Optional) FA4 kernel
/usr/bin/python3 -c "import flash_attn_4; print(flash_attn_4.__version__)"
```

## Reproduce a v2 eval result

```bash
bash scripts/run_all_ckpts_eval_v2.sh
# → eval_results/track_v2/{R1prime_1cam,R1prime_3cam,B5_*,B6_*}.json
```

Checkpoints:
- `checkpoints_qwen25/nuscenes_planning_3b_full_sft/final` (R1' / A.0)
- `checkpoints_qwen25/nuscenes_planning_3cam_3b_full_sft/final` (R1'')
- `checkpoints_qwen25/nusc_planning_b5_multimodal/final` (B.5)
- `checkpoints_qwen25/nusc_planning_b6_multimodal_dropout/final` (B.6)
