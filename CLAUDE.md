# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Efficient VLM 项目——专注小模型 + 快速微调 + 知识蒸馏/持续学习，不做通用多模态预训练。以自动驾驶（DriveLM）为切入点，同时覆盖多模态推荐和视频理解场景。

用户背景：有 Pattern Recognition 2023 发表的持续学习论文（feature-level + logit-level KD），熟悉知识蒸馏。

## Environment

- **GPU**: NVIDIA GH200 480GB (96GB HBM3 GPU memory)
- **Arch**: aarch64
- **Conda env**: `qwen25vl` (Python 3.10, torch 2.10.0+cu128, transformers 5.3.0, peft 0.18.1)
- **Activate**: `export PATH="/home/ubuntu/miniconda3/bin:$PATH" && conda activate qwen25vl`
- **HuggingFace**: logged in (gated repo access for OpenDriveLab/DriveLM)
- **Git remote**: `git@github.com:QIU023/DriveLM_VLM_Project.git` (SSH, user: QIU023)

## Directory Structure

```
DriveLM/
├── configs/
│   ├── gh200.yaml              # GH200 base config: bf16, bs=4, no quant, 1 epoch
│   ├── 4070ti.yaml             # 4070 Ti config: 4-bit quant, bs=1, grad_accum=8
│   ├── baseline.yaml           # Layer 2: no compression baseline (inherits gh200)
│   ├── avg_pool_c4.yaml        # Layer 2: avg_pool 4x compression
│   ├── fastervlm_c4.yaml       # Layer 2: FasterVLM 4x compression
│   ├── prumerge_c4.yaml        # Layer 2: PruMerge 4x compression
│   ├── pyramiddrop_c4.yaml     # Layer 2: PyramidDrop 4x compression
│   └── run_all.sh              # Run all 5 compression experiments sequentially
├── scripts/
│   ├── train_lora.py           # Main training script, reads --config YAML
│   ├── visual_compress.py      # Visual token compression methods (avg_pool/fastervlm/prumerge/pyramiddrop)
│   ├── demo_inference.py       # Eval script, supports --config + --lora
│   ├── convert_data.py         # DriveLM → Qwen conversation format
│   ├── train_lora_qwen35.py    # Qwen3.5-4B variant (not yet updated to YAML)
│   └── ...
├── data/
│   ├── QA_dataset_nus/v1_1_train_nus.json   # 696 scenes, 377k QA pairs
│   └── nuscenes/samples/                     # 24,432 images, 6 cameras
├── data_processed/
│   ├── train.json              # 359,057 samples (95%)
│   ├── val.json                # 18,898 samples (5%)
│   └── train_mini.json         # 500 samples for quick test
├── checkpoints_qwen25/         # LoRA checkpoints (saved every 500 steps)
├── logs/                       # Training logs
└── docs/
    └── exploration_directions.md  # 10 exploration directions with priorities
```

## Running Commands

```bash
# Training (YAML-based config, all hyperparams in config file)
python scripts/train_lora.py --config configs/gh200.yaml --mini      # quick test
python scripts/train_lora.py --config configs/gh200.yaml             # full training
python scripts/train_lora.py --config configs/gh200.yaml --bs 4      # override batch size

# Layer 2: Visual token compression experiments
python scripts/train_lora.py --config configs/baseline.yaml          # no compression baseline
python scripts/train_lora.py --config configs/avg_pool_c4.yaml       # avg_pool 4x
python scripts/train_lora.py --config configs/fastervlm_c4.yaml      # fastervlm 4x
bash configs/run_all.sh                                              # run all 5 experiments

# Background training with logs
nohup python -u scripts/train_lora.py --config configs/gh200.yaml 2>&1 | tee logs/train_full.log &

# Inference (also YAML-based, supports --lora for any checkpoint)
python scripts/demo_inference.py --config configs/gh200.yaml                              # final LoRA
python scripts/demo_inference.py --config configs/gh200.yaml --lora checkpoints_qwen25/checkpoint-500
python scripts/demo_inference.py --config configs/gh200.yaml --no-lora                    # base model

# Data pipeline
python scripts/convert_data.py       # regenerate data_processed/ from raw DriveLM data
```

## Architecture Notes

- Training script reads ALL hyperparams from YAML config (no hardcoded values)
- Configs support `base_config: gh200.yaml` inheritance — child overrides parent fields
- CLI args `--bs`, `--lr`, `--epochs`, `--compress-method`, `--compress-ratio` can override config values
- Visual token compression: 4 methods in `scripts/visual_compress.py`, selected via `compress_method` in config
- Model loading: `quantize: true` → BitsAndBytes 4-bit; `quantize: false` → bf16 full precision
- On GH200: bf16 is faster than quantized (dequant overhead > memory savings)
- Checkpoints save LoRA adapter only (~50MB each), not full model
- tqdm progress bar shows: batch_loss, avg_loss, lr, opt_step, GPU memory

## Key Design Decisions

- All scripts use `os.path.dirname(os.path.dirname(os.path.abspath(__file__)))` for portable paths
- DriveLM v1.1 (not v1.0) — the HF repo only has v1.1
- Only CAM_FRONT images used for single-image fine-tuning (6-cam multi-view is a future direction)
- Image resolution capped to control GPU memory; configured via min_pixels/max_pixels in YAML

## Project Roadmap

See `docs/exploration_directions.md` for full 10-direction plan. Core execution path:
1. DriveLM LoRA (current) → 2. VLM Continual Learning (PR paper continuation) → 3. Token Compression → 4. Knowledge Distillation → 5. Serving/Deployment
