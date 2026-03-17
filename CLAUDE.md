# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VLM fine-tuning project for autonomous driving using QLoRA on DriveLM (nuScenes-based QA dataset). Supports two model architectures: Qwen2.5-VL-3B-Instruct and Qwen3.5-4B.

## Running Scripts

All scripts are standalone Python files executed directly:

```bash
# Data pipeline
python scripts/download_drivelm_data.py      # Fetch DriveLM QA JSON + images from HuggingFace
python scripts/download_images.py            # Download nuScenes subset images only
python scripts/inspect_data.py               # Explore DriveLM JSON structure
python scripts/convert_data.py               # Convert DriveLM → Qwen conversation format

# Model download
python scripts/download_model.py             # Download Qwen models from HuggingFace Hub

# Training
python scripts/train_lora.py --mini          # Quick test run (500 samples)
python scripts/train_lora.py                 # Full training (Qwen2.5-VL-3B)
python scripts/train_lora_qwen35.py          # Full training (Qwen3.5-4B)

# Inference / evaluation
python scripts/test_inference.py             # Verify inference pipeline works
python scripts/demo_inference.py --n 10      # Evaluate 10 diverse samples with LoRA
python scripts/demo_inference.py --no-lora   # Compare base model (no adapter)
python scripts/demo_inference_qwen35.py      # Evaluate Qwen3.5-4B variant
```

No build system, linter, or test framework is configured. No requirements.txt exists — dependencies are implicit (torch, transformers, peft, bitsandbytes, pillow, huggingface_hub, wandb optional).

## Architecture

**Pipeline stages:** Data download → Data conversion → Model download → QLoRA fine-tuning → Inference/evaluation

**Data flow:**
- Raw DriveLM QA JSON (HuggingFace) → `convert_data.py` → `data_processed/{train,val,train_mini}.json` in Qwen conversation format
- Each sample: system prompt (category context) + user message (image + question) + assistant response (ground truth)
- Categories: perception, prediction, planning, behavior
- Split: 95% train / 5% val; train_mini.json has 500 samples for quick iteration

**Training approach:**
- 4-bit NF4 quantization with double quantization (bitsandbytes)
- LoRA (r=16, alpha=32) targeting attention + MLP projections
- Gradient accumulation (8 steps) for effective batch size of 8 on constrained GPU memory
- Custom `DriveLMDataset` class handles image loading, tokenization, and label masking (masks everything before assistant turn)

**Key difference between model variants:**
- Qwen2.5-VL-3B-Instruct: separate ViT encoder + Transformer decoder, max_length=512
- Qwen3.5-4B: natively multimodal (integrated vision encoder, Gated DeltaNet), max_length=1024

## Important Notes

- All scripts use relative paths resolved from `__file__` — portable across environments
- Image resolution is capped (256×28×28 to 512×28×28 pixels) to fit in GPU memory
- Checkpoints save to `checkpoints_qwen25/` or `checkpoints/` depending on variant
- Evaluation uses exact-match accuracy (case-insensitive) with per-category breakdown