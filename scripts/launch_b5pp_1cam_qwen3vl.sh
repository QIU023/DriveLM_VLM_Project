#!/usr/bin/env bash
# B.5'' — Qwen3-VL-4B × 1-cam × multimodal full SFT (3 epochs)
# ETA: 2625 steps × ~6s = ~4.4h on 8x RTX 5090 FSDP.
# Output: checkpoints_qwen25/nusc_planning_b5pp_1cam_qwen3vl_multimodal/
set -uo pipefail
cd "$(dirname "$0")/.."

STAMP=$(date -Iseconds | tr ':' '-')
LOG="logs/qwen3vl/full_sft_${STAMP}.log"
mkdir -p logs/qwen3vl

echo "[launch] starting at $(date -Iseconds)"
echo "[launch] log=$LOG"

accelerate launch --config_file accelerate_configs/fsdp_8gpu.yaml \
  scripts/train_lora.py \
    --config configs/nuscenes_planning_1cam_qwen3vl_multimodal.yaml \
  2>&1 | tee "$LOG"

echo "[launch] done at $(date -Iseconds)"
