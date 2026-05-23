#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
STAMP=$(date -Iseconds | tr ':' '-')
LOG="logs/aseries/full_a3v2_${STAMP}.log"
mkdir -p logs/aseries
echo "[A.3 v2] starting $(date -Iseconds), log=$LOG"
accelerate launch --config_file accelerate_configs/fsdp_8gpu.yaml \
  scripts/train_lora.py \
    --config configs/nuscenes_planning_1cam_resampler_idefics2.yaml \
  2>&1 | tee "$LOG"
echo "[A.3 v2] done $(date -Iseconds)"
