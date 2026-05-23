#!/usr/bin/env bash
# A.1 v2: BLIP-2 pretrained Q-Former, 1cam camera-only, 3 epochs.
set -uo pipefail
cd "$(dirname "$0")/.."
STAMP=$(date -Iseconds | tr ':' '-')
LOG="logs/aseries/full_a1v2_${STAMP}.log"
mkdir -p logs/aseries
echo "[A.1 v2] starting $(date -Iseconds), log=$LOG"
accelerate launch --config_file accelerate_configs/fsdp_8gpu.yaml \
  scripts/train_lora.py \
    --config configs/nuscenes_planning_1cam_qformer_blip2.yaml \
  2>&1 | tee "$LOG"
echo "[A.1 v2] done $(date -Iseconds)"
