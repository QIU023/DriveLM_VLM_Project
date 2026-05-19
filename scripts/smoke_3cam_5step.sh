#!/usr/bin/env bash
# 5-step smoke for the 3-cam planning config. Runs on a single GPU (defaults
# to GPU 0; override with CUDA_VISIBLE_DEVICES) so it can run alongside R1' if
# absolutely necessary — but the recommended workflow is to wait for R1' to
# finish so we have all 8 GPUs free for the full run.
#
# Verifies: dataset loads without OOM/shape errors, processor builds 3 video
# blocks per sample, loss is finite at step 5.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/smoke_3cam_5step.sh
set -euo pipefail
cd "$(dirname "$0")/.."

STAMP=$(date -Iseconds | tr ':' '-')
LOG="logs/smoke_3cam_${STAMP}.log"
mkdir -p logs

echo "[smoke] starting at $(date -Iseconds), log=$LOG"
echo "[smoke] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

accelerate launch --config_file accelerate_configs/single_gpu_smoke.yaml \
  scripts/train_lora.py --config configs/nuscenes_planning_3cam_full.yaml \
  --max-steps 5 --no-validate --no-final-save --save-every 999999 \
  2>&1 | tee "$LOG"
