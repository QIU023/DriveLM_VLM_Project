#!/usr/bin/env bash
# Track A.1: 1-cam × 4f + Q-Former projector (BLIP-2 style, 64q × 6L).
# Anchor: R1' = A.0 (1-cam × 4f + Linear projector, L2 0.6423, collision 3.73%).
# Hypothesis: Q-Former cross-attention pooling beats Linear MLP at fusing 4f
# video into LM token stream when token budget is tight.
#
# AC off, LBS=4 × grad_accum=1 (GBS=32 via FSDP=8), warmup=36/lr_step=143
# (ratio-scaled to ~2050 total steps), save_every=val_every=50.
set -euo pipefail
cd "$(dirname "$0")/.."

STAMP=$(date -Iseconds | tr ':' '-')
LOG="logs/a1_1cam_qformer_${STAMP}.log"
mkdir -p logs

free_g=$(df --output=avail -BG /workspace | tail -1 | tr -dc '0-9')
if (( free_g < 25 )); then
  echo "DISK PANIC: free ${free_g}G < 25G; aborting." >&2
  exit 9
fi

echo "[A.1] starting at $(date -Iseconds), log=$LOG, free=${free_g}G"
nohup accelerate launch --config_file accelerate_configs/fsdp_8gpu.yaml \
  scripts/train_lora.py --config configs/nuscenes_planning_1cam_qformer.yaml \
  > "$LOG" 2>&1 &
echo "[A.1] pid=$!  log=$LOG"
