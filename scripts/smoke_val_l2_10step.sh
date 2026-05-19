#!/usr/bin/env bash
# 10-step smoke for the teacher-forced L2 (metres) validation signal added to
# validate() in train_lora.py. Uses the standard 1-cam planning config and
# overrides val_every to 5 so validation fires at step 5 and step 10. Verifies
# that the new "[VAL] step=N val_loss=X val_acc=Y L2 avg=Z (1s=A 2s=B 3s=C)"
# log line is emitted.
#
# Resource policy: this smoke uses 8 GPUs via fsdp_8gpu.yaml to mirror real
# training. Only run when no other GPU work is active.
#
# Usage:
#   bash scripts/smoke_val_l2_10step.sh
set -euo pipefail
cd "$(dirname "$0")/.."

STAMP=$(date -Iseconds | tr ':' '-')
LOG="logs/smoke_val_l2_${STAMP}.log"
mkdir -p logs

echo "[smoke-l2] starting at $(date -Iseconds), log=$LOG"

accelerate launch --config_file accelerate_configs/fsdp_8gpu.yaml \
  scripts/train_lora.py --config configs/nuscenes_planning_full.yaml \
  --max-steps 10 --val-every 5 --val-batches 2 \
  --no-final-save --save-every 999999 \
  2>&1 | tee "$LOG"

echo "[smoke-l2] checking for L2 log line..."
if grep -qE '\[VAL\] step=[0-9]+ val_loss=.* val_acc=.* L2 (avg=|skipped)' "$LOG"; then
  echo "[smoke-l2] PASS: L2 log line found"
  grep -E '\[VAL\]' "$LOG"
else
  echo "[smoke-l2] FAIL: no L2 log line found"
  exit 1
fi
