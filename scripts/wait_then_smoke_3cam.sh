#!/usr/bin/env bash
# Wait for R1' (PID 3600376, but generic-match the train_lora.py invocation)
# to finish, then run the 3-cam smoke. The main thread can use this when it
# is ready to validate the 3-cam pipeline; afterwards it manually launches
# scripts/launch_3cam_3ep.sh.
#
# Usage:
#   bash scripts/wait_then_smoke_3cam.sh [PID]
# If PID is not provided, waits for any running train_lora.py to vanish.
set -euo pipefail
cd "$(dirname "$0")/.."

PID=${1:-}
echo "[wait_then_smoke] start $(date -Iseconds)  arg PID=${PID:-<auto>}"

if [[ -n "$PID" ]]; then
  while kill -0 "$PID" 2>/dev/null; do
    sleep 60
    echo "[wait_then_smoke] $(date -Iseconds) still waiting on PID=$PID"
  done
else
  while pgrep -f "scripts/train_lora.py" >/dev/null; do
    sleep 60
    echo "[wait_then_smoke] $(date -Iseconds) train_lora.py still running"
  done
fi

echo "[wait_then_smoke] train_lora.py exited at $(date -Iseconds); launching smoke"
# Confirm GPUs are free
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F, '$2 > 1000 { print "GPU "$1" still uses "$2" MiB"; exit_code=1 } END { exit exit_code? 1 : 0 }' || {
  echo "[wait_then_smoke] WARN: some GPU memory still allocated; sleeping 30s and continuing"
  sleep 30
}

CUDA_VISIBLE_DEVICES=0 bash scripts/smoke_3cam_5step.sh
