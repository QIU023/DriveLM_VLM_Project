#!/usr/bin/env bash
# Launch R1''' = R1'' v2 (3-cam camera-only @ max_length=12288, truncation fix).
#
# QUEUED to run AFTER B.5' (nusc_planning_b5prime_3cam_multimodal) finishes —
# do NOT launch concurrently (both want all 8 GPUs).
#
# Disk floor: aborts if /workspace free < 25 GB.
set -euo pipefail
cd "$(dirname "$0")/.."

# Block-on: if B.5' is still running, refuse (would OOM by GPU contention).
B5PRIME_PID_FILE="logs/b5prime_3cam.pid"
if [[ -f "$B5PRIME_PID_FILE" ]]; then
  pid=$(cat "$B5PRIME_PID_FILE")
  if kill -0 "$pid" 2>/dev/null; then
    echo "ABORT: B.5' (pid=$pid) is still running. Wait for it to finish before launching R1'''." >&2
    exit 11
  fi
fi

STAMP=$(date -Iseconds | tr ':' '-')
LOG="logs/r1ppp_3cam_v2_${STAMP}.log"
mkdir -p logs

free_g=$(df --output=avail -BG /workspace | tail -1 | tr -dc '0-9')
if (( free_g < 25 )); then
  echo "DISK PANIC: free ${free_g}G < 25G; aborting launch." >&2
  exit 9
fi

# 3-cam sanity (same as R1'' v1)
fl_count=$(ls data/nuscenes/samples/CAM_FRONT_LEFT 2>/dev/null | wc -l)
fr_count=$(ls data/nuscenes/samples/CAM_FRONT_RIGHT 2>/dev/null | wc -l)
if (( fl_count < 10000 || fr_count < 10000 )); then
  echo "ERROR: FL/FR file counts low (FL=$fl_count FR=$fr_count)." >&2
  exit 7
fi

echo "[launch_r1ppp] FL=$fl_count FR=$fr_count free=${free_g}G"
echo "[launch_r1ppp] starting at $(date -Iseconds), log=$LOG"

nohup accelerate launch --config_file accelerate_configs/fsdp_8gpu.yaml \
  scripts/train_lora.py --config configs/nuscenes_planning_3cam_full_v2.yaml \
  > "$LOG" 2>&1 &
PID=$!
echo "[launch_r1ppp] pid=$PID  log=$LOG"
echo $PID > logs/r1ppp_3cam_v2.pid
