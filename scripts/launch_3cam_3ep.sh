#!/usr/bin/env bash
# Launch the 3-epoch R1'' (3-cam) training run on all 8 GPUs via FSDP.
# Main thread runs this AFTER R1' finishes (~20:20Z 2026-05-19) and after
# the FL/FR stream-extract download has finished extracting enough samples
# to make a meaningful training set.
#
# Disk floor: aborts if /workspace free < 25 GB at launch time (3-cam ckpts
# are same size as 1-cam since vision tower is frozen, but the smoke + 2
# ckpts + working space gives a 50G floor for safety).
set -euo pipefail
cd "$(dirname "$0")/.."

STAMP=$(date -Iseconds | tr ':' '-')
LOG="logs/r1pp_3cam_${STAMP}.log"
mkdir -p logs

free_g=$(df --output=avail -BG /workspace | tail -1 | tr -dc '0-9')
if (( free_g < 25 )); then
  echo "DISK PANIC: free ${free_g}G < 25G; aborting launch." >&2
  exit 9
fi

# Sanity: confirm the FL/FR download has produced more than the DriveLM-only
# 4072 baseline (otherwise the 3-cam filter will keep <2.1k samples and the
# run is not worth doing).
fl_count=$(ls data/nuscenes/samples/CAM_FRONT_LEFT 2>/dev/null | wc -l)
fr_count=$(ls data/nuscenes/samples/CAM_FRONT_RIGHT 2>/dev/null | wc -l)
echo "[launch] FL files: $fl_count  FR files: $fr_count  free: ${free_g}G"
if (( fl_count < 10000 || fr_count < 10000 )); then
  echo "WARN: FL/FR counts low ($fl_count / $fr_count) — train set will be small." >&2
fi

echo "[launch] starting at $(date -Iseconds), log=$LOG"
nohup accelerate launch --config_file accelerate_configs/fsdp_8gpu.yaml \
  scripts/train_lora.py --config configs/nuscenes_planning_3cam_full.yaml \
  > "$LOG" 2>&1 &
echo "[launch] pid=$!  log=$LOG"
