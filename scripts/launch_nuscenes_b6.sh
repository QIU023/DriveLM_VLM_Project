#!/usr/bin/env bash
# Launch Track B.6 — B.5 + per-modality random dropout p=0.1 (XPeng deployment
# robustness pattern: camera and ego state are NEVER dropped; HD-map and bbox
# are independently zeroed with p=0.1 per sample).
#
# Identical pre-flight to launch_nuscenes_b5.sh; only the config differs.
set -euo pipefail
cd "$(dirname "$0")/.."

STAMP=$(date -Iseconds | tr ':' '-')
LOG="logs/b6_multimodal_dropout_${STAMP}.log"
mkdir -p logs

free_g=$(df --output=avail -BG /workspace | tail -1 | tr -dc '0-9')
if (( free_g < 25 )); then
  echo "DISK PANIC: free ${free_g}G < 25G; aborting launch." >&2
  exit 9
fi

hdmap_n=$(ls data/preproc/hdmap_bev/train 2>/dev/null | wc -l)
bbox_n=$(wc -l < data/preproc/bbox_egostate_train.jsonl 2>/dev/null || echo 0)
echo "[launch_b6] HD-map train PNGs: $hdmap_n  bbox train rows: $bbox_n  free: ${free_g}G"
if (( hdmap_n < 27000 )); then
  echo "ERROR: HD-map train cache too small ($hdmap_n < 27000); rerun prep_hdmap_bev.py." >&2
  exit 8
fi
if (( bbox_n < 28000 )); then
  echo "ERROR: bbox train jsonl too small ($bbox_n < 28000); rerun prep_bbox_egostate.py." >&2
  exit 8
fi

echo "[launch_b6] starting at $(date -Iseconds), log=$LOG"
nohup accelerate launch --config_file accelerate_configs/fsdp_8gpu.yaml \
  scripts/train_lora.py --config configs/nuscenes_planning_b6.yaml \
  > "$LOG" 2>&1 &
echo "[launch_b6] pid=$!  log=$LOG"
