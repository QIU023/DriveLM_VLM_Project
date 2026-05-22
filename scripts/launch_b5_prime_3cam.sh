#!/usr/bin/env bash
# Launch Track B.5' — 顶配: 3-cam × multi-modal SFT
#
#   = R1'' 3-cam camera-only + B.5 multi-modal (HD-map BEV + bbox + ego)
#   = production-aligned condition for the XPeng JD bullet
#
# Disk floor: aborts if /workspace free < 25 GB at launch time.
#
# Pre-launch checklist (per MEMORY.md feedback_pre_launch_checklist.md):
#   - Audit table:        configs/nuscenes_planning_b5_prime_3cam.yaml header
#                         (paper-cited deltas: planning_cams from R1'',
#                          multimodal from B.5; nothing else changed)
#   - Save->load smoke:   N/A (no new nn.Module — same MultiModalPlanningDataset
#                         as B.5 which already trained + saved + loaded fine)
#   - Gradient sync:      N/A (same FSDP=8 / accelerate config as B.5)
#   - Real-data smoke:    scripts/_smoke_multimodal_planning_dataset.py
#   - Boot warning audit: grep WARN in first 200 lines of training log
#   - Disk:               this script's pre-flight 25G floor
#   - ETA:                ~6-8 h on 8×RTX-5090
#                         (3-cam step ~7s + HD map I/O ~10% overhead = ~8s/step
#                          × ~2625 steps × 3 epochs / FSDP=8 ≈ 6.5h)
#                         + ~30 min for boot + eval/ckpt cadence = ~7-8h
set -euo pipefail
cd "$(dirname "$0")/.."

STAMP=$(date -Iseconds | tr ':' '-')
LOG="logs/b5prime_3cam_${STAMP}.log"
mkdir -p logs

free_g=$(df --output=avail -BG /workspace | tail -1 | tr -dc '0-9')
if (( free_g < 25 )); then
  echo "DISK PANIC: free ${free_g}G < 25G; aborting launch." >&2
  exit 9
fi

# Sanity: HD-map cache (same as B.5)
hdmap_n=$(ls data/preproc/hdmap_bev/train 2>/dev/null | wc -l)
bbox_n=$(wc -l < data/preproc/bbox_egostate_train.jsonl 2>/dev/null || echo 0)
if (( hdmap_n < 27000 )); then
  echo "ERROR: HD-map train cache too small ($hdmap_n < 27000)." >&2
  exit 8
fi
if (( bbox_n < 28000 )); then
  echo "ERROR: bbox train jsonl too small ($bbox_n < 28000)." >&2
  exit 8
fi

# Sanity: 3-cam files (same as R1'')
fl_count=$(ls data/nuscenes/samples/CAM_FRONT_LEFT 2>/dev/null | wc -l)
fr_count=$(ls data/nuscenes/samples/CAM_FRONT_RIGHT 2>/dev/null | wc -l)
if (( fl_count < 10000 || fr_count < 10000 )); then
  echo "ERROR: FL/FR file counts low (FL=$fl_count FR=$fr_count); 3-cam set will be too small." >&2
  exit 7
fi

echo "[launch_b5'] HD-map=$hdmap_n  bbox=$bbox_n  FL=$fl_count  FR=$fr_count  free=${free_g}G"
echo "[launch_b5'] starting at $(date -Iseconds), log=$LOG"

nohup accelerate launch --config_file accelerate_configs/fsdp_8gpu.yaml \
  scripts/train_lora.py --config configs/nuscenes_planning_b5_prime_3cam.yaml \
  > "$LOG" 2>&1 &
PID=$!
echo "[launch_b5'] pid=$PID  log=$LOG"
echo $PID > logs/b5prime_3cam.pid
