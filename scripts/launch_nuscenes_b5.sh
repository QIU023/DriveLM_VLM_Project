#!/usr/bin/env bash
# Launch Track B.5 — image-as-modality multi-modal SFT (HD-map BEV + bbox text
# + ego state on top of R1' / A.0 baseline). 1-cam × 4f camera video; no new
# learnable module (OEM Qwen2.5-VL processor handles both image+video).
#
# Disk floor: aborts if /workspace free < 25 GB at launch time (B.5 ckpts are
# the same size as R1' since no new params; 18 GB × keep_latest_k=2 = 36 GB
# steady, ~50 GB peak during save+prune).
#
# Pre-launch checklist (per MEMORY.md feedback_pre_launch_checklist.md):
#   - Audit table:        configs/nuscenes_planning_b5.yaml header (paper-cited)
#   - Save->load smoke:   N/A (no new nn.Module)
#   - Gradient sync:      N/A (no new nn.Module)
#   - Real-data smoke:    scripts/_smoke_multimodal_planning_dataset.py (CPU; bbox + HD map round-trip)
#   - Boot warning audit: smoke log under /tmp/smoke_b5_*.log (grep WARN)
#   - Disk:               this script's pre-flight floor
#   - ETA:                ~9 h on 8×RTX-5090 (linear scale of R1' 8h + ~10 % image-side overhead)
set -euo pipefail
cd "$(dirname "$0")/.."

STAMP=$(date -Iseconds | tr ':' '-')
LOG="logs/b5_multimodal_${STAMP}.log"
mkdir -p logs

free_g=$(df --output=avail -BG /workspace | tail -1 | tr -dc '0-9')
if (( free_g < 25 )); then
  echo "DISK PANIC: free ${free_g}G < 25G; aborting launch." >&2
  exit 9
fi

# Sanity: confirm the HD-map and bbox caches exist.
hdmap_n=$(ls data/preproc/hdmap_bev/train 2>/dev/null | wc -l)
bbox_n=$(wc -l < data/preproc/bbox_egostate_train.jsonl 2>/dev/null || echo 0)
echo "[launch_b5] HD-map train PNGs: $hdmap_n  bbox train rows: $bbox_n  free: ${free_g}G"
if (( hdmap_n < 27000 )); then
  echo "ERROR: HD-map train cache too small ($hdmap_n < 27000); rerun prep_hdmap_bev.py." >&2
  exit 8
fi
if (( bbox_n < 28000 )); then
  echo "ERROR: bbox train jsonl too small ($bbox_n < 28000); rerun prep_bbox_egostate.py." >&2
  exit 8
fi

echo "[launch_b5] starting at $(date -Iseconds), log=$LOG"
nohup accelerate launch --config_file accelerate_configs/fsdp_8gpu.yaml \
  scripts/train_lora.py --config configs/nuscenes_planning_b5.yaml \
  > "$LOG" 2>&1 &
echo "[launch_b5] pid=$!  log=$LOG"
