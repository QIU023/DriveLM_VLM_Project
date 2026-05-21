#!/usr/bin/env bash
# Orchestrator: full LiDAR BEV cache, runs AFTER A.3 v2 training has freed
# the GPUs and (ideally) a human has confirmed disk pressure is OK.
#
# Step order:
#   1. Guard: refuse to run if any GPU has >100 MiB used (A.3 v2 still alive).
#   2. Guard: refuse to run if /workspace has < 50 GB free
#      (LiDAR keyframes pull = ~25-30 GB).
#   3. Pull samples/LIDAR_TOP/*.pcd.bin keyframes for all 10 trainval parts,
#      stream-extract via curl|tar like stream_extract_nuscenes_3cam.sh.
#   4. Run prep_lidar_bev.py twice: train + val. Skips already-cached samples
#      via --skip-existing so the script is idempotent on retries.
#
# NOTE: the encoder is CPU-only (occupancy mode), so this script never touches
# the GPU even after A.3 v2 frees them. The GPU check exists only because the
# task brief asked for it as a safety gate against partial overlap.
#
# Usage:
#   bash scripts/run_lidar_bev_after_a3v2.sh [--force-disk] [--skip-pull]
#       --force-disk   bypass the 50 GB free guard (you Know What You're Doing)
#       --skip-pull    skip the curl|tar step (LiDAR already on disk)
#
set -euo pipefail
cd "$(dirname "$0")/.."

FORCE_DISK=0
SKIP_PULL=0
for arg in "$@"; do
  case "$arg" in
    --force-disk) FORCE_DISK=1 ;;
    --skip-pull)  SKIP_PULL=1 ;;
    *) echo "unknown arg: $arg"; exit 64 ;;
  esac
done

LOG=logs/lidar_bev_orchestrator.log
mkdir -p logs data_processed/lidar_bev_occ_v1

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

# --- guard 1: GPU free? ---
if command -v nvidia-smi >/dev/null 2>&1; then
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
         | awk '{ if ($1 > 100) print $0 }' | wc -l)
  if [[ "$busy" -gt 0 ]]; then
    log "ABORT: $busy GPU(s) still have >100 MiB used. A.3 v2 likely still running."
    log "       Wait for training to finish, or pass nothing if you only need CPU run."
    exit 1
  fi
  log "GPU guard PASS: all GPUs idle"
fi

# --- guard 2: disk free? ---
free_g=$(df --output=avail -BG /workspace | tail -1 | tr -dc '0-9')
log "disk free: ${free_g} GB"
if [[ "$FORCE_DISK" -ne 1 && "$free_g" -lt 50 ]]; then
  log "ABORT: < 50 GB free. LiDAR keyframe pull alone is ~25-30 GB."
  log "       Free up disk or pass --force-disk."
  exit 2
fi

# --- step 3: pull LiDAR keyframes ---
if [[ "$SKIP_PULL" -eq 0 ]]; then
  log "STAGE 3: stream-extract samples/LIDAR_TOP/ from all 10 trainval parts"
  for i in $(seq 1 10); do
    ii=$(printf "%02d" "$i")
    url="https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval${ii}_blobs.tgz"
    free_g=$(df --output=avail -BG /workspace | tail -1 | tr -dc '0-9')
    log "part $i: free=${free_g}G  url=${url}"
    if [[ "$free_g" -lt 15 ]]; then
      log "part $i: ABORT — free <15G (disk-panic threshold)"
      exit 9
    fi
    curl -sS --fail --retry 3 --retry-delay 5 "$url" \
      | tar -x -z -C data/nuscenes --wildcards \
          'samples/LIDAR_TOP/*' \
          2>>"$LOG" || {
        log "part $i: tar|curl failed; continuing"
        continue
      }
    log "part $i: done"
  done
  log "STAGE 3 DONE"
fi

# --- step 4a: train cache ---
log "STAGE 4a: encode train BEV cache"
/usr/bin/python3 scripts/prep_lidar_bev.py \
    --infos-pkl data/nuscenes/_hf_meta/nuscenes_mmdet3d-12Hz/nuscenes_interp_12Hz_infos_train.pkl \
    --lidar-root data/nuscenes \
    --output-dir data_processed/lidar_bev_occ_v1/train \
    --encoder occupancy \
    --device cpu \
    --workers 8 \
    --skip-existing \
    --log-every 2000 2>&1 | tee -a "$LOG"

# --- step 4b: val cache ---
log "STAGE 4b: encode val BEV cache"
/usr/bin/python3 scripts/prep_lidar_bev.py \
    --infos-pkl data/nuscenes/_hf_meta/nuscenes_mmdet3d-12Hz/nuscenes_interp_12Hz_infos_val.pkl \
    --lidar-root data/nuscenes \
    --output-dir data_processed/lidar_bev_occ_v1/val \
    --encoder occupancy \
    --device cpu \
    --workers 8 \
    --skip-existing \
    --log-every 1000 2>&1 | tee -a "$LOG"

log "ALL DONE"
echo
echo "Final disk usage:"
du -sh data_processed/lidar_bev_occ_v1/{train,val}
df -h /workspace | tail -2
