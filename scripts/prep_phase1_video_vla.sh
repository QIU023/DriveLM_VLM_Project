#!/usr/bin/env bash
# Phase 1 Video-VLA data prep (idempotent).
#
# Inputs (already on disk in this repo's environment):
#   /workspace/.hf_home/DriveLM/v1_1_train_nus.json
#   /workspace/.hf_home/DriveLM/v1_1_val_nus_q_only.json
#   /workspace/.hf_home/DriveLM/drivelm_nus_imgs_{train,val}.zip
#   data/nuscenes/v1.0-trainval/ego_pose.json (+ sample.json, sample_data.json)
#
# Run:
#   bash scripts/prep_phase1_video_vla.sh
#
# Steps:
#   1. Unzip the two DriveLM image zips into data/nuscenes/samples_drivelm/{train,val}.
#      The train zip extracts as nuscenes/samples/CAM_FRONT/...; the val zip
#      extracts as val_data/CAM_FRONT/... (different layouts; documented here).
#   2. Symlink data/nuscenes/samples -> samples_drivelm/train/nuscenes/samples
#      so the convert_data_video.py default IMAGE_ROOT keeps working.
#   3. Convert DriveLM QA -> 4-frame video JSON (~328K records).
#   4. Extract real ego trajectories from nuScenes meta with the frozen-tail
#      bugfix (~328K kept of 378K, drop ~50K where scene ends before horizon).
#
# Disk floor: stops if /workspace free drops below 10G between steps.
set -euo pipefail
cd "$(dirname "$0")/.."

DRIVELM=/workspace/.hf_home/DriveLM
NUSC=data/nuscenes

check_disk() {
  local avail_g
  avail_g=$(df --output=avail -BG /workspace | tail -1 | tr -dc '0-9')
  if (( avail_g < 10 )); then
    echo "DISK PANIC: /workspace free ${avail_g}G < 10G. Aborting." >&2
    exit 9
  fi
}

# --- 1. Unzip ---
if [[ ! -d "${NUSC}/samples_drivelm/train/nuscenes/samples/CAM_FRONT" ]]; then
  mkdir -p "${NUSC}/samples_drivelm"
  if [[ -f "${DRIVELM}/drivelm_nus_imgs_train.zip" ]]; then
    echo "[1a] unzip train (3.5 GB)..."
    unzip -q -o "${DRIVELM}/drivelm_nus_imgs_train.zip" -d "${NUSC}/samples_drivelm/train"
  else
    echo "[1a] WARN: train zip already deleted; skipping (expect samples already extracted)"
  fi
fi
if [[ ! -d "${NUSC}/samples_drivelm/val/val_data/CAM_FRONT" ]]; then
  if [[ -f "${DRIVELM}/drivelm_nus_imgs_val.zip" ]]; then
    echo "[1b] unzip val (705 MB)..."
    unzip -q -o "${DRIVELM}/drivelm_nus_imgs_val.zip" -d "${NUSC}/samples_drivelm/val"
  fi
fi
check_disk

# --- 2. Symlink ---
if [[ ! -L "${NUSC}/samples" ]]; then
  if [[ -d "${NUSC}/samples" ]]; then
    echo "[2] backing up legacy samples/ -> samples_legacy_smoke/"
    mv "${NUSC}/samples" "${NUSC}/samples_legacy_smoke"
  fi
  ln -s samples_drivelm/train/nuscenes/samples "${NUSC}/samples"
fi
ls "${NUSC}/samples/CAM_FRONT" | wc -l | xargs -I{} echo "[2] symlink OK ({} CAM_FRONT frames)"

# --- 3. Convert QA -> 4-frame video JSON ---
if [[ ! -f data_processed/v1_1_video_n4_FULL.json ]]; then
  echo "[3] convert_data_video.py --num-frames 4 ..."
  /venv/main/bin/python scripts/convert_data_video.py \
    --qa-json "${DRIVELM}/v1_1_train_nus.json" \
    --image-root "${NUSC}/samples" \
    --num-frames 4 \
    --output v1_1_video_n4_FULL.json
fi
check_disk

# --- 4. Extract ego trajectories (real nuScenes meta) ---
if [[ ! -f data_processed/v1_1_video_n4_FULL_with_traj.json ]]; then
  echo "[4] extract_ego_trajectory.py ..."
  /venv/main/bin/python scripts/extract_ego_trajectory.py \
    --input data_processed/v1_1_video_n4_FULL.json
fi

echo ""
echo "==> done."
ls -lh data_processed/v1_1_video_n4_FULL*.json
