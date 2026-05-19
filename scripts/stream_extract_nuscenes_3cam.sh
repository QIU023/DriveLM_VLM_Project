#!/usr/bin/env bash
# Stream-extract CAM_FRONT_LEFT and CAM_FRONT_RIGHT from nuScenes trainval
# blob tarballs. The tar files are large (~30-42 GB each, 10 parts), but each
# archive is organized as samples/<cam>/<file>.jpg + sweeps/... — so we can
# filter on the fly with `tar --wildcards` and never have to land the .tgz to
# disk. Total disk needed: ~5-7 GB for FL+FR combined (the keyframes for the
# entire trainval set), no temp tarball storage required.
#
# Usage:
#   bash scripts/stream_extract_nuscenes_3cam.sh [START_PART] [END_PART]
# Defaults: parts 1..10. Pass e.g. `1 1` to do just part 1.
set -euo pipefail
cd "$(dirname "$0")/.."

START=${1:-1}
END=${2:-10}
DEST=data/nuscenes
LOG=logs/stream_extract_3cam.log

mkdir -p "$DEST" logs

free_g() {
  df --output=avail -BG /workspace | tail -1 | tr -dc '0-9'
}

for i in $(seq "$START" "$END"); do
  # nuScenes parts are numbered v1.0-trainval01..10_blobs.tgz (two-digit, zero-padded)
  ii=$(printf "%02d" "$i")
  url="https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval${ii}_blobs.tgz"
  echo "[part $i] $(date -Iseconds) free=$(free_g)G  pulling $url" | tee -a "$LOG"
  if (( $(free_g) < 15 )); then
    echo "[part $i] ABORT: free <15G" | tee -a "$LOG"
    exit 9
  fi
  # `tar -z` decompresses gzip; `--wildcards` enables glob match; we keep only
  # FL/FR keyframes under samples/ (not sweeps/, which are 10x bigger).
  curl -sS --fail --retry 3 --retry-delay 5 "$url" \
    | tar -x -z -C "$DEST" --wildcards \
        'samples/CAM_FRONT_LEFT/*' \
        'samples/CAM_FRONT_RIGHT/*' \
        2>>"$LOG" || {
      echo "[part $i] FAILED (curl|tar exit)" | tee -a "$LOG"
      continue
    }
  echo "[part $i] $(date -Iseconds) done; free=$(free_g)G" | tee -a "$LOG"
done
echo "ALL DONE $(date -Iseconds); free=$(free_g)G" | tee -a "$LOG"
