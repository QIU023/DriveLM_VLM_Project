#!/usr/bin/env bash
# Parallel version of stream_extract_nuscenes_3cam.sh — spawns N concurrent
# curl|tar pipes (one per S3 connection) to roughly multiply network
# throughput, since each single S3 stream caps around 4 MB/s on this network.
# Each part's tar extracts to data/nuscenes/samples/<cam>/ — same target as the
# serial version, and tar handles concurrent writers (different filenames,
# no overlap) fine.
#
# Usage:
#   bash scripts/stream_extract_nuscenes_3cam_par.sh PART1 PART2 PART3 ...
# Each PART is a one-or-two-digit index 1..10.
set -euo pipefail
cd "$(dirname "$0")/.."

DEST=data/nuscenes
LOG=logs/stream_extract_3cam_par.log
mkdir -p logs

free_g() { df --output=avail -BG /workspace | tail -1 | tr -dc '0-9' ; }

extract_one() {
  local i="$1"
  local ii=$(printf "%02d" "$i")
  local url="https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval${ii}_blobs.tgz"
  echo "[part $ii] start $(date -Iseconds) pid=$$ free=$(free_g)G" | tee -a "$LOG"
  curl -sS --fail --retry 3 --retry-delay 5 "$url" \
    | tar -x -z -C "$DEST" --wildcards \
        'samples/CAM_FRONT_LEFT/*' \
        'samples/CAM_FRONT_RIGHT/*' \
        2>>"$LOG" \
    && echo "[part $ii] done $(date -Iseconds) free=$(free_g)G" | tee -a "$LOG" \
    || echo "[part $ii] FAIL $(date -Iseconds)" | tee -a "$LOG"
}

for p in "$@"; do
  if (( $(free_g) < 15 )); then
    echo "DISK PANIC: free=$(free_g)G — skipping rest" | tee -a "$LOG"
    break
  fi
  extract_one "$p" &
done
wait
echo "ALL PARALLEL DONE $(date -Iseconds) free=$(free_g)G" | tee -a "$LOG"
