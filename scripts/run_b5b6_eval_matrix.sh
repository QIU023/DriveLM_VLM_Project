#!/usr/bin/env bash
# Run the 4-cell B.5/B.6 eval matrix (multimodal × camera-only) after both
# trainings have finished. Saves results to eval_results/track_b/ and prints
# a 2x2 summary table.
#
# Pre-requisite:
#   - checkpoints_qwen25/nusc_planning_b5_multimodal/final exists
#   - checkpoints_qwen25/nusc_planning_b6_multimodal_dropout/final exists
#   - GPUs free (NOT during training; this owns all 8 GPUs via torchrun)
#
# Cells:
#   B5 + multimodal     -> true B.5 L2 (full HD map + bbox + ego at eval)
#   B5 + camera-only    -> B.5 robustness baseline (modality missing)
#   B6 + multimodal     -> true B.6 L2 (no-dropout eval)
#   B6 + camera-only    -> B.6 dropout robustness (modality missing)
#
# Usage: bash scripts/run_b5b6_eval_matrix.sh

set -uo pipefail
cd "$(dirname "$0")/.."

CKPT_B5="checkpoints_qwen25/nusc_planning_b5_multimodal/final"
CKPT_B6="checkpoints_qwen25/nusc_planning_b6_multimodal_dropout/final"

STAMP=$(date -Iseconds | tr ':' '-')
LOG="logs/b5b6_eval_matrix_${STAMP}.log"
SUMMARY="logs/b5b6_eval_matrix_summary.md"
mkdir -p logs eval_results/track_b
exec > >(tee -a "$LOG") 2>&1

log() { echo "[$(date -Iseconds)] $*"; }

# Pre-flight
for ckpt in "$CKPT_B5" "$CKPT_B6"; do
  if [[ ! -d "$ckpt" ]]; then
    log "ABORT: missing ckpt $ckpt"
    exit 2
  fi
done

# Disk floor
free_g=$(df --output=avail -BG /workspace | tail -1 | tr -dc '0-9')
log "disk free: ${free_g}G"
if (( free_g < 25 )); then
  log "ABORT: free <25G — eval should be cheap on disk but better safe"
  exit 9
fi

run_eval () {
  local ckpt="$1" mm_flag="$2" tag="$3" display="$4"
  local out="eval_results/track_b/${tag}.json"
  local elog="logs/dp_eval_${tag}.log"
  log "==================================================================="
  log "EVAL $display"
  log "==================================================================="
  log "  ckpt=$ckpt  mm_flag='$mm_flag'  out=$out"
  BATCH_SIZE=16 bash scripts/launch_planning_eval_dp.sh \
    "$ckpt" \
    --infos-val data/uniad_infos/nuscenes_infos_temporal_val.pkl \
    --nusc-root data/nuscenes \
    --output "$out" \
    $mm_flag \
    > "$elog" 2>&1
  if [[ ! -f "$out" ]]; then
    log "  [$display] ERROR: result JSON missing"
    return 1
  fi
  local l2 coll
  l2=$(/usr/bin/python3 -c "import json; d=json.load(open('$out')); print(f\"{d['L2_avg']:.4f}\")" 2>/dev/null)
  coll=$(/usr/bin/python3 -c "import json; d=json.load(open('$out')); print(f\"{d['collision_avg']*100:.2f}\")" 2>/dev/null)
  log "  [$display] L2_avg=$l2  collision_avg=$coll%"
  echo "$tag,$display,$l2,$coll" >> /tmp/b5b6_matrix_results.csv
}

# Reset accumulator
: > /tmp/b5b6_matrix_results.csv
echo "tag,display,L2_avg,collision_avg_%" >> /tmp/b5b6_matrix_results.csv

run_eval "$CKPT_B5" "--multimodal"   "B5_multimodal_eval"   "B.5 ckpt + multimodal eval (TRUE L2)"
run_eval "$CKPT_B5" ""               "B5_cameraonly_eval"   "B.5 ckpt + camera-only eval (robustness baseline)"
run_eval "$CKPT_B6" "--multimodal"   "B6_multimodal_eval"   "B.6 ckpt + multimodal eval (TRUE L2)"
run_eval "$CKPT_B6" ""               "B6_cameraonly_eval"   "B.6 ckpt + camera-only eval (dropout robustness)"

# Summary
echo
log "==================================================================="
log "FINAL 2x2 MATRIX"
log "==================================================================="
cat /tmp/b5b6_matrix_results.csv | column -t -s,

{
  echo "# B.5 / B.6 eval matrix $(date -Iseconds)"
  echo
  echo "| Tag | L2_avg | collision% |"
  echo "|---|---|---|"
  tail -n +2 /tmp/b5b6_matrix_results.csv | while IFS=, read -r tag display l2 coll; do
    echo "| $display | $l2 | $coll |"
  done
  echo
  echo "Reference: R1' / A.0 baseline (camera-only training+eval) = L2 0.6423, collision 3.73%"
} > "$SUMMARY"

log "Summary written to $SUMMARY"
