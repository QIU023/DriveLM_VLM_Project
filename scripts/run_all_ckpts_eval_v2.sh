#!/usr/bin/env bash
# Re-eval ALL preserved ckpts with the v2 pipeline:
#   - FIXED multimodal eval (no longer rebuilds prompt and discards HD map / bbox)
#   - NEW sub-scenario + behavior metrics (added by agent, no breaking change)
#
# 6 cells:
#   R1' / A.0 1-cam baseline (camera-only training+eval)
#   R1''     3-cam baseline (camera-only training+eval, 3 cams)
#   B.5      multimodal eval (true L2 with HD map + bbox)
#   B.5      camera-only eval (modality-degradation baseline)
#   B.6      multimodal eval (true L2)
#   B.6      camera-only eval (dropout robustness)
#
# Saves to eval_results/track_v2/<tag>.json. All cells produce the same
# augmented schema: existing TemAvg/NoAvg/collision_rate + new
# scenario_counts/scenario_metrics/behavior.

set -uo pipefail
cd "$(dirname "$0")/.."

CKPT_R1="checkpoints_qwen25/nuscenes_planning_3b_full_sft/final"
CKPT_R1PRIME3="checkpoints_qwen25/nuscenes_planning_3cam_3b_full_sft/final"
CKPT_B5="checkpoints_qwen25/nusc_planning_b5_multimodal/final"
CKPT_B6="checkpoints_qwen25/nusc_planning_b6_multimodal_dropout/final"

STAMP=$(date -Iseconds | tr ':' '-')
LOG="logs/all_ckpts_eval_v2_${STAMP}.log"
SUMMARY="logs/all_ckpts_eval_v2_summary.md"
mkdir -p logs eval_results/track_v2
exec > >(tee -a "$LOG") 2>&1

log() { echo "[$(date -Iseconds)] $*"; }

# Pre-flight
for ckpt in "$CKPT_R1" "$CKPT_R1PRIME3" "$CKPT_B5" "$CKPT_B6"; do
  if [[ ! -d "$ckpt" ]]; then
    log "ABORT: missing ckpt $ckpt"
    exit 2
  fi
done

free_g=$(df --output=avail -BG /workspace | tail -1 | tr -dc '0-9')
log "disk free: ${free_g}G"
if (( free_g < 25 )); then
  log "ABORT: free <25G"
  exit 9
fi

: > /tmp/all_ckpts_v2_results.csv
echo "tag,display,L2_avg,collision_avg_%,turning_L2,turning_n,heading_err,brake_rate,progress" \
  >> /tmp/all_ckpts_v2_results.csv

run_eval () {
  local ckpt="$1" extra_args="$2" tag="$3" display="$4"
  local out="eval_results/track_v2/${tag}.json"
  local elog="logs/dp_eval_v2_${tag}.log"
  log "==================================================================="
  log "EVAL $display"
  log "==================================================================="
  log "  ckpt=$ckpt  extra='$extra_args'  out=$out"
  BATCH_SIZE=16 bash scripts/launch_planning_eval_dp.sh \
    "$ckpt" \
    --infos-val data/uniad_infos/nuscenes_infos_temporal_val.pkl \
    --nusc-root data/nuscenes \
    --output "$out" \
    $extra_args \
    > "$elog" 2>&1
  if [[ ! -f "$out" ]]; then
    log "  [$display] ERROR: result JSON missing — see $elog"
    return 1
  fi
  # Extract headline metrics; fall back to "n/a" on missing keys (NaN -> nan).
  /usr/bin/python3 -c "
import json
d = json.load(open('$out'))
l2 = d.get('L2_avg', float('nan'))
coll = d.get('collision_avg', float('nan')) * 100
turn = d.get('scenario_metrics', {}).get('turning', {})
tl2 = turn.get('L2_avg', float('nan')); tn = turn.get('n', 0)
beh = d.get('behavior', {})
he = beh.get('heading_error_rad', float('nan'))
br = beh.get('hard_brake_rate', float('nan'))
pr = beh.get('progress_ratio', float('nan'))
print(f'$tag,$display,{l2:.4f},{coll:.2f},{tl2:.4f},{tn},{he:.4f},{br:.4f},{pr:.4f}')
" >> /tmp/all_ckpts_v2_results.csv
  log "  [$display] DONE"
  tail -1 /tmp/all_ckpts_v2_results.csv | awk -F, '{ printf("    L2_avg=%s  coll=%s%%  turning_L2=%s (n=%s)  heading_err=%s  brake_rate=%s  progress=%s\n", $3, $4, $5, $6, $7, $8, $9) }'
}

run_eval "$CKPT_R1"      ""                                                                          "R1prime_1cam"      "R1' / A.0 baseline (1-cam camera-only)"
run_eval "$CKPT_R1PRIME3" "--planning-cams CAM_FRONT,CAM_FRONT_LEFT,CAM_FRONT_RIGHT"                 "R1prime_3cam"      "R1'' 3-cam baseline"
run_eval "$CKPT_B5"      "--multimodal"                                                              "B5_multimodal"     "B.5 multimodal eval (TRUE)"
run_eval "$CKPT_B5"      ""                                                                          "B5_cameraonly"     "B.5 camera-only eval"
run_eval "$CKPT_B6"      "--multimodal"                                                              "B6_multimodal"     "B.6 multimodal eval (TRUE)"
run_eval "$CKPT_B6"      ""                                                                          "B6_cameraonly"     "B.6 camera-only eval"

echo
log "==================================================================="
log "FINAL TABLE (v2 with sub-scenario + behavior metrics)"
log "==================================================================="
column -t -s, /tmp/all_ckpts_v2_results.csv

{
  echo "# All-ckpt eval v2 (fixed multimodal + new metrics) $(date -Iseconds)"
  echo
  echo "| Tag | L2_avg | Coll% | turning_L2 | turn_n | heading_err | brake_rate | progress |"
  echo "|---|---|---|---|---|---|---|---|"
  tail -n +2 /tmp/all_ckpts_v2_results.csv | while IFS=, read -r tag display l2 coll tl2 tn he br pr; do
    echo "| $display | $l2 | $coll | $tl2 | $tn | $he | $br | $pr |"
  done
} > "$SUMMARY"

log "Summary -> $SUMMARY"
