#!/usr/bin/env bash
# B.5 → B.6 chained orchestrator (image-as-modality multi-modal SFT).
# Same pattern as temporal_compress_orchestrator.sh: train -> DP eval ->
# backup result -> compare to R1' baseline -> cleanup intermediates.
#
# B.5 = camera + HD map BEV + bbox text + ego state (no dropout)
# B.6 = B.5 + modality_dropout_p 0.1 (HD map + bbox can drop; cam + ego never)
#
# Per user 2026-05-22 priority shift: B.5 + B.6 promoted; D.1 dropped; F.1
# scheduled after. NO 16f phases.

set -uo pipefail
cd "$(dirname "$0")/.."

TS=$(date -Iseconds | tr ':' '-')
LOG="logs/b5b6_${TS}.log"
SUMMARY="logs/b5b6_summary.md"
mkdir -p logs eval_results/track_b

exec > >(tee -a "$LOG") 2>&1

log() { echo "[$(date -Iseconds)] $*"; }
md() { echo "$@" >> "$SUMMARY"; }

wait_for_train_done() {
  local who="$1"
  local last_step=""
  while pgrep -f "scripts/train_lora.py" > /dev/null; do
    sleep 60
    local lastlog=$(ls -t logs/${who}_*.log 2>/dev/null | head -1)
    if [[ -n "$lastlog" ]]; then
      local step=$(tail -c 4000 "$lastlog" 2>/dev/null | tr '\r' '\n' | grep -oE "opt_step=[0-9]+/[0-9]+" | tail -1)
      if [[ -n "$step" && "$step" != "$last_step" ]]; then
        log "  [$who progress] $step"
        last_step="$step"
      fi
    fi
  done
  log "  [$who] training process exited"
}

cleanup_intermediates() {
  local exp="$1"
  local dir="checkpoints_qwen25/$exp"
  if [[ -d "$dir" ]]; then
    log "  [cleanup] removing $dir/checkpoint-* (keeping final/)"
    rm -rf "$dir"/checkpoint-*
    du -sh "$dir" 2>/dev/null
  fi
}

run_dp_eval() {
  local ckpt="$1" out="$2" suffix="$3"
  log "[DP eval] ckpt=$ckpt -> $out"
  BATCH_SIZE=16 bash scripts/launch_planning_eval_dp.sh \
    "$ckpt" \
    --infos-val data/uniad_infos/nuscenes_infos_temporal_val.pkl \
    --nusc-root data/nuscenes \
    --output "$out" \
    > "logs/dp_eval_${suffix}.log" 2>&1
  log "[DP eval] done"
}

compare_baseline() {
  local name="$1" json="$2" backup="$3"
  if [[ ! -f "$json" ]]; then
    log "  [compare $name] ERROR: result file $json missing"
    md "## $name — ❌ ERROR (no result JSON)"
    md ""
    return 1
  fi
  cp "$json" "eval_results/track_b/${backup}.json"
  log "  [compare $name] result -> eval_results/track_b/${backup}.json"
  local l2=$(/usr/bin/python3 -c "import json; d=json.load(open('$json')); print(f\"{d['L2_avg']:.4f}\")" 2>/dev/null)
  local coll=$(/usr/bin/python3 -c "import json; d=json.load(open('$json')); print(f\"{d['collision_avg']*100:.2f}\")" 2>/dev/null)
  log "  [compare $name] L2_avg=$l2 collision_avg=$coll%"
  log "  [compare $name] R1' baseline: L2_avg=0.6423 collision=3.73%"
  /usr/bin/python3 -c "
l2 = float('$l2')
if l2 < 0.4:
    print(f'    [SANITY WARN] L2={l2} < 0.4 — too good, suspect bug')
elif l2 > 1.0:
    print(f'    [SANITY WARN] L2={l2} > 1.0 — too bad, training broken')
else:
    print(f'    [SANITY OK] L2={l2} in [0.4, 1.0]')
"
  md "## $name — L2_avg=$l2  collision_avg=$coll%"
  md "  vs R1' baseline 0.6423 / 3.73%"
  md ""
}

run_phase() {
  local exp="$1" cfg="$2" prefix="$3" backup="$4" display="$5"
  log "==================================================================="
  log "PHASE: $display"
  log "==================================================================="
  log "  cfg=$cfg  exp=$exp"
  df -h /workspace | tail -1
  rm -rf "checkpoints_qwen25/$exp/"
  TS_inner=$(date -Iseconds | tr ':' '-')
  local run_log="logs/${prefix}_${TS_inner}.log"
  nohup accelerate launch --config_file accelerate_configs/fsdp_8gpu.yaml \
    scripts/train_lora.py --config "$cfg" \
    > "$run_log" 2>&1 &
  sleep 30
  wait_for_train_done "$prefix"
  sleep 30
  run_dp_eval "checkpoints_qwen25/$exp/final" "/tmp/eval_${backup}.json" "$backup"
  compare_baseline "$display" "/tmp/eval_${backup}.json" "$backup"
  cleanup_intermediates "$exp"
  log ""
}

md "# B.5 + B.6 image-as-modality SFT $(date -Iseconds)"
md ""
md "Branch HEAD: $(git rev-parse HEAD)"
md ""

run_phase \
  "nusc_planning_b5_multimodal" \
  "configs/nuscenes_planning_b5.yaml" \
  "b5_multimodal" \
  "B5_multimodal" \
  "B.5 image-as-modality (cam + HD map + bbox + ego)"

run_phase \
  "nusc_planning_b6_multimodal_dropout" \
  "configs/nuscenes_planning_b6.yaml" \
  "b6_multimodal_dropout" \
  "B6_multimodal_dropout" \
  "B.6 = B.5 + modality dropout p=0.1"

log "==================================================================="
log "B.5 + B.6 ORCHESTRATOR DONE at $(date -Iseconds)"
log "==================================================================="
md ""
md "## Final summary"
df -h /workspace | tail -1 | tee -a "$SUMMARY"
md ""
ls -la eval_results/track_b/ | tee -a "$SUMMARY"
