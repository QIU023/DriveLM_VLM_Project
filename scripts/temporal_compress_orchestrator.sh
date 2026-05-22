#!/usr/bin/env bash
# Temporal token compression overnight: 8f/16f × meanpool/vtm/longvu (6 runs)
# All compressors are ZERO-PARAM variants (temporal_pool mean / vtm merge=mean /
# longvu cosine+norm), so the projector save/sync fixes do not apply.
#
# Per user instruction "可以不直接kill停下来 因为我在睡觉" — NO set -e; phases
# continue on errors so partial results ship.
#
# Each phase: train (~2-4h depending on frame count) -> DP eval (BS=16) ->
# backup to eval_results/track_c/ -> compare to R1' baseline + sanity check ->
# cleanup intermediate ckpts (keep final/).

set -uo pipefail
cd "$(dirname "$0")/.."

TS=$(date -Iseconds | tr ':' '-')
LOG="logs/temporal_compress_${TS}.log"
SUMMARY="logs/temporal_compress_summary.md"
mkdir -p logs eval_results/track_c

exec > >(tee -a "$LOG") 2>&1

log() { echo "[$(date -Iseconds)] $*"; }
md() { echo "$@" >> "$SUMMARY"; }

# ----------------------------------------------------------------------------
# helpers (mirror overnight_orchestrator.sh patterns)
# ----------------------------------------------------------------------------

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
  cp "$json" "eval_results/track_c/${backup}.json"
  log "  [compare $name] result -> eval_results/track_c/${backup}.json"
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
  # $1 = exp_name (matches experiment: in YAML)
  # $2 = config path
  # $3 = log prefix (matches launcher logs/<prefix>_<stamp>.log)
  # $4 = backup name in eval_results/track_c/
  # $5 = display name
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
  sleep 30  # let it spin up
  wait_for_train_done "$prefix"
  sleep 30  # final/ save settle
  run_dp_eval "checkpoints_qwen25/$exp/final" "/tmp/eval_${backup}.json" "$backup"
  compare_baseline "$display" "/tmp/eval_${backup}.json" "$backup"
  cleanup_intermediates "$exp"
  log ""
}

# ----------------------------------------------------------------------------
# 6 phases — 8f then 16f (lighter first so partial completion still useful)
# ----------------------------------------------------------------------------

md "# Temporal compression overnight $(date -Iseconds)"
md ""
md "Branch HEAD: $(git rev-parse HEAD)"
md ""

run_phase \
  "nusc_planning_8f_meanpool" \
  "configs/nuscenes_planning_8f_meanpool.yaml" \
  "8f_meanpool" \
  "8f_meanpool" \
  "8f mean-pool (R3) — zero-param temporal pool"

run_phase \
  "nusc_planning_8f_vtm" \
  "configs/nuscenes_planning_8f_vtm.yaml" \
  "8f_vtm" \
  "8f_vtm" \
  "8f VTM (R4) — zero-param video token merging"

run_phase \
  "nusc_planning_8f_longvu" \
  "configs/nuscenes_planning_8f_longvu.yaml" \
  "8f_longvu" \
  "8f_longvu" \
  "8f LongVU (R5) — cosine+norm prune"

# 16f phases skipped per user 2026-05-21 14:30Z — they want to focus on
# multi-modal B (cross-modal Q-Former) next, not extend frame count.
log "==================================================================="
log "16f phases SKIPPED per user — stopping orchestrator after 8f cycle"
log "==================================================================="
md ""
md "## 16f phases SKIPPED (user request)"
md "  Reason: focus on multi-modal B (cross-modal Q-Former) over 16f frame extension"

log "==================================================================="
log "TEMPORAL COMPRESS ORCHESTRATOR DONE at $(date -Iseconds)"
log "==================================================================="
md ""
md "## Summary"
df -h /workspace | tail -1 | tee -a "$SUMMARY"
md ""
ls -la eval_results/track_c/ | tee -a "$SUMMARY"
