#!/usr/bin/env bash
# Overnight orchestrator: A.3 v2 finish → DP eval → A.2 v2 retrain → eval →
# A.1 v2 retrain → eval → 2D compression Pareto sweep.
#
# Per user 2026-05-21 instructions: do NOT kill on errors (user is sleeping);
# log + continue. Watchdog process kills only on disk panic (<12G).
#
# Each phase logs to logs/overnight_<ts>.log + a phase-summary appended to
# logs/overnight_summary.md so morning review is fast.

# DELIBERATELY no set -e: phases must continue on error per user instruction.
set -uo pipefail
cd "$(dirname "$0")/.."

TS=$(date -Iseconds | tr ':' '-')
LOG="logs/overnight_${TS}.log"
SUMMARY="logs/overnight_summary.md"
mkdir -p logs eval_results/track_a

exec > >(tee -a "$LOG") 2>&1

log() {
  echo "[$(date -Iseconds)] $*"
}
md() {
  echo "$@" >> "$SUMMARY"
}

# ============================================================================
# Helpers
# ============================================================================

wait_for_train_done() {
  # Block until no train_lora.py process is running.
  local who="$1"
  local last_step=""
  while pgrep -f "scripts/train_lora.py" > /dev/null; do
    sleep 60
    # Best-effort progress probe
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
  # $1 = ckpt path  $2 = output json  $3 = log name suffix
  local ckpt="$1"
  local out="$2"
  local suffix="$3"
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
  # $1 = run_name  $2 = result json path  $3 = backup name (in eval_results/track_a/)
  local name="$1"
  local json="$2"
  local backup="$3"
  if [[ ! -f "$json" ]]; then
    log "  [compare $name] ERROR: result file $json missing"
    md "## $name — ❌ ERROR (no result JSON)"
    md ""
    return 1
  fi
  cp "$json" "eval_results/track_a/${backup}.json"
  log "  [compare $name] result backed up to eval_results/track_a/${backup}.json"
  local l2=$(/usr/bin/python3 -c "import json; d=json.load(open('$json')); print(f\"{d['L2_avg']:.4f}\")" 2>/dev/null)
  local coll=$(/usr/bin/python3 -c "import json; d=json.load(open('$json')); print(f\"{d['collision_avg']*100:.2f}\")" 2>/dev/null)
  log "  [compare $name] L2_avg=$l2 collision_avg=$coll%"
  log "  [compare $name] R1' baseline: L2_avg=0.6423 collision=3.73%"
  # Sanity: warn if L2 < 0.4 (suspiciously good = likely bug) or > 1.0 (suspiciously bad)
  /usr/bin/python3 -c "
l2 = float('$l2')
if l2 < 0.4:
    print(f'    [SANITY WARN] L2={l2} < 0.4 — too good, suspect bug (eval set leak?)')
elif l2 > 1.0:
    print(f'    [SANITY WARN] L2={l2} > 1.0 — too bad, suspect training broken')
else:
    print(f'    [SANITY OK] L2={l2} in expected range [0.4, 1.0]')
"
  md "## $name — L2_avg=$l2  collision_avg=$coll%"
  md "  vs R1' baseline 0.6423 / 3.73%"
  md ""
}

# ============================================================================
# PHASE 1: wait for A.3 v2 to finish, eval, cleanup
# ============================================================================
log "==================================================================="
log "PHASE 1: A.3 v2 (resampler, with sync fix) — wait for finish + eval"
log "==================================================================="
md "# Overnight $(date -Iseconds)"
md ""
md "Branch HEAD: $(git rev-parse HEAD)"
md ""

wait_for_train_done "a3_1cam_resampler"
sleep 30  # let final/ save complete

run_dp_eval \
  "checkpoints_qwen25/nuscenes_planning_1cam_resampler_3b_full_sft/final" \
  "/tmp/eval_a3_v2.json" \
  "a3_v2"
compare_baseline "A.3 v2 (1-cam Resampler, sync fix)" "/tmp/eval_a3_v2.json" "A3_1cam_resampler_v2"
cleanup_intermediates "nuscenes_planning_1cam_resampler_3b_full_sft"

# ============================================================================
# PHASE 2: A.2 v2 retrain (pixelshuffle with sync fix)
# ============================================================================
log "==================================================================="
log "PHASE 2: A.2 v2 retrain (1-cam PixelShuffle, sync fix)"
log "==================================================================="
log "  pre-launch: numpy=$(/usr/bin/python3 -c 'import numpy;print(numpy.__version__)')"
df -h /workspace | tail -1

# Wipe any leftover (final/ of buggy v3 was already cleaned)
rm -rf checkpoints_qwen25/nuscenes_planning_1cam_pixelshuffle_3b_full_sft/

bash scripts/launch_a2_1cam_pixelshuffle.sh
sleep 30
wait_for_train_done "a2_1cam_pixelshuffle"
sleep 30

run_dp_eval \
  "checkpoints_qwen25/nuscenes_planning_1cam_pixelshuffle_3b_full_sft/final" \
  "/tmp/eval_a2_v4.json" \
  "a2_v4"
compare_baseline "A.2 v4 (1-cam PixelShuffle, sync fix)" "/tmp/eval_a2_v4.json" "A2_1cam_pixelshuffle_v4"
cleanup_intermediates "nuscenes_planning_1cam_pixelshuffle_3b_full_sft"

# ============================================================================
# PHASE 3: A.1 v2 retrain (qformer with sync fix)
# ============================================================================
log "==================================================================="
log "PHASE 3: A.1 v2 retrain (1-cam Q-Former, sync fix)"
log "==================================================================="
df -h /workspace | tail -1

rm -rf checkpoints_qwen25/nuscenes_planning_1cam_qformer_3b_full_sft/

bash scripts/launch_a1_1cam_qformer.sh
sleep 30
wait_for_train_done "a1_1cam_qformer"
sleep 30

run_dp_eval \
  "checkpoints_qwen25/nuscenes_planning_1cam_qformer_3b_full_sft/final" \
  "/tmp/eval_a1_v6.json" \
  "a1_v6"
compare_baseline "A.1 v6 (1-cam Q-Former, sync fix)" "/tmp/eval_a1_v6.json" "A1_1cam_qformer_v6"
cleanup_intermediates "nuscenes_planning_1cam_qformer_3b_full_sft"

# ============================================================================
# PHASE 4: 2D compression Pareto sweep on R1' winner (training-free)
# ============================================================================
log "==================================================================="
log "PHASE 4: 2D compression Pareto sweep on R1' Linear baseline"
log "==================================================================="
df -h /workspace | tail -1

if [[ -f scripts/sweep_compress_pareto.sh ]]; then
  bash scripts/sweep_compress_pareto.sh \
    checkpoints_qwen25/nuscenes_planning_3b_full_sft/final \
    > "logs/sweep_compress_pareto_${TS}.log" 2>&1 || \
    log "  [sweep] non-zero exit, see logs/sweep_compress_pareto_${TS}.log"
  log "  [sweep] done"
  md ""
  md "## 2D compression Pareto sweep on R1' Linear baseline"
  md "  see logs/sweep_compress_pareto_${TS}.log"
else
  log "  [sweep] scripts/sweep_compress_pareto.sh missing; skipping"
  md "## 2D compression Pareto sweep — SKIPPED (script missing)"
fi

# ============================================================================
# DONE
# ============================================================================
log "==================================================================="
log "OVERNIGHT DONE at $(date -Iseconds)"
log "==================================================================="
md ""
md "## Summary"
df -h /workspace | tail -1 | tee -a "$SUMMARY"
md ""
ls -la eval_results/track_a/ | tee -a "$SUMMARY"
