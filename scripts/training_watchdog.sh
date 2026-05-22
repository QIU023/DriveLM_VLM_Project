#!/usr/bin/env bash
# Training watchdog: verify B.5' (and any subsequent queued training) is
# alive, log file is fresh, and loss curve isn't pathological. Logs to
# logs/training_watchdog.log every minute. Does NOT auto-kill — alerting only.
#
# Used in overnight mode; disk_watchdog.sh handles disk panic separately.
#
# Watchdog signals (per minute):
#   HANG:    log file hasn't grown in >5 min while pid is alive
#   DEAD:    pid is gone (training crashed or finished)
#   NAN:     loss is NaN or > 100 (training diverged)
#   STALE:   no opt_step progression in 3 consecutive minutes
#   OK:      pid alive + log fresh + last loss in [0, 100]
#
# Usage:
#   bash scripts/training_watchdog.sh > logs/training_watchdog.log 2>&1 &
#
# Stop:
#   pkill -f training_watchdog.sh

set -u
INTERVAL=60
STALE_THRESH_S=300   # 5 min — if log file mtime > this old, mark HANG

PID_FILES=(
  "logs/b5prime_3cam.pid"          # B.5' current
  "logs/r1ppp_3cam_v2.pid"          # R1''' queued
)

cd "$(dirname "$0")/.." || exit 1

echo "[training_watchdog] start $(date -Iseconds) | interval ${INTERVAL}s | stale_threshold ${STALE_THRESH_S}s"

last_step=-1
stale_count=0

while true; do
  ts=$(date -Iseconds)
  active_pid=""
  active_log=""
  active_tag=""

  # Find which queued run is currently active.
  for pf in "${PID_FILES[@]}"; do
    [[ -f "$pf" ]] || continue
    p=$(cat "$pf" 2>/dev/null)
    if [[ -n "$p" ]] && kill -0 "$p" 2>/dev/null; then
      active_pid="$p"
      # Map pid file → matching log
      base=$(basename "$pf" .pid)
      active_log=$(ls -t logs/${base}_*.log 2>/dev/null | head -1)
      active_tag="$base"
      break
    fi
  done

  if [[ -z "$active_pid" ]]; then
    echo "[${ts}] IDLE no active training pid (b5prime+r1ppp both inactive)"
    sleep "${INTERVAL}"
    continue
  fi

  if [[ -z "$active_log" || ! -f "$active_log" ]]; then
    echo "[${ts}] WARN  ${active_tag} pid=${active_pid} alive but no log file found"
    sleep "${INTERVAL}"
    continue
  fi

  # Log file freshness
  now_s=$(date +%s)
  mtime_s=$(stat -c %Y "$active_log" 2>/dev/null || echo "$now_s")
  age_s=$(( now_s - mtime_s ))
  if (( age_s > STALE_THRESH_S )); then
    echo "[${ts}] HANG  ${active_tag} pid=${active_pid} log unchanged for ${age_s}s (>${STALE_THRESH_S}s)"
    sleep "${INTERVAL}"
    continue
  fi

  # Last logged loss + step from training tqdm line
  last_loss=$(grep -oE 'loss\(w100\)=[0-9.NaN]+' "$active_log" | tail -1 | sed 's/.*=//')
  cur_step=$(grep -oE 'opt_step=[0-9]+/' "$active_log" | tail -1 | tr -dc '0-9')
  cur_step=${cur_step:-0}

  # NaN / divergence check
  if [[ "$last_loss" == "NaN" || "$last_loss" == "nan" ]]; then
    echo "[${ts}] NAN   ${active_tag} pid=${active_pid} loss is NaN at step=${cur_step}"
    sleep "${INTERVAL}"
    continue
  fi
  if [[ -n "$last_loss" ]]; then
    is_high=$(awk -v x="$last_loss" 'BEGIN{print (x > 100) ? 1 : 0}')
    if [[ "$is_high" == "1" ]]; then
      echo "[${ts}] DIVR  ${active_tag} pid=${active_pid} loss=${last_loss} > 100 at step=${cur_step}"
      sleep "${INTERVAL}"
      continue
    fi
  fi

  # Step-progression check
  if (( cur_step == last_step )); then
    stale_count=$(( stale_count + 1 ))
    if (( stale_count >= 3 )); then
      echo "[${ts}] STALE ${active_tag} pid=${active_pid} step stuck at ${cur_step} for ${stale_count} consecutive ${INTERVAL}s windows"
      sleep "${INTERVAL}"
      continue
    fi
  else
    stale_count=0
  fi
  last_step="$cur_step"

  echo "[${ts}] ok    ${active_tag} pid=${active_pid} step=${cur_step} loss=${last_loss:-?}"
  sleep "${INTERVAL}"
done
