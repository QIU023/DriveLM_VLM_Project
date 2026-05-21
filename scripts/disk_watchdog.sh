#!/usr/bin/env bash
# Disk watchdog: monitor /workspace free space + alert at thresholds + emergency kill
# at panic threshold. Background sidecar for overnight runs.
#
# Thresholds (per memory rule feedback_disk_panic_protocol):
#   WARN:   < 25 GB free  -> log warning
#   ALERT:  < 18 GB free  -> log alert + send PushNotification (if available)
#   PANIC:  < 12 GB free  -> kill all train_lora.py procs (last-resort)
#
# Usage:
#   bash scripts/disk_watchdog.sh > logs/disk_watchdog.log 2>&1 &
#
# To stop:
#   pkill -f disk_watchdog.sh

set -u
WORKSPACE="/workspace"
WARN_GB=25
ALERT_GB=18
PANIC_GB=12
INTERVAL=60

echo "[disk_watchdog] start $(date -Iseconds) | WARN<${WARN_GB}G ALERT<${ALERT_GB}G PANIC<${PANIC_GB}G | interval ${INTERVAL}s"

while true; do
  free_g=$(df --output=avail -BG "${WORKSPACE}" | tail -1 | tr -dc '0-9')
  ts=$(date -Iseconds)
  if (( free_g < PANIC_GB )); then
    echo "[${ts}] PANIC free=${free_g}G < ${PANIC_GB}G  -- killing train_lora.py procs"
    pkill -9 -f "scripts/train_lora.py" 2>/dev/null || true
    pkill -9 -f "scripts/planning_eval.py" 2>/dev/null || true
    echo "[${ts}] PANIC: emergency kill issued; watchdog continues running"
    sleep "${INTERVAL}"
    continue
  fi
  if (( free_g < ALERT_GB )); then
    echo "[${ts}] ALERT free=${free_g}G < ${ALERT_GB}G  -- training will be killed at ${PANIC_GB}G"
    # Top 5 largest dirs for diagnostics
    du -sh "${WORKSPACE}/DriveLM_VLM_Project/checkpoints_qwen25"/* 2>/dev/null | sort -hr | head -5 | sed 's/^/  /'
  elif (( free_g < WARN_GB )); then
    echo "[${ts}] WARN  free=${free_g}G < ${WARN_GB}G"
  else
    echo "[${ts}] ok    free=${free_g}G"
  fi
  sleep "${INTERVAL}"
done
