#!/bin/bash
# GRPO overnight watchdog v2: + auto-clean core dumps every 30s
LOG_GLOB="${LOG_GLOB:-/workspace/DriveLM_VLM_Project/grpo_vla/logs/*.log}"
INTERVAL=30
last_step=-1
last_heartbeat=$(date +%s)
declare -A seen_lines
while true; do
  # Auto-clean any new vastai core dumps (each ~2-5G) — match ALL core-* patterns
  CORES=$(ls /var/lib/vastai_kaalia/data/core-* 2>/dev/null | wc -l)
  if [ "$CORES" -gt 0 ]; then
    rm -f /var/lib/vastai_kaalia/data/core-* 2>&1
    echo "AUTO-CLEAN: removed $CORES core dump(s) (VLLM/sglang/etc)"
  fi
  # disk
  FREE_G=$(df --output=avail -BG /workspace | tail -1 | tr -d 'G ')
  if [[ "$FREE_G" -lt 15 ]]; then
    echo "PANIC: disk free=${FREE_G}G <15G — kill GRPO?"
  elif [[ "$FREE_G" -lt 25 ]]; then
    echo "WARN: disk free=${FREE_G}G <25G"
  fi
  for f in $LOG_GLOB; do
    [ -f "$f" ] || continue
    fatal=$(tail -200 "$f" 2>/dev/null | grep -E "(Traceback|CUDA out of memory|OOM|loss=nan|loss = nan|NaN|Killed|RuntimeError|EngineDeadError)" | tail -3)
    if [ -n "$fatal" ]; then
      while IFS= read -r ln; do
        key=$(echo "$ln" | md5sum | cut -c1-12)
        if [ -z "${seen_lines[$key]}" ]; then
          echo "FATAL [$f]: $ln"
          seen_lines[$key]=1
        fi
      done <<< "$fatal"
    fi
    progress=$(tail -50 "$f" 2>/dev/null | grep -oE "(global_step|step):[ ]*[0-9]+" | tail -1)
    if [ -n "$progress" ]; then
      step_num=$(echo "$progress" | grep -oE "[0-9]+$")
      if [ "$step_num" -gt "$last_step" ] && [ $((step_num % 25)) -eq 0 ]; then
        echo "PROGRESS [$f]: $progress  free=${FREE_G}G"
        last_step=$step_num
      fi
    fi
  done
  now=$(date +%s)
  if [[ $((now - last_heartbeat)) -ge 600 ]]; then
    echo "HEARTBEAT: free=${FREE_G}G  last_step=${last_step}"
    last_heartbeat=$now
  fi
  sleep $INTERVAL
done
