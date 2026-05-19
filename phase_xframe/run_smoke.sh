#!/usr/bin/env bash
# Smoke driver for the 3 cross-frame compressors (5 opt steps each).
#
# Usage:
#   bash phase_xframe/run_smoke.sh meanpool
#   bash phase_xframe/run_smoke.sh vtm
#   bash phase_xframe/run_smoke.sh longvu
#
# Each invocation:
#   - 8 GPUs FSDP via accelerate_configs/fsdp_8gpu.yaml
#   - --max-steps 5  (stop after 5 optimizer steps)
#   - --save-every 99999  (no ckpts written)
#   - --no-validate  (skip eval pass)
#   - --train-max-samples 64  (tiny train slice for fast smoke; with grad_accum=4
#     world=8 that's 64/8 = 8 per-rank batches -> 2 opt steps per pass, enough
#     to hit 5 steps in <= ~3 mini-passes due to dataloader cycling)
set -euo pipefail

variant=${1:?'usage: bash run_smoke.sh {meanpool|vtm|longvu}'}
config="configs/nuscenes_planning_16f_${variant}.yaml"
if [[ ! -f "$config" ]]; then
  echo "Missing config: $config" >&2; exit 2
fi

cd "$(dirname "$0")/.."
mkdir -p logs/xframe_smoke
log="logs/xframe_smoke/${variant}_$(date +%Y%m%d-%H%M%S).log"
echo "[smoke] $variant -> $log"

# Force experiment dir to /tmp so any accidental ckpt write lands in a
# disposable place. We use train_lora's --experiment to redirect the dir name
# (output_dir = checkpoints_qwen25/<experiment>), but we ALSO pass
# --save-every 99999 so the save guard never fires.
exp_name="xframe_smoke_${variant}"

# Disk discipline: refuse to start if /workspace free < 15 GB.
free_gb=$(df -BG /workspace | awk 'NR==2 {gsub("G","",$4); print $4}')
if (( free_gb < 15 )); then
  echo "[smoke] ABORT: /workspace free=${free_gb}G < 15G threshold" >&2; exit 3
fi

accelerate launch --config_file accelerate_configs/fsdp_8gpu.yaml \
  scripts/train_lora.py \
    --config "$config" \
    --experiment "$exp_name" \
    --max-steps 5 \
    --save-every 99999 \
    --no-validate \
    --no-final-save \
    --train-max-samples 64 \
  2>&1 | tee "$log"

echo "[smoke] done: $log"
