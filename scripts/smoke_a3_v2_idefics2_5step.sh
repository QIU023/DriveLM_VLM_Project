#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
STAMP=$(date -Iseconds | tr ':' '-')
LOG="logs/aseries/smoke_a3v2_${STAMP}.log"
EXP="smoke_a3_v2_idefics2_${STAMP}"
mkdir -p logs/aseries
echo "[smoke A.3 v2] $(date -Iseconds), log=$LOG"
accelerate launch --config_file accelerate_configs/fsdp_8gpu.yaml \
  scripts/train_lora.py \
    --config configs/nuscenes_planning_1cam_resampler_idefics2.yaml \
    --experiment "$EXP" \
    --max-steps 5 \
    --no-validate \
    --save-every 999 \
  2>&1 | tee "$LOG"
CKPT="checkpoints_qwen25/${EXP}/final"
echo
echo "=== ckpt verify ==="
ls -la "$CKPT" 2>&1 | head -10
du -sh "$CKPT" 2>&1
echo
echo "=== reload test ==="
/usr/bin/python3 -c "
import torch
from transformers import AutoModelForImageTextToText
m = AutoModelForImageTextToText.from_pretrained('$CKPT', torch_dtype=torch.bfloat16)
print(f'LM OK: {sum(p.numel() for p in m.parameters())/1e9:.3f}B')
" 2>&1
echo "[smoke A.3 v2] done $(date -Iseconds)"
