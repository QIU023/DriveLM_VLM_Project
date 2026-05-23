#!/usr/bin/env bash
# 5-step smoke for A.1 v2 (BLIP-2 pretrained Q-Former).
# Verifies: BLIP-2 weights load, projector forward+backward, save+reload.
set -uo pipefail
cd "$(dirname "$0")/.."

STAMP=$(date -Iseconds | tr ':' '-')
LOG="logs/aseries/smoke_a1v2_${STAMP}.log"
EXP="smoke_a1_v2_blip2_${STAMP}"
mkdir -p logs/aseries

echo "[smoke A.1 v2] starting $(date -Iseconds), log=$LOG"

accelerate launch --config_file accelerate_configs/fsdp_8gpu.yaml \
  scripts/train_lora.py \
    --config configs/nuscenes_planning_1cam_qformer_blip2.yaml \
    --experiment "$EXP" \
    --max-steps 5 \
    --no-validate \
    --save-every 999 \
  2>&1 | tee "$LOG"

CKPT="checkpoints_qwen25/${EXP}/final"
echo
echo "=== verify final/ ==="
ls -la "$CKPT" 2>&1 | head -10
du -sh "$CKPT" 2>&1
echo
echo "=== external projector saved? ==="
ls -la "${CKPT}/" 2>&1 | grep -iE "projector|qformer"
echo
echo "=== reload test ==="
/usr/bin/python3 -c "
import torch
from transformers import AutoModelForImageTextToText
m = AutoModelForImageTextToText.from_pretrained('$CKPT', torch_dtype=torch.bfloat16)
print(f'OK: model loaded, {sum(p.numel() for p in m.parameters())/1e9:.3f}B params')
" 2>&1
echo
echo "[smoke A.1 v2] done $(date -Iseconds)"
