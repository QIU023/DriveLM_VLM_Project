#!/usr/bin/env bash
# 5-step E2E smoke for Qwen3-VL-4B × 1-cam × multimodal config.
# Validates: backbone load, processor pipeline, FSDP wrap (Qwen3VLTextDecoderLayer),
# real-data forward+backward, weights-only ckpt save, reload round-trip.
#
# Usage: bash scripts/smoke_1cam_qwen3vl_5step.sh
set -uo pipefail   # NOT -e: we want post-train reload check to run even if train logs warn
cd "$(dirname "$0")/.."

STAMP=$(date -Iseconds | tr ':' '-')
LOG="logs/qwen3vl/smoke_5step_${STAMP}.log"
SMOKE_EXP="smoke_1cam_qwen3vl_5step_${STAMP}"
mkdir -p logs/qwen3vl

echo "[smoke] starting at $(date -Iseconds)"
echo "[smoke] log=$LOG"
echo "[smoke] experiment=$SMOKE_EXP"

# 5-step train + final save (no intermediate saves; --save-every 999 ensures
# only the post-train _save_model_and_state writes the ckpt — exercises both
# weights-only save path AND FSDP gather of full state_dict on rank 0).
accelerate launch --config_file accelerate_configs/fsdp_8gpu.yaml \
  scripts/train_lora.py \
    --config configs/nuscenes_planning_1cam_qwen3vl_multimodal.yaml \
    --experiment "$SMOKE_EXP" \
    --max-steps 5 \
    --no-validate \
    --save-every 999 \
  2>&1 | tee "$LOG"

CKPT_DIR="checkpoints_qwen25/${SMOKE_EXP}/final"
echo
echo "=== verify ckpt was written ==="
ls -la "$CKPT_DIR" 2>&1 | head -20
echo
echo "=== verify model.safetensors loads back ==="
/usr/bin/python3 -c "
import sys
from transformers import AutoModelForImageTextToText, AutoProcessor
import torch
ckpt = '$CKPT_DIR'
print(f'loading from {ckpt} ...')
model = AutoModelForImageTextToText.from_pretrained(ckpt, torch_dtype=torch.bfloat16)
proc = AutoProcessor.from_pretrained(ckpt)
n = sum(p.numel() for p in model.parameters()) / 1e9
print(f'OK: loaded model with {n:.3f}B params, processor type {type(proc).__name__}')
" 2>&1

echo
echo "=== ckpt dir size ==="
du -sh "$CKPT_DIR" 2>&1
echo
echo "=== verify NO accelerate_state dir (weights_only=true default) ==="
if [[ -d "${CKPT_DIR}/accelerate_state" ]]; then
  echo "FAIL: accelerate_state exists — weights_only default broken"
  ls -la "${CKPT_DIR}/accelerate_state"
else
  echo "OK: no accelerate_state/ — weights_only default working"
fi
echo
echo "[smoke] done at $(date -Iseconds)"
