#!/usr/bin/env bash
# Track A.2: 1-cam × 4f + PixelShuffle 2× + Linear projector (~17M params).
# Anchor: R1' = A.0 (1-cam × 4f + Linear projector, L2 0.6423, collision 3.73%).
# Companion: A.1 = Q-Former-64 (same 1-cam × 4f setup, BLIP-2-style 64q × 6L).
# Hypothesis: deterministic 2×2 space-to-depth + single Linear projector beats
# both the in-encoder PatchMerger-MLP linear baseline AND the learned-query
# Q-Former at fusing 4-frame video into the LM token stream — by trading
# adaptive query selection for a smaller, easier-to-train parameter budget
# (~17M vs ~81M) at a comparable post-projector token count (140 vs 64).
#
# Same overrides as A.1: AC off, LBS=4 × grad_accum=1 (GBS=32 via FSDP=8),
# warmup=36/lr_step=143 (ratio-scaled to ~2050 total steps),
# save_every=val_every=50, full_l2_every=250 (re-enabled — the
# summon_full_params + unwrap fix from a3f46fc cleared the mid-train DP eval
# deadlock; pixelshuffle's generate path mirrors qformer so the same fix
# applies).
set -euo pipefail
cd "$(dirname "$0")/.."

STAMP=$(date -Iseconds | tr ':' '-')
LOG="logs/a2_1cam_pixelshuffle_${STAMP}.log"
mkdir -p logs

free_g=$(df --output=avail -BG /workspace | tail -1 | tr -dc '0-9')
if (( free_g < 25 )); then
  echo "DISK PANIC: free ${free_g}G < 25G; aborting." >&2
  exit 9
fi

echo "[A.2] starting at $(date -Iseconds), log=$LOG, free=${free_g}G"
nohup accelerate launch --config_file accelerate_configs/fsdp_8gpu.yaml \
  scripts/train_lora.py --config configs/nuscenes_planning_1cam_pixelshuffle.yaml \
  > "$LOG" 2>&1 &
echo "[A.2] pid=$!  log=$LOG"
