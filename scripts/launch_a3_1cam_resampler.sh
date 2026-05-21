#!/usr/bin/env bash
# Track A.3: 1-cam x 4f + Perceiver Resampler-64 projector (Flamingo-style,
# temporal-aware, ~90M params).
# Anchor: R1' = A.0 (1-cam x 4f + Linear projector, L2 0.6423, collision 3.73%).
# Companions:
#   A.1 = Q-Former-64       (BLIP-2-style 64q x 6L, NO temporal pos, ~81M)
#   A.2 = PixelShuffle 2x   (deterministic 2x2 space-to-depth + Linear, ~17M)
#   A.3 = Perceiver Resampler-64 (THIS RUN; latent self-attn + cross-attn +
#                                 per-frame temporal_pos[t_max=32], 90.44M)
# Hypothesis: per-frame learnable temporal_pos + latent self-attn give the
# Perceiver Resampler the right inductive bias for short multi-frame driving
# clips — beating both the no-temporal-pos Q-Former (A.1) AND the determ.
# PixelShuffle (A.2) at the SAME 64-token / cam LM budget.
#
# Token budget: 1 cam x 64 latents = 64 LM-input visual tokens / sample
# (matches A.1's qformer count; A.2 was 180 due to 10x18 even-grid requirement).
# Resampler is data-shape agnostic — base 109760 min/max_pixels (8x15 post-
# merger) works fine, no min/max_pixels override needed. Verified via
# scripts/_smoke_resampler_roundtrip.py against (4,8,15) AND (4,7,14)
# (both odd dims).
#
# Same overrides as A.1: AC off, LBS=4 x grad_accum=1 (GBS=32 via FSDP=8),
# warmup=36 / lr_step=143 (ratio-scaled to ~2050 total steps),
# save_every=val_every=50, full_l2_every=null (in-train DP eval design broken
# even with summon_full_params+unwrap fix; A.3 final L2 via post-train
# standalone launch_planning_eval_dp.sh, same path as A.0/A.1/A.2).
set -euo pipefail
cd "$(dirname "$0")/.."

STAMP=$(date -Iseconds | tr ':' '-')
LOG="logs/a3_1cam_resampler_${STAMP}.log"
mkdir -p logs

free_g=$(df --output=avail -BG /workspace | tail -1 | tr -dc '0-9')
if (( free_g < 25 )); then
  echo "DISK PANIC: free ${free_g}G < 25G; aborting." >&2
  exit 9
fi

echo "[A.3] starting at $(date -Iseconds), log=$LOG, free=${free_g}G"
nohup accelerate launch --config_file accelerate_configs/fsdp_8gpu.yaml \
  scripts/train_lora.py --config configs/nuscenes_planning_1cam_resampler.yaml \
  > "$LOG" 2>&1 &
echo "[A.3] pid=$!  log=$LOG"
