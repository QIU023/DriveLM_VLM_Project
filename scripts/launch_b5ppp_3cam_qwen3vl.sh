#!/usr/bin/env bash
# Launch B.5''' — Qwen3-VL-4B × 3-cam × multi-modal full SFT (3 epochs)
#
# = B.5' (Qwen2.5-VL-3B 3-cam multimodal) field-by-field 1:1, only backbone
# swapped to Qwen3-VL-4B for TRT-LLM 1.3 native deploy (modeling_qwen3vl.py
# only — NVIDIA never added modeling_qwen2_5_vl.py per issues #10069, #12824).
#
# Pre-launch checks (per [[feedback_pre_launch_checklist]]):
#   - max_length token audit:  docs/B5ppp_token_audit.md (max=9139 vs 12288, 0% over)
#   - boot warnings:           grep WARN in first 200 lines below
#   - real-data smoke:          this script's MODE=smoke 5-step run
#   - disk:                     this script's pre-flight 30G floor
#   - ETA:                     2625 steps × ~15s = ~11h on 8×RTX-5090
#   - paper-hyperparam audit:   inherits B.5' (paper-aligned AutoVLA 3-cam)
#
# MODE=smoke runs 5-step only (validate OOM/NaN/M-RoPE/step-time), no ckpt.
# MODE=full runs 3 epochs full SFT.
set -uo pipefail
cd "$(dirname "$0")/.."

MODE=${MODE:-full}
STAMP=$(date -Iseconds | tr ':' '-')
LOG_DIR="logs/qwen3vl"
LOG="$LOG_DIR/b5ppp_${MODE}_${STAMP}.log"
mkdir -p "$LOG_DIR"

# Disk floor (per [[feedback_disk_panic_protocol]])
FREE_G=$(df -BG /workspace | awk 'NR==2{gsub("G","",$4); print $4}')
if [ "${FREE_G:-0}" -lt 30 ]; then
  echo "ABORT: disk free ${FREE_G}G < 30G floor" >&2
  exit 2
fi
echo "[launch] disk free=${FREE_G}G OK"

# vastai core dump prevention (5G each can fill disk fast)
ulimit -c 0 || true

# HF cache
export HF_HOME=${HF_HOME:-/workspace/.hf_home}

# Qwen3-VL-4B + native 3-cam 9100 tokens exceeds 32GB w/o offload; param offload
# moves FSDP-sharded params to CPU between fwd passes. Cost: ~30-50% step time,
# fit gain: enables LBS=1 GA=4 on 32GB 5090. Set FSDP_CPU_OFFLOAD=0 to disable.
export FSDP_CPU_OFFLOAD=${FSDP_CPU_OFFLOAD:-0}
# expandable_segments mitigates fragmentation across train steps (smoke v4 saw
# step 0 succeed then step 1 OOM at same shape — classic frag).
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

echo "[launch] mode=$MODE starting at $(date -Iseconds)"
echo "[launch] log=$LOG"
echo "[launch] config=configs/nuscenes_planning_3cam_qwen3vl_multimodal.yaml"

EXTRA=""
if [ "$MODE" = "smoke" ]; then
  EXTRA="--max-steps 5 --save-every 99999 --val-every 99999"
fi

accelerate launch --config_file accelerate_configs/fsdp_8gpu.yaml \
  scripts/train_lora.py \
    --config configs/nuscenes_planning_3cam_qwen3vl_multimodal.yaml \
    $EXTRA \
  2>&1 | tee "$LOG"
RC=$?

echo "[launch] done at $(date -Iseconds) rc=$RC"

if [ "$MODE" != "smoke" ] && [ "$RC" -eq 0 ]; then
  echo "[launch] training complete — ckpts preserved per [[feedback_never_delete_sft_outputs]]"
  ls -lh checkpoints_qwen25/nusc_planning_b5ppp_3cam_qwen3vl_multimodal/ 2>/dev/null
fi
exit $RC
