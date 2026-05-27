#!/usr/bin/env bash
# 1-cam cached-vs-live A/B, grad-ckpt OFF (--no-grad-ckpt) — IDENTICAL config
# both arms (1cam NATIVE, no AC, LBS=2), 500 steps, no-save. This is the
# un-diluted ViT-skip test: with no recompute the LM forward isn't doubled, so
# the cached arm's per-step win shows in full (vs the 3-cam AC-on 1.04x).
# Stall-detector py-spy-dumps + kills if a step stalls >120s; disk<8G kill guard.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export HF_HOME=/workspace/.hf_home; unset HF_HUB_OFFLINE || true
CFG=configs/nuscenes_planning_1cam_qwen3vl_NATIVE.yaml
CACHE=/workspace/DriveLM_VLM_Project/data_infra/lance/nusc_1cam_native.lance
ACC="accelerate launch --config_file accelerate_configs/fsdp_8gpu.yaml scripts/train_lora.py"
COMMON="--config $CFG --save-every 999999 --val-every 999999 --no-final-save --no-grad-ckpt --epochs 99 --max-steps 500"

stall_watch() {  # $1=logfile $2=tag
  local last=-1 lastt=$(date +%s)
  while true; do
    sleep 20
    cur=$(grep -aoE "opt_step=[0-9]+" "$1" 2>/dev/null | tail -1 | grep -oE "[0-9]+")
    now=$(date +%s)
    if [ -n "$cur" ] && [ "$cur" != "$last" ]; then last=$cur; lastt=$now; fi
    if [ $((now-lastt)) -gt 120 ]; then
      echo "[STALL $2 @step $last] py-spy dump + kill"
      for p in $(pgrep -f train_lora.py); do py-spy dump --pid $p >> /tmp/ab1_stall_$2.stacks 2>&1; done
      pkill -9 -f train_lora.py; return 1
    fi
    fr=$(df -BG /workspace|awk "NR==2{gsub(\"G\",\"\",\$4);print \$4}")
    if [ "$fr" -lt 8 ]; then echo "[DISK<8G $2] 停训练保平台"; pkill -9 -f train_lora.py; return 2; fi
    grep -aq "opt_step=499\|Training complete" "$1" 2>/dev/null && return 0
    pgrep -f train_lora.py >/dev/null || return 0
  done
}

echo "===== ARM A: LIVE (ViT every step), 1cam NATIVE no-AC, 500 steps ====="
A0=$(date +%s)
$ACC $COMMON --compress-method fastervlm --compress-ratio 4 --experiment ab1_live_DELETEME > /tmp/ab1_live.log 2>&1 &
stall_watch /tmp/ab1_live.log live; wait
A1=$(date +%s); echo "LIVE wall=$((A1-A0))s"

echo "===== ARM B: CACHED (no ViT), 1cam NATIVE no-AC, 500 steps ====="
B0=$(date +%s)
$ACC $COMMON --cached-vision-lance "$CACHE" --experiment ab1_cached_DELETEME > /tmp/ab1_cached.log 2>&1 &
stall_watch /tmp/ab1_cached.log cached; wait
B1=$(date +%s); echo "CACHED wall=$((B1-B0))s"

echo "===== 稳态 s/it 中位 ====="
for tag in live cached; do
  med=$(grep -aoE "[0-9.]+s/it" /tmp/ab1_$tag.log 2>/dev/null | tail -400 | sort -n | awk '{a[NR]=$1}END{print a[int(NR/2)]}')
  echo "$tag median s/it = $med"
done
echo "ALL_AB1_DONE"
