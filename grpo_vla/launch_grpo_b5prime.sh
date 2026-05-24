#!/usr/bin/env bash
# GRPO B.5' launcher -- 8x 5090 Option B (6 actor FSDP + 2 rollout TP=2 SGLang).
# Reads Agent B's config (configs/grpo_b5prime_3cam.yaml), Agent C's reward
# (reward.py:planning_reward) + dataset adapter (dataset_adapter.py), plus
# Agent D-built parquet snapshots under data/.
# Operator MUST run `launch_with_checklist.py` first.
set -euo pipefail
set -x

GRPO_DIR=/workspace/DriveLM_VLM_Project/grpo_vla
RUN_NAME=${RUN_NAME:-grpo_b5prime_3cam_$(date +%Y%m%d-%H%M%S)}
SAVE_ROOT=/workspace/DriveLM_VLM_Project/checkpoints_qwen25/$RUN_NAME
LOG_DIR=$GRPO_DIR/logs/$RUN_NAME
DATA_DIR=$GRPO_DIR/data
CFG=$GRPO_DIR/configs/grpo_b5prime_3cam.yaml
mkdir -p "$SAVE_ROOT" "$LOG_DIR" "$DATA_DIR"

# ------- pre-flight gate -------
DISK_FREE_G=$(df -BG /workspace | awk 'NR==2{gsub("G","",$4); print $4}')
if [ "${DISK_FREE_G:-0}" -lt 15 ]; then
  echo "ABORT: disk free ${DISK_FREE_G}G < 15G (feedback_disk_panic_protocol)" >&2
  exit 2
fi
for f in $CFG $GRPO_DIR/reward.py $GRPO_DIR/dataset_adapter.py; do
  [ -f "$f" ] || { echo "ABORT: missing $f"; exit 2; }
done

# ------- env -------
export PYTHONPATH=$GRPO_DIR:$(dirname $GRPO_DIR):/workspace/verl:${PYTHONPATH:-}
export WANDB_MODE=${WANDB_MODE:-offline}
export VERL_ROLLOUT_SGLANG_URL=http://localhost:30001
export HF_HOME=${HF_HOME:-/workspace/hf_cache}
export RAY_DEDUP_LOGS=0

# ------- step 1: build parquet snapshots if missing -------
# 12K train + 500 val subsample (vs full 24K/5K) — keeps build < 10 min and
# parquet < 5 GB. 12K × 6 epoch = 2000-3000 step training range.
TRAIN_SAMPLES=${TRAIN_SAMPLES:-12000}
VAL_SAMPLES=${VAL_SAMPLES:-500}
N_WORKERS=${BUILD_WORKERS:-8}
for split in train val; do
  pq=$DATA_DIR/nusc_planning_${split}.parquet
  max_samples=$([ "$split" = "train" ] && echo $TRAIN_SAMPLES || echo $VAL_SAMPLES)
  if [ ! -f "$pq" ]; then
    echo "[launcher] building $pq ($max_samples samples, $N_WORKERS workers) ..."
    /usr/bin/python3 $GRPO_DIR/build_parquet.py --config $CFG --split $split \
      --max-samples $max_samples --workers $N_WORKERS --max-edge 448 --jpeg-q 75 \
      2>&1 | tee -a "$LOG_DIR/build_parquet_${split}.log"
  fi
done

# ------- step 2: launch veRL GRPO trainer in background -------
# Overrides forced by Agent D per memory rules:
#  - save_freq=50 / test_freq=50 (~30 min @ ~35s/step ; feedback_ckpt_interval_half_hour)
#  - total_training_steps=500 (per task spec; ~0.5 epoch on 24K data)
#  - max_actor_ckpt_to_keep=2 (feedback_post_train_cleanup_intermediates)
/usr/bin/python3 -m verl.trainer.main_ppo \
    --config-path=$GRPO_DIR/configs \
    --config-name=grpo_b5prime_3cam \
    'hydra.searchpath=[file:///workspace/verl/verl/trainer/config]' \
    trainer.experiment_name=$RUN_NAME \
    trainer.default_local_dir=$SAVE_ROOT \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_training_steps=${TOTAL_STEPS:-1500} \
    trainer.max_actor_ckpt_to_keep=${KEEP_CKPTS:-1} \
    actor_rollout_ref.actor.checkpoint.save_contents='[model]' \
    actor_rollout_ref.actor.checkpoint.load_contents='[model]' \
  2>&1 | tee "$LOG_DIR/train.log" &
TRAIN_PID=$!
echo "TRAIN_PID=$TRAIN_PID  RUN_NAME=$RUN_NAME  SAVE_ROOT=$SAVE_ROOT" | tee "$LOG_DIR/run.meta"

# ------- step 3: mid-training val watcher (eval every 50 steps) -------
( set +x
  last=-1
  while kill -0 $TRAIN_PID 2>/dev/null; do
    latest=$(ls -d $SAVE_ROOT/global_step_* 2>/dev/null | sort -V | tail -n1 || true)
    if [ -n "$latest" ]; then
      step=$(basename "$latest" | sed 's/global_step_//')
      if [ "$step" != "$last" ] && [ $((step % 50)) -eq 0 ]; then
        /usr/bin/python3 $GRPO_DIR/eval_during_training.py \
          --step "$step" --ckpt "$latest/actor" \
          --n-samples 200 --sglang-url http://localhost:30001 --config $CFG \
          >> "$LOG_DIR/eval.log" 2>&1 || echo "eval step=$step crashed" >> "$LOG_DIR/eval.log"
        last=$step
      fi
    fi
    sleep 60
  done
) &
EVAL_PID=$!
echo "EVAL_PID=$EVAL_PID" | tee -a "$LOG_DIR/run.meta"

# ------- step 4: wait + post-train cleanup (feedback_post_train_cleanup_intermediates) -------
wait $TRAIN_PID
TRAIN_RC=$?
kill $EVAL_PID 2>/dev/null || true
echo "TRAIN exit=$TRAIN_RC" | tee -a "$LOG_DIR/run.meta"

if [ "$TRAIN_RC" -eq 0 ]; then
  keep=$(ls -d $SAVE_ROOT/global_step_* 2>/dev/null | sort -V | tail -n 2 | xargs -n1 basename || true)
  for d in $SAVE_ROOT/global_step_*; do
    name=$(basename "$d")
    if ! echo "$keep" | grep -qx "$name"; then
      echo "rm intermediate $d" >> "$LOG_DIR/cleanup.log"
      rm -rf "$d"
    fi
  done
  final_src=$(ls -d $SAVE_ROOT/global_step_* 2>/dev/null | sort -V | tail -n1)
  [ -n "$final_src" ] && ln -sfn "$final_src" "$SAVE_ROOT/final"
fi
df -h /workspace | tee -a "$LOG_DIR/run.meta"
exit $TRAIN_RC
