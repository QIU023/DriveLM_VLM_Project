# grpo_vla — GRPO training orchestrator for B.5' (Qwen2.5-VL-3B, 3 cam + HD-map)

Glue + launcher + mid-training eval for veRL-driven GRPO of the planning VLA.
Pieces by agent:

| Owner | File | Role |
|---|---|---|
| Agent A | (external) `http://localhost:30001` | standalone SGLang server (eval + smoke only) |
| Agent B | `configs/grpo_b5prime_3cam.yaml` | veRL Hydra config (paper-audit table inline) |
| Agent C | `reward.py`, `dataset_adapter.py`, `test_reward.py`, `smoke_rollout_step.py` | 5-dim reward, val/train torch Dataset, unit + veRL-stack smokes |
| Agent D | `launch_with_checklist.py`, `launch_grpo_b5prime.sh`, `eval_during_training.py`, `smoke/e2e_save_load_smoke.py`, `build_parquet.py` | this README |

## Run order (operator)

```bash
cd /workspace/DriveLM_VLM_Project/grpo_vla
# 1. Pre-launch gate (disk / GPU / sglang / ckpt / config / 3 smokes / ETA)
/usr/bin/python3 launch_with_checklist.py
# Output ends with either "RESULT: GO" or "RESULT: NO-GO -- [...]".
# Do NOT skip a NO-GO; surface the failure (memory: feedback_pre_launch_checklist).

# 2. If GO: launch training (auto-builds parquets first, ~30 min one-time)
nohup ./launch_grpo_b5prime.sh > logs/launch.out 2>&1 &
echo "TRAIN_PID=$!"

# 3. Watch
tail -f logs/grpo_b5prime_3cam_*/train.log
tail -f logs/grpo_b5prime_3cam_*/eval.log
```

## What each script does

- `launch_with_checklist.py` — disk/GPU/sglang/ckpt/config checks, audit table
  print, 3 smokes (reward unit test, save-load parity, veRL stack 1-step),
  ETA estimate. Exits 0 = GO, 1 = NO-GO.
- `launch_grpo_b5prime.sh` — (a) build train/val parquet if missing, (b) start
  `verl.trainer.main_ppo` with config + 4 CLI overrides
  (`save_freq=50 / test_freq=50 / total_training_steps=500 /
  max_actor_ckpt_to_keep=2`), (c) background watcher runs
  `eval_during_training.py` every 50 steps on the latest ckpt,
  (d) post-train cleanup deletes all but the last 2 intermediates + final.
- `eval_during_training.py` — pulls 200 val samples via the adapter, queries
  SGLang `/v1/chat/completions`, tokenizes responses to bin ids, scores with
  `reward.compute_reward`, appends to `logs/eval_curve.jsonl` + tensorboard.
- `build_parquet.py` — streams `VeRLNuScenesDataset` -> parquet shards for
  veRL's default `RLHFDataset` reader. Idempotent (skips if file exists).
- `smoke/e2e_save_load_smoke.py` — load base ckpt, forward, `save_pretrained`,
  reload, forward identical batch, assert `max|logits|` delta < 1e-3.

## Expected wall-clock

| Phase | Time |
|---|---|
| Pre-launch checklist (3 smokes + checks) | ~10 min |
| Parquet build (train 24K + val 5119) | ~30 min one-time |
| Training 500 steps @ ~35 s/step | ~5.0 h |
| 10 mid-train evals × ~1 min each | ~10 min |
| Post-train cleanup | ~1 min |
| **TOTAL (overnight target 6-10h)** | **~5-6 h** |

## Kill instructions

```bash
# Soft (lets ckpt finish):
kill $(cat logs/grpo_b5prime_3cam_*/run.meta | grep TRAIN_PID | tail -1 | cut -d= -f2)

# Hard (kills veRL + Ray workers immediately):
pkill -f "verl.trainer.main_ppo"
ray stop --force 2>/dev/null || true

# Clean partial run:
rm -rf /workspace/.../checkpoints_qwen25/grpo_b5prime_3cam_<timestamp>
```

## Disk + memory budget (per memory rules)

- Disk panic threshold: 15 G free at `/workspace`. Launcher aborts if breached.
- Ckpt every 50 steps × 18 GB × keep_last_2 = 36 GB peak under run dir; +1 final.
- Smoke resave dir auto-cleaned by `e2e_save_load_smoke.py` after PASS.
