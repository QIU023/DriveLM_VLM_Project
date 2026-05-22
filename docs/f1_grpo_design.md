# F.1 — GRPO via veRL: design + adaptation to local 8× 5090 setup

Status: paper-only design. NO install / NO env change while 8f_longvu is running.
Decision deferred until after B.5 + B.6 finish. Two execution paths:
- **Path B (recommended)**: copy B.5 ckpt to H100/A100 cluster with stock veRL env
- **Path C (fallback)**: implement minimal GRPO in our train_lora.py (no vllm, HF generate rollout)

---

## 1. veRL GRPO reference recipe (extracted from `examples/grpo_trainer/run_qwen2_5_vl_7b_fsdp.sh`)

| Parameter | veRL Qwen2.5-VL-7B default | Why |
|---|---|---|
| MODEL_PATH | Qwen/Qwen2.5-VL-7B-Instruct | base policy |
| TRAIN_BATCH_SIZE | 512 prompts | global GRPO batch |
| PPO mini-batch | 128 | micro-step within global |
| Actor LR | **1e-6** | GRPO LR is 10-100× smaller than SFT |
| KL coef (β) | 0.01 | reference-policy constraint |
| Rollout n | 5 | candidates per prompt |
| TP (rollout) | 2 | vllm engine tensor parallel |
| Strategy | fsdp2 | actor strategy |
| Max prompt | 1024 | input cap |
| Max response | 2048 | output cap |
| Epochs | 15 | full RL |
| Save freq | 20 steps | ckpt |
| Test freq | 5 steps | mid-training eval |
| GPU mem util (rollout) | 0.6 | vllm engine portion |
| PPO max tokens/GPU | 24576 | activation cap |

---

## 2. Adaptation to our setup (8× 5090 32GB)

### 2.1 Resource layout

```
8 × RTX 5090 32GB

OPTION A (vllm rollout co-located with actor — TIGHT):
  All 8 GPU: actor FSDP=8 + vllm engine TP=2 timeshared with offload
  → vllm GPU mem util 0.3 (leaves room for actor activations)
  → likely OOM on 32GB; needs aggressive offload

OPTION B (split):
  6 GPU: actor FSDP=6
  2 GPU: vllm engine TP=2 (dedicated)
  → ~10 GB per rollout rank (3B + KV cache) → fits 32GB easily
  → actor FSDP=6 → 3B/6 = 0.5GB params/rank, fits

RECOMMENDED: OPTION B (safer on 32GB)
```

### 2.2 Hyperparams scaled to our scale

| Parameter | veRL 7B default | Our 3B / 5090 |
|---|---|---|
| MODEL_PATH | Qwen2.5-VL-7B-Instruct | **B.5 final ckpt** (not raw HF) |
| TRAIN_BATCH_SIZE | 512 | **24** (match our SFT GBS) |
| PPO mini-batch | 128 | **8** |
| Actor LR | 1e-6 | **1e-6** (unchanged) |
| KL coef β | 0.01 | **0.01** (unchanged; reference = B.5 SFT ckpt) |
| Rollout n | 5 | **8** (slightly bigger group for stability) |
| TP (rollout) | 2 | **2** (option B) |
| Strategy | fsdp2 | **fsdp** (or fsdp2 if veRL supports torch 2.11) |
| Max prompt | 1024 | **2400** (our seq len incl. video tokens) |
| Max response | 2048 | **50** (trajectory tokens only) |
| Epochs | 15 | **1** (24K data, single pass first) |
| Save freq | 20 step | **20** |
| Test freq | 5 step | **5** |
| GPU mem util (rollout) | 0.6 | **0.5** (32GB tighter) |
| PPO max tokens/GPU | 24576 | **16384** |

**3 hyperparams that NEED paper justification before final launch**:
- TRAIN_BATCH_SIZE 24 (vs paper 512) — local compute constraint, doc this clearly
- Rollout n=8 (vs paper 5) — DeepSeek R1-Zero default; more group samples reduce variance
- Epoch=1 (vs paper 15) — 24K data scale, one epoch matches our SFT budget

---

## 3. Reward function design

### 3.1 Signature

```python
def planning_reward(
    predicted_tokens: List[int],          # GRPO-generated trajectory tokens
    ground_truth_waypoints: np.ndarray,   # (6, 2) GT trajectory
    ego_speed_mps: float,                 # CAN bus speed at sample time
    ego_box_xy_dims: Tuple[float, float], # 4.084 × 1.85 (UniAD)
) -> float:
    """Returns a scalar reward in roughly [-2, +1] range."""
    ...
```

### 3.2 Components

```python
# 1. Decode tokens → predicted waypoints (OpenVLA-bin)
pred_wp = openvla_bin_decode(predicted_tokens)  # (6, 2)

# Handle malformed output (wrong token count, out-of-bin, etc.)
if pred_wp is None:
    return -2.0  # max penalty, signals "complete failure"

# 2. L2 to GT (TemAvg protocol, paper-aligned)
l2_per_step = np.linalg.norm(pred_wp - gt_wp, axis=-1)  # (6,)
l2_avg = l2_per_step.mean()

# 3. Collision rate (UniAD-port: ego box 4.084×1.85 + 0.5m forward shift)
collision_count = uniad_collision_check(pred_wp, sample_token)
collision_rate = collision_count / 6.0  # in [0, 1]

# 4. Progress shaping (anti reward-hacking: prevent "predict zeros" attack)
# If GT speed > 1 m/s, penalize predicting near-zero motion
pred_avg_speed = np.linalg.norm(pred_wp[-1] - pred_wp[0]) / (3.0)  # m/s
gt_avg_speed = ego_speed_mps  # from CAN bus
if gt_avg_speed > 1.0:
    speed_ratio = pred_avg_speed / max(gt_avg_speed, 0.5)
    # Reward when ratio ∈ [0.5, 1.5]; penalize stationary predictions
    progress_bonus = 0.1 if 0.5 <= speed_ratio <= 1.5 else -0.2
else:
    progress_bonus = 0.0  # ego is stationary, no constraint

# 5. Compose
reward = -l2_avg - 0.5 * collision_rate + progress_bonus
return reward
```

### 3.3 Reward magnitude analysis

| Outcome | L2_avg | collision | progress | reward |
|---|---|---|---|---|
| Perfect prediction | 0.0 | 0 | +0.1 | **+0.1** |
| R1' baseline (L2=0.64, coll=3.7%) | 0.64 | 0.037 | +0.1 | -0.56 |
| Bad prediction (L2=2.0, coll=20%) | 2.0 | 0.2 | +0.1 | -2.0 |
| Stationary attack (L2=ego_speed×3, coll=0) | ~3 | 0 | -0.2 | -3.2 |
| Malformed output | -2.0 (fixed) | — | — | -2.0 |

Group-relative GRPO normalizes per-batch, so absolute magnitudes are less critical than monotonicity. Above design keeps clean signal: lower L2 → higher reward, collision is penalized, "predict zeros" cheat is worse than R1' baseline.

---

## 4. Data format conversion

veRL expects parquet with prompt + image + label columns:

```python
# Convert our PlanningDataset → veRL parquet
# Required columns:
#   - prompt: str (user text)
#   - images: List[bytes] (PNG-encoded image bytes per sample)
#   - extra_info: dict (sample_token, gt_waypoints, ego_speed for reward fn lookup)

# Note: veRL serializes videos as a list of image frames. For our 4-frame
# camera video, that's 4 PIL.Image → 4 bytes blobs.
# HD map BEV (B.5) = +1 image. So images per sample = 5 (4 cam + 1 hdmap).
```

Estimated conversion time: 1-2h script + 30min run on 24K samples.

---

## 5. Open issues to resolve before launch

| Issue | Status |
|---|---|
| veRL torch 2.11 / transformers 5.6 compat | UNKNOWN — test on path B/H100 first |
| 5090 sm_120 vllm engine compat | KNOWN BROKEN (vllm #13306) — confirms Path B over A |
| Trajectory token decoder integration (`openvla_bin_decode`) | exists in `scripts/trajectory_tokenizer.py` — confirm callable from veRL reward fn |
| UniAD-port collision check call site | exists in `scripts/_planning_metric.py` — confirm callable from veRL reward fn |
| Reward fn must be importable from veRL process | veRL spawns Ray workers; reward fn needs `register_custom_reward` style |
| Reference policy init | use B.5 final ckpt (SFT trained) — same model as actor at step 0 |
| KL schedule | start β=0.01, anneal to 0.001 over training (DeepSeek R1 recipe) |

---

## 6. Launch checklist (when going live)

Path B (H100/A100 cluster):
- [ ] Copy `checkpoints_qwen25/nusc_planning_b5_multimodal/final` to remote
- [ ] Copy `data/` (HD map cache + bbox jsonl + infos pkl) to remote
- [ ] `pip install verl[vllm]` on remote (stable env: torch 2.5, transformers 4.46)
- [ ] Build parquet conversion script (24K samples × 5 images each ≈ 8 GB)
- [ ] Write our `planning_reward.py` based on §3 above; verify imports
- [ ] Smoke run with 10 samples × n=2 rollouts × 5 steps (1 GPU, 30 min)
- [ ] Audit table for hyperparam deviations from veRL default
- [ ] Pre-launch checklist gate (per `feedback_pre_launch_checklist`)
- [ ] Launch full GRPO 1 epoch; estimated 15h on 8× H100

Path C (local 5090, minimal GRPO in train_lora.py):
- [ ] Extend `train_lora.py` with a GRPO mode: rollout (HF generate K=4 candidates) + reward scoring + policy gradient step
- [ ] No KL constraint (R1-Zero style) initially; add later if mode collapse
- [ ] Rollout via existing model (no vllm) — slower, ~5× generate latency
- [ ] Estimated 24-36h on 8× 5090

---

## 7. Out of scope for F.1

- World-model simulator (X-World style) — needs separate video-diffusion infra
- Online interaction with simulator — pure offline data + reward fn here
- Multi-modal reward (currently single L2+collision+progress scalar)
- Vision-side RL (we freeze vision encoder per veRL default)
