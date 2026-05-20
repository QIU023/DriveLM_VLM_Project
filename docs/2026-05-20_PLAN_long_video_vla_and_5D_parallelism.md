# 2026-05-20 — Long-Video VLA + 5D Parallelism Master Plan

> Living doc. Last updated 2026-05-20 ~07:00Z. Snapshots state and direction.

## TL;DR

R1' (1-cam, 4-frame, ego speed, tight bins) is the paper-grade baseline:
**L2 avg 0.642 m / collision 3.73 %** on nuScenes-planning val 5119. Matches
single-cam baselines (BEV-Planner-front 0.65, AD-MLP 0.70). Eval pipeline now
multi-GPU DP-batched (45 min → 3 min, 18.5× faster).

Next steps split into two parallel tracks:

1. **Track L (Long-video compression ablation)** — finish ablation matrix
   {mean-pool, VTM, LongVU} × {8 frames} on R1' baseline. Drop 16 / 32-frame
   single-cam ablation as not paper-realistic for our scale.
2. **Track P (5D Parallelism on Qwen2.5-VL)** — adapt torchtitan's `qwen3_vl`
   → Qwen2.5-VL via subclass, add PP + CP support (PR-able to torchtitan
   upstream). 3 sub-agents working in parallel on CPU-only code; smoke later
   in authorised 10-20 min GPU windows between ablation configs.

After both tracks converge, do **"production-scale" AD VLA training**:
**3-cam × 16-frame uncompressed via CP** — matches rumored XPeng/Tesla
production VLA visual scale, infeasible without CP due to attention memory.

## Why dropped 32-frame single-cam ablation

Survey of AD-VLA paper frame counts:

| Paper | Cameras | Frames | Notes |
|---|---|---|---|
| AutoVLA (NeurIPS'25) | 3 | 4 @ 2Hz | Reference |
| DriveVLM | 8 multi-cam | 8 | |
| EMMA (Waymo) | multi-cam | 1 | Single keyframe |
| OpenEMMA | 1 | 4 | |
| Impromptu VLA | 1 | 4 | |
| UniAD / VAD | 6 surround | 3 BEV-stack | Not visual tokens |
| BEV-Planner-front | 1 | BEV-stack | |
| CoVLA | 1 | 60 waypoints (action, not visual) | |

**No production AD-VLA paper uses 32+ frames per camera.** Compute and
information-theoretic reasons: 3-s planning horizon doesn't benefit from
beyond ~3s past context. Long-video VLM (LLaVA-Video 32-64 frames, LongVU
128-512 frames) is general-video QA territory, not AD-planning.

We DO want CP infra for resume value (advanced parallelism), but the right
demo target is multi-cam (3-cam × 8-16 frames), NOT single-cam × 128 frames.

## State as of 2026-05-20 07:00Z

### Completed

| Run | Config | L2 avg TemAvg | Collision avg | Notes |
|---|---|---|---|---|
| **R1** | 1-cam 4-frame, no ego speed, bins [-50, 50], warmup 500 | 2.13 m | 24.7 % | Broken — modality + tokenizer + warmup ratio + collision-check all off |
| **R1'** | 1-cam 4-frame, +ego speed, bins [-2,40]/[-8,8], max_len 4096, UniAD-port collision | **0.642 m** | **3.73 %** | **Paper-grade single-cam baseline** |
| **R3 (8f-meanpool v1)** | 1-cam 8-frame mean-pool, warmup 500 (over-warmuped) | — | — | Not finalised; superseded by v2 |
| **8f-meanpool v2** | 1-cam 8-frame mean-pool, **warmup 39, lr_step_freq 156** (ratio-scaled) | 0.653 m | 3.97 % | ≈ R1' baseline — naive temporal pooling does not help over 4-frame |

### Running

| Task | Status |
|---|---|
| **8f-vtm** SFT (PID 1075320) | training, ETA ~2.5 h |
| **Agent A**: Qwen2.5-VL torchtitan subclass + nuScenes adapter | CPU work, ~3 h budget |
| **Agent B**: PP for `torchtitan/models/qwen3_vl/` | CPU work, ~4 h budget |
| **Agent C**: CP for `torchtitan/models/qwen3_vl/` | CPU work, ~5 h budget |

### Queued

- **8f-longvu** SFT + eval
- 5D smoke matrix (Agent D, after A/B/C done): FSDP=8 / FSDP=2×PP=2 / CP=4×FSDP=2 / PP=2×CP=2×FSDP=2 / PP=2×TP=4
- **3-cam × 16-frame uncompressed with CP** — production-scale AD VLA demo

### Dropped from earlier plan

- 16-frame and 32-frame **single-cam** ablation — not paper-realistic
- R1' v2 (warmup-corrected 4-frame retrain) — empirical R1' was already
  paper-grade despite over-warmup; not worth 2.5 h GPU to repeat for
  marginal improvement
- R1'' 3-cam SFT *under FSDP-only* — was OOM-bound; will be unlocked by CP
- TP for 3B Qwen2.5-VL — unnecessary (single layer fits)
- PP for 3B Qwen2.5-VL — unnecessary; PP value-add is at 8B+ or multi-node.
  Agent B work remains valuable as PR-able torchtitan upstream contribution
  but won't be used in our specific run

## Architecture decisions

### Why CP and not PP for our 3-cam target

- **Param memory** for Qwen2.5-VL-3B is small: ~12 GB even unsharded; FSDP=8
  drops to ~1.5 GB / rank. **PP doesn't help with param memory we don't
  have.**
- **Activation memory** scales with sequence length. 3-cam × 16-frame ≈
  6720 visual tokens → ~7 K LM sequence. Attention activation O(N²) is the
  bottleneck. **CP halves this per rank.**

### Why subclass torchtitan `Qwen3VLModel` for Qwen2.5-VL

- Qwen3-VL is ~95 % architecturally identical to Qwen2.5-VL. Differences:
  - QK-norm in Qwen3 (Qwen2.5 doesn't have it) — disable in subclass
  - DeepStack injection (intermediate ViT features into early decoder) in
    Qwen3 — disable in subclass
  - MRoPE interleaving in Qwen3 — fall back to plain 3D-RoPE
- Subclass keeps torchtitan core clean (no qwen2_5_vl/ fork) and lets us PR
  PP/CP additions to qwen3_vl upstream.

### Why drop sglang/vllm for eval-DP

- Pure torchrun + `transformers.generate` with bs=4 hit 18.5× speedup, well
  inside the <5 min target. Adding sglang/vllm would be 3× more infra to set
  up for ~2× more speedup. Not worth it.

## Hyperparameter audit gate (2026-05-20 status)

All ablation configs share R1' AutoVLA-derived recipe:

| Knob | Value | Source |
|---|---|---|
| Optimizer | AdamW | AutoVLA |
| LR (4-8 frames) | 2e-5 | AutoVLA / Video-LLaVA |
| LR (16-32 frames) | 1e-5 | LLaVA-NeXT-Video / LLaVA-Video / Apollo |
| Schedule | step-decay ×0.98 / lr_step_freq | AutoVLA |
| **warmup_steps** | **39** (ratio 1.74 % of 2243 total) | AutoVLA's 500 / 28750 ratio applied to our scale |
| **lr_step_freq** | **156** (ratio 6.96 %) | Same ratio derivation |
| Weight decay | 0.01 | AutoVLA |
| Global batch | 32 (1 × 4 × 8) | AutoVLA |
| Epochs | 3 | Smaller dataset than AutoVLA's 5-epoch 185 K mix |
| Max pixels / frame | 109760 fixed | AutoVLA |
| Precision | bf16 | All |
| Activation checkpointing | on | All |
| Vision tower | frozen | All |

**Rule going forward (saved as `feedback_paper_hyperparam_audit_gate` memory expansion 2)**:
when copying an absolute step count from paper, compute the ratio against
paper's total steps and apply that ratio to OUR total. Do not copy absolutes.

## What we'll showcase for XPeng-class JD

**Mid-level XPeng JD focus weights (estimate):**
1. Distributed training (~50 %)
2. Visual / video VLM (~25 %)
3. RLHF / GRPO (~15 %)
4. Deployment (~10 %)

**Coverage**:

- **Distributed training**:
  - phase3 attn_res: PP adapter for Kimi-Linear in torchtitan (real
    implementation experience, not framework calls)
  - This project: FSDP+AC end-to-end Qwen2.5-VL 3B; bugs caught
    (AcceleratedScheduler 8× LR multiplier, frozen-vision-encoder + FSDP
    re-freeze, FSDP-safe xframe compressor monkey-patch)
  - This project: PP + CP for `torchtitan/models/qwen3_vl/` (PR-able to
    Meta upstream)
  - Eval-side DP: multi-GPU DP + batched generate, 18.5× speedup
- **Video / multimodal VLM**:
  - This project: nuScenes planning VLA with paper-grade L2 0.642 / coll
    3.73 %; modality fix narrative (ego speed, tight bins, max_length,
    UniAD-port collision)
  - Cross-frame compression ablation (mean-pool, VTM, LongVU)
- **RLHF**:
  - phase11: GRPO infra with SGLang rollout + torchstore + Monarch fixes
- **Deployment**:
  - phase10: DCP → HF safetensors conversion
  - Open: TensorRT / vLLM inference latency (not yet done)

## Workflow rules

- **Compression ablation continues uninterrupted** unless a torchtitan agent
  finishes their core code and needs a 10–20 min GPU window for smoke
  validation. In that case: authorised pause; if smoke uncovers blocker not
  fixable in the window, drop GPU and continue ablation, debug agent-side;
  if smoke passes, commit + push and resume ablation.
- **After all compression ablation done**, gate moment: full machine takeover
  by Track P. Run 3-cam × 16-frame with CP=2 + FSDP=4 as the headline 5D
  parallel result.
- **Document in this file** every checkpoint deliverable and decision change.

## Pointers

- DriveLM_VLM_Project main code: `scripts/`
- Compression compressors: `scripts/compressors/`
- Configs: `configs/nuscenes_planning_{4f,8f,16f}_*.yaml`
- Eval (now multi-GPU): `scripts/planning_eval.py` + `scripts/launch_planning_eval_dp.sh`
- torchtitan fork (PP/CP work): `torchtitan_qwen25/` (branch `qwen25_vl_video_vla`)
- torchtitan parallelism agents (will commit there): Agent A subclass /
  Agent B PP / Agent C CP
- phase12 (AD-VLA research): `/workspace/torchtitan_attention_residual/phase12_ad_vla_research/`
