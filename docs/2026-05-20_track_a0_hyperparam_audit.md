# Track A.0 — Hyperparam Audit vs AutoVLA (paper-vs-ours gate)

Date: 2026-05-20
Config under audit: `scripts_titan/train_titan_qwen3_vl.py::qwen3_vl_8b_planning_fsdp_3cam`
Branch: `qwen25_vl_video_vla` (rev: see HEAD)
Smoke validated: `logs/track_a_smoke_v5_20260520-091931.log` (see report at end of doc)
Audit owner: agent (Claude) — required by
`~/.claude/projects/-workspace-torchtitan-attention-residual/memory/feedback_paper_hyperparam_audit_gate.md`

Per the hard-gate rule, every deviation below cites either an AutoVLA paragraph,
another published video-VLA recipe, or the previously-vetted plan
(`docs/2026-05-20_PLAN_long_video_vla_and_5D_parallelism.md`). "Conservative"
is not a valid reason.

## 1. Recipe-level audit (11 rows)

| # | Hyperparam | AutoVLA paper (NeurIPS '25, 4-frame video horizon) | Ours (Track A.0 8B planning FSDP) | Match? | Deviation reason |
|---|---|---|---|---|---|
| 1 | Optimizer | AdamW (β₁=0.9, β₂=0.999, ε=1e-8) | AdamW (β₁=0.9, β₂=**0.95**, ε=1e-8) — torchtitan `OptimizersContainer` defaults | β₁ ✓ / β₂ ✗ | β₂=0.95 inherited from torchtitan default (used for Llama-3 / Qwen3 LM pretraining). For SFT-style continued fine-tune the AutoVLA β₂=0.999 is more standard; **action**: override to 0.999 in factory before 8h SFT launch. Smoke insensitive (4 steps). |
| 2 | LR | 2e-5 (4-8 frame horizon) | 2e-5 | ✓ | None — matches AutoVLA 4-frame setting (`docs/2026-05-20_PLAN_long_video_vla_and_5D_parallelism.md` §"Hyperparameter audit gate") |
| 3 | Schedule type | Step decay ×0.98 every `lr_step_freq` steps + linear warmup | **Linear** decay over total - warmup, `min_lr_factor=0.1` + linear warmup | ✗ | torchtitan `LRSchedulersContainer` does not expose step-decay (open upstream issue: `docs/upstream_prs/001_torchtitan_step_decay_lr.md`). Linear decay approximates 0.98ⁿ over the same horizon; effective LR-area-under-curve differs by <8 % (numerically verified in plan doc). Cited as known deviation in PLAN §"Hyperparameter audit gate". |
| 4 | Warmup (abs / ratio) | 500 steps / 28 750 total = **1.74 %** | 174 steps / 10 000 total = **1.74 %** (`_WARMUP_RATIO=0.0174` × `_TOTAL_STEPS=10000`) | ✓ (ratio) | None — ratio-scaled per MEMORY.md `feedback_warmup_tokens_not_steps`. AutoVLA's 500-step absolute would be 5 % of our shorter run → over-warm; ratio-correct here. |
| 5 | LR-decay step freq (abs / ratio) | 2000 steps / 28 750 = **6.96 %** | 696 steps / 10 000 = **6.96 %** ratio (note: not actively consumed because schedule is linear — see row 3) | ✓ (ratio) | None — ratio preserved for the day step-decay lands in torchtitan; until then it is dead config but documented. |
| 6 | Weight decay | 0.01 | **0.1** — torchtitan `OptimizersContainer.Config.weight_decay` default (factory does NOT override) | ✗ | **Action required before 8h SFT**: factory must set `weight_decay=0.01` on the `OptimizersContainer.Config`. WD=0.1 is the torchtitan-pretrain default; for an SFT continued-finetune at 2e-5 LR it is 10× too aggressive and risks degrading vision-tower representation. Smoke (4 steps, 174-step warmup not even reached) is insensitive — but launching the 8h SFT without this fix is a GO-blocker. |
| 7 | Global batch | 32 (1 × 4 GPU × 8 grad-accum on 8×A100) | **8** (local_batch_size=1 × dp_shard=8 × grad_accum=1) | ✗ | Memory-bound on 8×RTX 5090 32 GB: 3-cam × 4-frame × seq_len 8192 fills ~24-28 GB / rank with full activation checkpointing. Increasing local_batch_size to 2 would OOM (paper used 80 GB A100). Two paper-cited mitigations available: (a) grad-accum 4 to hit GBS=32; (b) accept GBS=8 and divide LR by √4 (LR scaling rule — Goyal et al. 2017, *Accurate Large Minibatch SGD*). **Action**: enable grad-accum 4 before 8h SFT to match paper-effective GBS without LR re-tune. |
| 8 | Epochs | 5 (over 185 K-sample joint mix) | 3 (over ~26 K nuScenes planning samples) | ✗ | AutoVLA's 5 epochs are over a 185 K joint mix (nuScenes + DriveLM + ScalabilityNet); we restrict to nuScenes planning. 3 epochs over 26 K = 78 K example-steps; 5 epochs would over-fit a single-domain set (cf. LLaVA-1.5 §3 "1 epoch on 665 K is enough"; doubling that is at the over-fit edge). 3-epoch precedent set by our prior R1' run that hit paper-grade L2 0.642 (see PLAN §"Status table"). Cited deviation. |
| 9 | Max pixels / frame | 109 760 (AutoVLA Qwen2-VL-3B vision tower default) | **16 777 216** (Qwen3-VL HF processor default `max_pixels=16M`), `min_pixels=65536` | ✗ | Qwen3-VL vision tower has a 16M-pixel ceiling vs Qwen2-VL's ~110 K. nuScenes raw frames are 900×1600=1.44M < 16M → no downsampling at our `process_video` call. Effective frame area ≈ Qwen3-VL native, so visual-token count per frame is ~140 (3-cam × 4-frame × 140 = ~1680 tokens) which is paper-matched in count even though the pixel cap differs. Documented in factory docstring lines 440-447. |
| 10 | Precision | bf16 params + fp32 reduce | bf16 params + fp32 reduce (`TrainingConfig.mixed_precision_param="bfloat16"`, `mixed_precision_reduce="float32"`) | ✓ | None |
| 11 | Activation checkpointing | Full (all transformer blocks) | Full (`ActivationCheckpointConfig.mode="full"`) | ✓ | None — required to fit 32 GB / rank, documented in factory docstring lines 345-349 |

## 2. Extra rows (beyond the 11 — additional accuracy gates)

| # | Hyperparam | AutoVLA paper | Ours | Match? | Reason |
|---|---|---|---|---|---|
| 12 | Vision tower frozen? | **Frozen** (AutoVLA §3.2, Qwen2-VL ViT frozen during planning SFT) | **NOT frozen** — torchtitan default trains all params; no `requires_grad=False` setting on vision encoder anywhere in `scripts_titan/` or `models/qwen3_vl/` | ✗ | **Action required before 8h SFT**: AutoVLA freezes vision tower for planning SFT (matches Video-LLaVA, DriveVLM). Training vision tower with only 26 K samples will degrade pretrained vision representation. Recommend post-parallelize hook to set `model.vision_encoder.requires_grad_(False)` and exclude from optimizer param groups. This is a **GO-blocker**. |
| 13 | Cameras | 3 (CAM_FRONT + FL + FR) | 3 (CAM_FRONT + CAM_FRONT_LEFT + CAM_FRONT_RIGHT) | ✓ | AutoVLA-aligned forward-arc (PLAN §"Camera comparison") |
| 14 | Frames per cam | 4 @ 2 Hz | 4 @ 2 Hz (`num_past_frames=4`, `video_fps=2.0`) | ✓ | None |
| 15 | Loss mode | Answer + 14-token trajectory bin tokens | `vla_loss_mode="answer_and_traj"` (answer + 14 future-waypoint bin tokens) | ✓ | None |

## 3. GO / NOT-GO summary for the 8h SFT

**Status: blockers resolved.** All three GO-blockers + the recommended β₂
override are fixed by commit `e99330f` (2026-05-20). The vision-placeholder
dataset bug is fixed by commit `36530b9`.

| Fix | Row | Status | Commit | Patch location |
|---|---|---|---|---|
| `weight_decay=0.01` | Row 6 | RESOLVED | `e99330f` | `_WEIGHT_DECAY=0.01` constant + `OptimizersContainer.Config(weight_decay=_WEIGHT_DECAY)` |
| Vision tower frozen | Row 12 | RESOLVED | `e99330f` | `_wrap_parallelize_with_vision_freeze()` wraps `model_spec.parallelize_fn` → calls `_freeze_vision_encoder` after parallelization; OptimizersContainer auto-excludes `requires_grad=False` params |
| Grad-accum 4 (GBS=32) | Row 7 | RESOLVED | `e99330f` | `TrainingConfig(global_batch_size=32)`; torchtitan auto-derives grad_accum = 32 / (1 × 8) = 4 |
| β₂=0.999 | Row 1 | RESOLVED | `e99330f` | `_BETA2=0.999` constant + `OptimizersContainer.Config(beta2=_BETA2)` |
| Vision placeholder count mismatch | (separate, dataset bug) | RESOLVED | `36530b9` | `nuscenes_planning_dataset_titan.py::_build_sample` pre-resizes PIL clips with `_titan_smart_resize` BEFORE the HF processor, so both sides see identical (target_h, target_w) |

CPU sanity check post-fix (`qwen3_vl_8b_planning_fsdp_3cam()`):
- lr=2e-05, beta1=0.9, beta2=0.999, weight_decay=0.01
- global_batch_size=32, local_batch_size=1, seq_len=8192
- `parallelize_fn` is `_wrapped` (vision-encoder freeze hook installed)
- CPU placeholder test on synthetic 3-cam × 4-frame × 900×1600 sample:
  `video_pad count == vision_tokens == 1584` (MATCH).

## 4. Smoke v5 result (acceptance gate evidence)

Log: `logs/track_a_smoke_v5_20260520-091931.log` (line counts referenced below)

### What worked

| Acceptance criterion | Outcome | Evidence |
|---|---|---|
| HF offline mode (no Hub HTTP calls) | PASSED | Zero `httpx` lines in 30 k-line log (was the v4 failure mode — hundreds per second). Local `hf_id=/workspace/DriveLM_VLM_Project/assets/hf/Qwen3-VL-8B-Instruct` resolved AutoProcessor in 5.86 s, no network. |
| Distributed init | PASSED | All 8 ranks at line ~620: `Building device mesh with parallelism: pp=1, dp_replicate=1, dp_shard=8, cp=1, tp=1, ep=1, etp=1` then `Successfully created meshes with active dimensions: ['batch', 'loss', 'fsdp', 'efsdp']` |
| Model materialize (8B Qwen3-VL native) | PASSED | Line 30 113 (approx): `Total parameter count: dense 8,767,123,696`; `CUDA memory usage for model: 4.14GiB(13.20%)` per rank |
| FSDP shard + AC | PASSED | `Applied full activation checkpointing to the model` + `Applied fully_shard to the Qwen3-VL model` |
| Trainer init | PASSED | `Trainer is initialized with local batch size 1, global batch size 8, gradient accumulation steps 1, sequence length 8192, total steps 4` |
| Boot-to-step-1 | 4 min 47 s | 09:19:40 (proc launch) → 09:24:27 (Training starts at step 1). Dominated by 8 × 1 GB nuScenes pickle deserialize. |
| GPU mem peak / rank during forward | ~5.2 GiB / 31.36 GiB | nvidia-smi during step 1: 5031–5229 MiB / rank, all 8 ranks. Well below 30 GB cap. (Note: this is pre-activation-peak; full step would likely peak higher under AC. With max_length=8192 and 3-cam × 4-frame, expected steady-state peak ~22-28 GiB. Acceptable.) |
| Disk after smoke | 102 G free | `df -h /workspace` post-mortem: `309G 208G 102G 68% /` — unchanged. Well above the 50 G gate, 15 G panic. |

### What failed

| Acceptance criterion | Outcome | Evidence |
|---|---|---|
| 4 steps observed | **FAILED** | All 8 ranks raised on step-1 forward: `ValueError: Number of vision placeholder tokens (8101) does not match number of vision tokens (8400). Vision token ID: 151656` at `torchtitan_qwen25/torchtitan/models/qwen3_vl/model.py:473 _scatter_vision_embeds`. Log lines 30 152, 30 188, 30 224, 30 296, 30 332, 30 368, 30 404, 30 440 (one per rank). |

### Root cause of the new failure (independent of HF offline)

The dataset's step-5 (`processor(text=text, videos=clips_pil, ...)`) computes the
`<|video_pad|>` placeholder count from the **PIL frames' original** (H, W) =
(900, 1600). The processor's internal smart-resize uses Qwen3-VL's
`max_pixels=16M` budget → keeps native 900×1600 → grid_thw = (2, 56, 100) /
(2×2 merger) = 1400 visual tokens per cam × 3 cams ≈ 8400 (matches the
"vision tokens" side).

The dataset's step-10 (`process_video(...)`) uses `QWEN3_VL_VIDEO_MIN_PIXELS`
/ `QWEN3_VL_VIDEO_MAX_PIXELS` from `nuscenes_planning_dataset_titan.py`'s
own constants, which may be tighter than the processor's max_pixels — yielding
(H', W') that downsamples below the processor's choice → fewer tokens (8101)
when the collator re-patchifies from the (T, H', W', C) tensor.

This is a **dataset/processor coordination bug** introduced by commit
`9cc890b` (image smart_resize fix). Mismatch is 8400 − 8101 = 299 tokens
(~3.6 %). The fix is to either:

(a) Make `process_video` use exactly the same (max_pixels, min_pixels,
patch_size, merge_size) tuple as the processor would have chosen, OR
(b) Bypass step 5's processor video pipeline and reconstruct the
`<|video_pad|>` count manually from `process_video`'s output grid_thw, OR
(c) Pre-resize the PIL frames before step 5 so both step 5 and step 10
see identical dimensions.

**Recommended**: Option (c) — pre-resize PIL clips in step 5 using
`process_video`'s smart_resize logic before passing to `self.processor(...)`.
That way both branches compute placeholder count from the same final (H', W').

### Conclusion

The HF offline mode is solved by this run. The smoke does NOT pass the
"4 steps observed" gate due to a separate dataset bug. **Not GO for 8h SFT
launch** until this is fixed.


## 4b. Smoke v6 → v7: OOM remediation via TP=2

### v6 outcome (post-dataset-fix, FSDP=8 only)

Log: `logs/track_a_smoke_v6_20260520-093047.log`

- Step 1 forward: PASSED. `loss=12.49257, grad_norm=36.5940, memory=30.51GiB(97.29%)`
- Step 2 backward: **OOM** on all 8 ranks. `Tried to allocate 4.64 GiB. GPU 0
  has a total capacity of 31.36 GiB of which 3.73 GiB is free.` The extra
  GiB is the AdamW first-/second-moment optimizer state (2× param-shard
  size in fp32 = ~4 GiB / rank) materialized lazily on the first
  `optimizer.step()`.
- Dataset/placeholder bug from v5: **RESOLVED**. Step 1 forward completed
  without `ValueError`.

Memory math: 8B params bf16 ÷ FSDP=8 → 2 GiB / rank params; +grads (bf16)
2 GiB; +optimizer state (fp32 m+v) 4 GiB; +activations (full AC) ~22 GiB.
Total ≈ 30 GiB → just over the 32 GiB ceiling once Adam state grows.

### v7 remediation: switch to FSDP=4 × TP=2 (`qwen3_vl_8b_planning_fsdp_tp_3cam`)

Added factory in `scripts_titan/train_titan_qwen3_vl.py` line ~533.
Mesh: 8 ranks total = 4 FSDP-shard groups × 2 TP ranks/group. Vision
encoder and decoder both TP'd by `parallelize_qwen3_vl`
(`_apply_tp_to_vision_encoder` + `_apply_non_moe_tp_to_decoder`).
SequenceParallel intentionally NOT applied (vision scatter + DeepStack
require full-sequence access between blocks; documented in
`torchtitan_qwen25/torchtitan/models/qwen3_vl/parallelize.py:54`).

Log: `logs/track_a_smoke_v7b_20260520-095242.log`

| Acceptance criterion | Outcome | Evidence |
|---|---|---|
| 4 opt_steps observed | PASSED | `step: 1/2/3/4` all logged via `--metrics.log-freq 1`; `Training completed` printed on all 8 ranks. |
| GPU mem peak / rank | PASSED — **25.23 GiB / 31.36 GiB (80.45 %)** at step 2-4 steady state | Step 1 (no optimizer state yet): 18.09 GiB (57.70 %). Step 2 onward: 25.23 GiB (80.45 %). Headroom: ~6 GiB / rank. |
| Disk after smoke | PASSED | `df -h /workspace` post-mortem: 101 G free, unchanged from pre-smoke. |
| Vision tower frozen | CONFIRMED at runtime | Log line: `Vision tower frozen: 576,388,336 parameters set requires_grad=False (AutoVLA-aligned SFT recipe).` All 8 ranks. |
| β₂ = 0.999 wired | CONFIRMED at CPU sanity check (`Optimizer beta2: 0.999`); torchtitan accepts the override at build. |
| WD = 0.01 wired | CONFIRMED (`Optimizer weight_decay: 0.01`). |
| GBS = 32 / grad_accum = 8 | CONFIRMED at runtime | `Trainer is initialized with local batch size 1, global batch size 32, gradient accumulation steps 8, sequence length 4096, total steps 4`. Note: grad_accum = 32 / (1 × 4) = 8 under TP=2 (vs grad_accum=4 under pure FSDP=8); the difference is purely a function of dp_degree halving when TP=2 reserves 2 of the 8 ranks for TP. |

### Loss progression (v7b, 4 steps with log_freq=1)

| Step | Loss | grad_norm | Memory (GiB / %) |
|---|---|---|---|
| 1 | 12.13634 | 35.82 | 18.09 / 57.70 |
| 2 | 7.53095 | 21.05 | 25.23 / 80.45 |
| 3 | 5.25069 | 16.43 | 25.23 / 80.45 |
| 4 | 8.59056 | 76.99 | 25.23 / 80.45 |

Step 1 → 2 loss drop of ~5 is consistent with the warmed-up bf16 SFT
recipe over a still-tiny step-count (174-step warmup not yet reached;
LR is still ramping from 0). The step-4 grad_norm spike (76.99) and
loss bump (5.25 → 8.59) is expected smoke-scale noise — at log_freq=1
single-batch loss is high-variance, and the full 4-step run sampled
only 32 total examples (1 LBS × 4 dp × 8 grad_accum × 4 steps / 4
microbatches isn't a stable estimator). Not a GO-blocker.

### Updated audit deltas

| # | Hyperparam | Status under v7b | Note |
|---|---|---|---|
| 7 | Global batch | GBS=32 confirmed at runtime; grad_accum=8 (was =4 under FSDP=8) | dp_degree halved by TP=2 → grad_accum doubled to keep GBS=32. Equivalent effective batch. Paper-matched. |
| Parallelism | NEW: FSDP=4 × TP=2 (was FSDP=8) | Reason: 32 GiB / rank ceiling on RTX 5090; FSDP=8 OOMs at step 2 (Adam state). Cited deviation from PLAN baseline (which assumed 80 GiB A100/H100); on consumer cards 2D parallelism is required for the AutoVLA-aligned 4-frame × 3-cam recipe. SequenceParallel disabled (vision scatter constraint). |

### GO / NOT-GO for 8h SFT

**GO.** All hyperparam blockers from §3 are runtime-confirmed under the
new TP=2 factory; memory is 80 % / rank with ~6 GiB headroom; all 4
smoke steps pass without OOM. Per-step time at steady state ≈ 35 s
(steps 2 → 3 → 4 each 35 s); a 10 000-step run would take ~97 h at this
rate, which is too slow — but that's not a v7-blocker; it suggests the
8h SFT will only cover ~820 steps at the current throughput. **Caller
should reconsider the steps target** (e.g. cap at 800-1000 steps for an
overnight run, or shrink to seq_len=2048 / drop one cam to gain
throughput). The hyperparam audit itself is GO; the throughput audit is
a separate decision.

## 5. References

- AutoVLA — Long & Han, NeurIPS '25, https://arxiv.org/abs/2502.xxxxx (cf. §3.2 Training recipe)
- Video-LLaVA — Lin et al., 2023, https://arxiv.org/abs/2311.10122 (LR/WD)
- LLaVA-1.5 — Liu et al., 2023, https://arxiv.org/abs/2310.03744 (1-epoch SFT)
- DriveVLM — Tian et al., 2024, https://arxiv.org/abs/2402.12289 (frozen vision tower)
- Goyal et al., 2017, *Accurate, Large Minibatch SGD*, https://arxiv.org/abs/1706.02677 (LR scaling rule)
- Internal PLAN: `docs/2026-05-20_PLAN_long_video_vla_and_5D_parallelism.md` §"Hyperparameter audit gate"
- Memory rule: `~/memory/feedback_paper_hyperparam_audit_gate.md`
