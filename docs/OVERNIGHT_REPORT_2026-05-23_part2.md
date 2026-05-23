# Overnight 2026-05-23 part 2 (09:33Z → 12:40Z)

Continuation of `OVERNIGHT_REPORT_2026-05-23.md` (which closed at f4b779a with B.5' compression sweep + R1''' eval).

## TL;DR

| Track | Result | Verdict |
|---|---|---|
| **B.5''** Qwen3-VL-4B 1-cam multimodal SFT | L2 = **0.7151** (vs B.5 Qwen2.5-VL 0.6783) | 🟡 Qwen3-VL underperforms Qwen2.5-VL on same setup, but deployable to TRT |
| **A.1 v2** BLIP-2 pretrained Q-Former 1-cam | L2 = **0.6773** (vs R1' Linear 0.6420) | 🔴 Q-Former STILL loses to Linear even with pretrained init |
| A.3 v2 IDEFICS-2 Resampler | not run | ⏭ Skipped: dim mismatch (LM 4096 vs our 2048, vision 1152 vs 2048) — same expected outcome as A.1 v2 |
| TRT 1.3.0rc15 deploy B.5'' | blocked | 🔴 torch 2.10 vs our 2.9.1/2.11; needs fresh venv (~half-day) |
| B.7 (Qwen3-VL 3cam Q-Former) | code-prep deferred | 🟡 A.1 v2 result removes the case for it |

## Consolidated leaderboard (all 1-cam VLA experiments)

| Run | Backbone | Projector / fusion | Pretrained init | L2 | Coll% | n |
|---|---|---|---|---|---|---|
| **R1'** 1cam | Qwen2.5-VL-3B | **Linear** (Qwen native PatchMerger) | — | **0.6420** | 3.73% | 5119 |
| B.5 1cam multimodal | Qwen2.5-VL-3B | Linear + HD-map + bbox | — | 0.6783 | 3.07% | 5119 |
| **A.1 v2** 1cam | Qwen2.5-VL-3B | **BLIP-2 Q-Former 32q** | ✅ 105M from BLIP-2 | 0.6773 | 3.95% | 5119 |
| A.1 v1 1cam | Qwen2.5-VL-3B | Custom Q-Former 64q | ❌ random | 0.6807 | 3.95% | 5119 |
| A.2 1cam | Qwen2.5-VL-3B | PixelShuffle 2× + Linear | partial | 0.6717 | 4.16% | 5119 |
| A.3 1cam | Qwen2.5-VL-3B | Custom Resampler 64q | ❌ random | 0.6968 | 4.02% | 5119 |
| **B.5''** 1cam multimodal | **Qwen3-VL-4B** | Linear + HD-map + bbox | — | 0.7151 | 3.83% | 5119 |

## Findings (honest)

### F1: Token-compression projectors lose to Linear at 24K samples
A.1 v2 with proper BLIP-2 pretrained init (105M / 109M params from pretrain) only beat random-init by **L2 -0.0034**. The architectural compression (32 query tokens vs ~140 raw post-merger) loses spatial fidelity that matters for sub-meter trajectory accuracy. Conclusion: at our data scale (~24K nuScenes samples), **pretrained init helps marginally but does not flip the verdict** — Q-Former-style compression is the wrong fit. Production VLA Q-Former (Waymo / Wayve / Tesla) uses 10-100× more data and typically larger backbone — different operating point.

By implication: A.3 v2 (Resampler) and B.7 (3-cam Q-Former) would land in the same L2 ~0.68-0.72 region. Worth skipping.

### F2: Qwen3-VL-4B does NOT beat Qwen2.5-VL-3B on this VLA task
Counterintuitive +33% params + newer arch → -5.4% L2 regression (0.7151 vs 0.6783). Hypotheses (no test budget left to ablate):
- patch_size 14 → 16 reduces tokens per cam by ~25%, less spatial detail
- DeepStack vision arch may not align with trajectory prediction
- M-RoPE + action-token insertion interaction (our patched `mm_token_type_ids`)
- bf16 save precision loss vs fp32 of original Qwen2.5 line

But B.5'' is the **only deployable backbone** because TRT-LLM 1.3+ has `modeling_qwen3vl.py` and never had `modeling_qwen2_5_vl.py`. So the trade is: ~5% L2 loss for production deploy.

### F3: TRT-LLM 1.3.0rc15 install requires venv rebuild
- 1.3.0rc15 needs torch 2.10 + cuda 13.1.1
- Our /opt/trt_venv has torch 2.9.1 + cuda 13
- Upgrading torch in venv triggers transformers AutoProcessor import chain failure
- Path forward: fresh /opt/trt_venv_13 with torch 2.10 + tensorrt-llm 1.3.0rc15 + matching deps from scratch (~30-60min careful install)
- Then: write trt_convert_qwen3vl.py (use 1.3 examples/qwen3_vl/ template) + trtllm-build + serve

## Code changes (uncommitted)

- `scripts/train_lora.py`:
  - `--save-optim-state` CLI flag (default OFF → weights-only ckpt, ~80% smaller; was default-on causing 40GB ckpts)
  - bf16 cast in `_save_model_and_state` (18GB → 9GB per ckpt for 4B)
  - FSDP wrap class branch for `Qwen3VLTextDecoderLayer`
  - `_projector_constructor_kwargs` v1/v2 Q-Former branch
  - Q-Former dispatch supports `qformer.pretrained: true`
- `scripts/multimodal_planning_dataset.py`: M-RoPE `mm_token_type_ids` extract/align/include (Qwen3-VL compat)
- `scripts/qformer_projector_blip2.py`: NEW — Blip2QFormerProjector wrapping HF Blip2QFormerModel
- `scripts/planning_eval.py`: branch on `pretrained=True` for Blip2QFormerProjector loader
- `configs/nuscenes_planning_1cam_qwen3vl_multimodal.yaml`: B.5'' config
- `configs/nuscenes_planning_1cam_qformer_blip2.yaml`: A.1 v2 config
- `scripts/smoke_1cam_qwen3vl_5step.sh`, `scripts/smoke_a1_v2_blip2_5step.sh`, `scripts/launch_b5pp_1cam_qwen3vl.sh`, `scripts/launch_a1_v2_blip2.sh`: launchers

## Recommendations for next session

1. **Keep R1' Linear as the production reference** — it's our best L2 (0.642). For XPeng JD: lead with R1' + the architectural ablation table (A.1 v1/v2, A.2, A.3) showing we *tested* Q-Former and chose Linear based on data-scale evidence.
2. **TRT deploy via B.5''** (only option). Half-day venv rebuild + conversion script needed.
3. **Don't run more Q-Former-class experiments at 24K scale** — A.1 v2 with proper pretrained init is the strongest signal we'll get and it lost. To make Q-Former competitive, would need 100K+ paired trajectories (not available).
4. **Future B.7-class work**: only justified if we add data (DriveLM-VLA + Waymo Open + nuScenes ≈ 200K samples). At that scale, native compression starts to make sense.
