# B.5'' Full Multimodal TRT Bench — Resume Spec

**Date**: 2026-05-24
**Status**: Phase 1 partially done. Phases 2-6 pending. Total ETA ~8-13h
**User directive**: 不动 assertion，走 disagg + e2e overnight。
**Task #140 in TaskList.**

## Phase 1 — DONE (research)

Established:
- TRT-LLM 1.3.0rc15 has `modeling_qwen3vl.py:920` single-modality assertion blocking B.5'' (video + image)
- WORKAROUND: cache-hit path in `get_multimodal_embeddings` — if `param.multimodal_data["multimodal_embedding"]` is pre-populated, encoder forward is never called → assertion never fires
- Required pre-computed embed shape: `(mm_total, hidden * (1 + deepstack_levels)) = (157, 2560 * 4 = 10240)`
- B.5'' val sample[0] structure:
  - pixel_values: (484, 1536), image_grid_thw: [[1,22,22]] → post-merger 121 tokens
  - pixel_values_videos: (144, 1536), video_grid_thw: [[2,6,12]] → post-merger 36 tokens
  - input_ids: shape (743,), prompt_len=727
  - mm_token_type_ids: 0=text, 1=image, 2=video
  - **Order in input_ids**: VIDEO FIRST (positions 10-53, 36 tokens), then IMAGE (positions 56-176, 121 tokens)
  - image_pad=151655, video_pad=151656
- HF API for getting embeds:
  - `model.get_image_features(pixel_values, image_grid_thw)` → `BaseModelOutputWithDeepstackFeatures`
  - `.pooler_output` is the POST-MERGER base embed (121, 2560 for image; 36, 2560 for video)
  - `.deepstack_features` is list of 3 tensors, each (121, 2560) or (36, 2560)
  - `.last_hidden_state` is pre-merger (484, 1024) — NOT what we want
- HF Qwen3VLModel.forward path (relevant lines):
  ```python
  image_outputs = self.get_image_features(...)
  image_embeds = image_outputs.pooler_output       # post-merger
  deepstack_image_embeds = image_outputs.deepstack_features  # list of 3
  image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.dtype)
  inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
  # ... then deepstack passed separately to llm as kwarg
  ```
- TRT format for pre-computed embed (per modeling_qwen3vl.py:934 pattern):
  ```python
  per_modality_concat = torch.cat([base_post_merger] + deepstack_list, dim=1)  # (n_tokens, 10240)
  mm_embed = torch.cat([video_concat, image_concat], dim=0)  # (157, 10240) ← video first, image second
  ```

## Phase 2 — TODO: compute mrope_config

`mrope_config` must contain:
- `mrope_position_ids`: shape (3, 1, seq_len) — 3-D position ids for T/H/W axes
- `mrope_position_deltas`: shape (1, 1) per request

Source: HF's `get_rope_index` / `compute_mrope_position_ids` in modeling_qwen3_vl.py. Need to either:
- a) Call HF's compute method directly (find it via `model.get_vision_position_ids` or similar)
- b) Port the logic to TRT format

Decision pending until Phase 3 (might let HF processor output stay in the request).

## Phase 3 — TODO: MultimodalParams construction + inject path

Two sub-tasks:
- 3a: Find LLM.generate's hook for raw MultimodalParams. Options:
  - `llm.generate({"prompt_token_ids": [...], "multi_modal_data": {"multimodal_embedding": ...}})` (probably won't work — multi_modal_data is interpreted as raw HF inputs)
  - Patch Qwen3VLInputProcessorBase.__call__ to detect a special "skip_encoder" flag and inject our pre-computed embed
  - Use lower-level Executor.enqueue_request() API (bypass LLM class entirely)
- 3b: For each request, set `multimodal_data["multimodal_embedding"] = mm_embed (157, 10240)` BEFORE forward

## Phase 4 — TODO: logit parity gate

- Run HF model forward on the full sample (text + video + image) — capture prefill last-token logits (top-5)
- Run TRT engine with our pre-computed embeds + same input_ids — capture prefill last-token logits
- Top-5 overlap ≥ 4 → PASS, proceed to bench
- < 3 → ABORT, embeds are wrong

## Phase 5 — TODO: bench

- TTFT, decode/tok, full 14-tok, throughput on N=20 runs
- Output: `deploy/trt_bench/B5pp_trt_qwen3vl_multimodal_bf16.json`
- Companion HF baseline (full multimodal HF forward) at `deploy/trt_bench/B5pp_hf_qwen3vl_multimodal_bf16.json`
- Note: HF bench needs the same M-RoPE manual prefill+decode loop hack that v4 used (from previous session) since `generate()` doesn't propagate mm_token_type_ids — OR use HF's `processor.apply_chat_template` form which routes through correct path

## Phase 6 — TODO: COMPARISON.md update + commit

- Add "Multimodal full payload" rows next to existing text-only rows
- Honest comparison
- Commit + push

## Known risks / future blockers

1. **Phase 3a inject hook**: TRT-LLM 1.3 public API doesn't expose a clean way to inject pre-computed embeds. Likely need custom InputProcessor or low-level Executor API. May take 2-4h to find right hook.
2. **mrope_position_ids drift**: HF's compute logic interacts with `<|vision_start|>` / `<|vision_end|>` markers + grid_thw to produce 3-D coords. Easy to get wrong; port carefully.
3. **Deepstack ordering**: TRT does `cat([base] + ds_list, dim=1)`. `ds_list` order = deepstack_visual_indexes order = [5, 11, 17]. Match exactly.
4. **Disk**: at 69G now. Each load of HF + TRT models eats ~20G RAM + temp ckpt. Watch for `/tmp` fill.

## Restart commands

```bash
cd /workspace/DriveLM_VLM_Project
# Phase 1 re-run if needed (it does just vision pre-compute, fast, saves to _phase1_embeds.pt)
/usr/bin/python3 deploy/multimodal_trt/phase1_vision_precompute.py

# Phase 2-4 not written yet — see TODO section
```

## Files

- `deploy/multimodal_trt_plan.md` — original plan doc
- `deploy/multimodal_trt/phase1_vision_precompute.py` — Phase 1 done
- `deploy/multimodal_trt/_phase1_embeds.pt` — Phase 1 output cache
- `deploy/multimodal_trt/RESUME_SPEC.md` — THIS FILE
