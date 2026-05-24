# B.5'' Full Multimodal TRT Bench — Overnight Plan

**Goal**: Bench TRT-LLM 1.3.0rc15 on B.5'' (Qwen3-VL-4B) with REAL multimodal
payload (1cam×4f video + HD-map image + bbox/ego text → 14 traj tokens),
WITHOUT patching the upstream single-modality assertion at modeling_qwen3vl.py:920.

## Strategy

Use the **cache-hit path** in `get_multimodal_embeddings`:
1. Pre-compute mm embeds in HF vision tower (image + video, with 3-level deepstack)
2. Inject as `param.multimodal_data["multimodal_embedding"]`
3. `get_multimodal_embeddings._get_uncached_multimodal_params()` finds the cache hit
4. Encoder forward never called → assertion never fires
5. `fuse_input_embeds` scatters concat embed into LM input by mm_token_indices

## Phases (with gates between)

| Phase | What | Gate |
|---|---|---|
| 1 | HF vision pre-compute on B.5'' val sample | embed shape == (mm_total, 2560*4=10240) |
| 2 | mrope_config compute (port HF compute_3d_position_ids) | 3D position_ids shape == (3, batch, seq_len) |
| 3 | MultimodalParams manual construction (bypass InputProcessor) | LLM.generate runs without error |
| 4 | Logit parity gate vs HF full forward | top-5 agreement ≥ 4/5 on prefill last-token |
| 5 | Full bench TTFT/decode/throughput | results JSON saved |
| 6 | Update COMPARISON.md with honest mixed-modality numbers + commit | git push |

**ABORT criteria** (any of):
- Phase 4 parity < 3/5 top-5 → embeds are scattered wrong, results not meaningful
- Phase 3 yields > 5 cascading errors with no path forward → STOP and surface
- Disk drops below 15G during run → halt per disk panic protocol

## Pre-launch checklist (MANDATORY per memory)

- [ ] paper-hyperparam audit table — N/A (inference bench, not training)
- [ ] boot warnings audited — TBD before Phase 1
- [ ] e2e save+load smoke — N/A (no training)
- [ ] gradient sync — N/A
- [ ] real-data smoke — IS the smoke (Phase 4)
- [ ] disk ≥ 30G free — currently 69G, ample
- [ ] ETA estimate — Phase 1-3: ~4h, Phase 4: ~1h, Phase 5: ~1h, Phase 6: ~1h. Total ~7h.

## Risks (top 3)

1. **Deepstack ordering**: B.5'' uses `deepstack_visual_indexes=[5, 11, 17]`.
   Pre-computed embed must be `cat([base, deep_5, deep_11, deep_17], dim=1)` of width
   `2560*4=10240`. If ordering wrong, LM gets garbage. Mitigation: replicate the
   `cat([image_embeds] + deepstack_image_embeds, dim=1)` pattern from
   modeling_qwen3vl.py:934.
2. **mrope_config drift**: HF's `compute_3d_position_ids` is intricate (uses
   mm_token_type_ids, video_grid_thw, second_per_grid_ts which Qwen3-VL doesn't
   have but old Qwen2.5-VL did). Need to port carefully. Mitigation: run HF first,
   capture its computed position_ids, feed to TRT directly.
3. **InputProcessor bypass**: TRT's pipeline normally calls Qwen3VLInputProcessorBase
   which does HF processor + mrope. Bypassing it means manually constructing
   MultimodalParams with all required fields. Mitigation: read InputProcessor.__call__
   to see exact output struct, replicate it.
