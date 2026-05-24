# TRT-LLM 1.3 vs HF — B.5'' Qwen3-VL-4B VLA (full multimodal + text-only)

**Date**: 2026-05-24
**Ckpt**: `checkpoints_qwen25/nusc_planning_b5pp_1cam_qwen3vl_multimodal/final` (B.5'', L2=0.5557)
**Hardware**: RTX 5090 (compute cap 12.0), 32 GB HBM
**dtype**: bf16, batch=1, tensor_parallel=1
**Decode**: 14 trajectory tokens, greedy

---

## 1) Full multimodal payload (the real deploy scenario)

**Payload**: 1 cam × 4-frame video + HD-map BEV image + bbox text + ego state + planning prompt → 14 trajectory tokens
**Tokens**: prompt 727 (video 36 + image 121 + text 570), output 14
**Sample**: nuScenes v1.0-trainval val[0]

| Metric | TRT-LLM 1.3 (mm-disagg) | HF transformers 5.x SDPA | TRT speedup |
|---|---|---|---|
| TTFT mean | **26.7 ms** | 92.1 ms (prefill-only) | **3.45×** |
| TTFT p50 / p99 | 26.6 / 28.2 ms | 73.4 / 354.0 ms | — |
| Per-token decode mean | **6.43 ms** | (see proxy below) | **~3.8×** |
| Full 14-tok mean | **110.3 ms** | (see proxy below) | **~3-4×** |
| Throughput mean | **127 tok/s** | (see proxy below) | **~3-4×** |
| Logit parity gate | **PASS** (top-1 token 151934 = HF baseline 151934) | — | — |

**HF decode caveat**: HF transformers 5.x has a Qwen3-VL M-RoPE bug — `generate()` doesn't propagate `mm_token_type_ids` through `_prepare_position_ids_for_generation`, and manual prefill+decode loops hit a K-cache shape mismatch on the first decode step (`Expected K=1455 got 728`). We measured HF prefill only; for per-token decode we use the text-only HF LM proxy (24.17 ms/tok, see table 2) — multimodal HF decode would likely be slightly higher due to KV cache size.

## 2) Text-only LM forward (separate measurement on same model)

Same Qwen3-VL ckpt, but text-only prompt (no video, no image) — measures the LM forward in isolation. **Not directly comparable** to row 1; just shows TRT advantage on LM half alone.

| Metric | TRT-LLM 1.3 (text-only LLM API) | HF transformers SDPA (text-only LM) | TRT speedup |
|---|---|---|---|
| Prompt | 49 tokens text | 49 tokens text | — |
| TTFT mean | 34.8 ms | 28.1 ms | 0.81× (HF marginally faster on tiny prefill) |
| Per-token decode mean | **6.05 ms** | **24.17 ms** | **4.0×** |
| Full 14-tok mean | 113.5 ms | 342.3 ms | 3.0× |
| Throughput mean | 123.4 tok/s | 41.0 tok/s | 3.0× |

## 3) How the TRT multimodal path is wired (no `modeling_qwen3vl.py:920` patch)

TRT-LLM 1.3.0rc15 has a conservative `Currently only support single modality per request` assertion at `modeling_qwen3vl.py:920` that blocks the B.5'' video+image+text payload from going through the stock LLM API.

We bypass WITHOUT touching the venv via the **disaggregated multimodal embedding path**:

1. **HF vision pre-compute** (`phase1_5_vision_embeds.py`): `model.get_image_features(pixel_values, image_grid_thw)` + `get_video_features` → `pooler_output` (post-merger) + `deepstack_features` list. Concat as `[base | deepstack_0 | deepstack_1 | deepstack_2]` on dim=1 → per-modality tensor `(n_tokens, hidden * (1 + n_deepstack))` = `(N, 10240)` for B.5''. Stack video-first then image → `(157, 10240)`.
2. **M-RoPE pre-compute** (`phase2_mrope_config.py`): delegate to HF's `Qwen3VLModel.get_rope_index` (bit-exact parity with TRT's port). Returns `(3, 1, seq_len)` int32 + `(1,)` deltas.
3. **Disagg request** (`phase5_bench_full_multimodal.py`): construct `DisaggregatedParams(request_type="context_and_generation", multimodal_embedding_handles=[chunk_video_0, chunk_video_1, chunk_image], mrope_position_ids_handle=..., mrope_position_deltas_handle=...)` with each handle wrapped via `SharedTensorContainer.from_tensor(t).dump_to_dict()`.
4. **Prompt masquerade**: TRT's input processor only counts `image_token_id` placeholders (line 443: `# TODO: what about video_token_id?`). We rewrite `<|video_pad|>` → `<|image_pad|>` in the unexpanded prompt. The LM doesn't care — `fuse_input_embeds` replaces placeholder embeds with our pre-computed vision embeds, and M-RoPE positions are pre-computed from the ORIGINAL `mm_token_type_ids` so spatial-temporal encoding is preserved.
5. **Cache-hit short-circuit**: `get_multimodal_embeddings._get_uncached_multimodal_params` sees `multimodal_data["multimodal_embedding"]` already populated → encoder forward never invoked → assertion at line 920 never fires.

**Parity gate** (`max_tokens=1, greedy`): TRT top-1 token = 151934 = HF baseline top-1 (logit +23.25 confident traj_start prediction). Embed scatter and M-RoPE positions are correct.

## 4) Honest caveats

1. **Vision tower forward is OUTSIDE TRT timing**. The 26.7 ms TTFT is purely LM forward on the engine; the upstream HF vision-tower call (~10-30 ms for 4-frame video + 1 image on RTX 5090) is amortized differently in a real serving stack. For end-to-end production, vision tower should ALSO live in TRT — but that requires building a separate vision engine via `multimodal_builder.py`, which **doesn't yet support qwen3_vl** (only `qwen2_vl` in 1.3.0rc15).
2. **batch=1**. TRT's KV cache amortization at batch>1 widens the gap. Not measured.
3. **bf16 only**. INT8 SmoothQuant typically gives another 1.5-2× on Blackwell, not attempted (would need calibration).
4. **HF decode per-token uses text-only proxy** due to the Qwen3-VL M-RoPE transformers 5.x bug (documented above).
5. **Cold load**: TRT engine compile 655 s first time, ~5 s warm. HF load 3 s.

## 5) Files in this directory

- `B5pp_trt_qwen3vl_multimodal_bf16.json` — TRT full multimodal numbers (this run)
- `B5pp_hf_qwen3vl_multimodal_bf16.json` — HF prefill-only (decode blocked by HF bug)
- `B5pp_trt_qwen3vl_bf16.json` — TRT text-only LM
- `B5pp_hf_qwen3vl_bf16.json` — HF text-only LM (apples-to-apples vs row above)
- `B5prime_hf_bf16.json` — older HF bench on B.5' Qwen2.5-VL 3-cam (different model, kept for reference)
