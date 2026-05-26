# Full-Pipeline Quantized Bench — Feasibility Note (B.5'' v2, TRT-LLM 1.3.0rc15)

Date: 2026-05-26 (CORRECTED 2026-05-26: removed the wrong "ViT is unquantizable by design" claim;
the ViT IS quantizable — see Limitation B). Target:
`checkpoints_qwen25/nusc_planning_b5pp_1cam_qwen3vl_multimodal/final`
(Qwen3-VL-4B, 1-cam native video [2,56,100]=2800 tok + HD-map 121 img tok + bbox/ego text).
venv: `/venv/trt_llm/bin/python` (TRT-LLM 1.3.0rc15).

## The central question
Can TRT-LLM 1.3.0rc15 run the Qwen3-VL **vision tower IN-ENGINE** (raw image/video -> logits in
one TRT `generate` call), so we can measure + quantize the full deployed path? Or is it limited to
precomputed embeddings (the mm-disagg workaround in `bench_trt.py`)?

## Evidence (TRT-LLM source, quoted)

`/venv/trt_llm/lib/python3.12/site-packages/tensorrt_llm/_torch/models/modeling_qwen3vl.py`

1. **There IS an in-engine vision tower.** `Qwen3VisionModelBase` (line 813) holds `self.visual`
   (the full ViT), implements `load_weights` (line 835, loads `model.visual.*`) and a `forward`
   (line 915) that takes raw `pixel_values` / `pixel_values_videos`. The model registers it as the
   vision encoder: `@register_vision_encoder(Qwen3VisionModelBase, ...)` (line 1203).

2. **The non-disagg path runs the vision tower inside `generate`.** `Qwen3VLModelBase.forward`
   (line 1109) dispatches:
   ```python
   if not _is_disagg():
       mm_embeds = get_multimodal_embeddings(
           encoder_forward_fn=self.mm_encoder.forward,   # <-- runs the ViT
           multimodal_params=mm_multimodal_params)
   ```
   `_is_disagg()` is just `os.getenv("TLLM_MULTIMODAL_DISAGGREGATED","0")=="1"` (modeling_multimodal_utils.py:38).
   Default = NOT disagg, so by default vision runs in-engine.

3. **The input processor accepts RAW media (PIL / video frames), not just embeddings.**
   `Qwen3VLInputProcessorBase.__call__` (line 355) reads `inputs["multi_modal_data"]`, runs the HF
   processor (`_preprocess`, line 294) and stores `multimodal_data["image"]={"pixel_values":...}` /
   `multimodal_data["video"]={"pixel_values_videos":...}`. So `LLM.generate([{"prompt":..,
   "multi_modal_data": {"image":[...], "video":[...]}}])` is the native multimodal API.

### So why does the existing bench use mm-disagg with precomputed embeddings?
Two hard limitations, both verified in source:

**LIMITATION A — single modality per request (HARD BLOCKER for B.5'' native path).**
`Qwen3VisionModelBase.forward` (line 920):
```python
if pixel_values is not None and pixel_values_videos is not None:
    raise ValueError("Currently only support single modality per request")
```
B.5'' v2 sends BOTH a video (the camera, video_pad) AND an image (HD-map BEV, image_pad) in the
**same** request. The in-engine vision tower refuses this. There is no per-request workaround short
of patching TRT-LLM (calling `self.visual` twice and concatenating) — which would be modifying the
installed package, not a deployable config. This is the real reason `bench_trt.py` precomputes
embeddings on HF and injects via DisaggregatedParams (where the dual-modality split is done by us
in `assemble_mm_embedding`).

**LIMITATION B (CORRECTED 2026-05-26) — the *bundled TRT-LLM Qwen3-VL path* keeps the ViT bf16; this
is NOT a TRT limit and the ViT is NOT "unquantizable".**

Earlier this note claimed the ViT is "NEVER quantized, by design" and that "none can be produced".
That was wrong. Quantizing a ViT (Linear + attention) is a standard deployment operation. The ViT
stayed bf16 in the prior run for TWO *fixable* reasons:

1. **Our PTQ scripts were LM-only.** `quant_fp8.py` / `quant_nvfp4.py` call
   `mtq.quantize(model.model.language_model, ...)` and never touch `model.model.visual`. modelopt
   quantizes the ViT just fine — it simply was not asked to. We now do exactly that in
   `quant_vision.py`: `mtq.quantize(model.model.visual, FP8_DEFAULT_CFG / NVFP4_DEFAULT_CFG, ...)`,
   calibrated on real multimodal samples. Verified: modelopt inserts 387 TensorQuantizers across the
   ViT blocks / attn / mlp / merger / deepstack_merger_list.

2. **TRT-LLM's bundled class strips the ViT quant config on LOAD** — a framework integration choice,
   not a capability limit. `Qwen3VisionModelBase.__init__` (line 823):
   ```python
   # Re-setting QuantConfig to exclude vision encoder weights from quantization load.
   self.model_config.quant_config = QuantConfig(
       kv_cache_quant_algo=self.model_config.quant_config.kv_cache_quant_algo)
   self.visual = model_class(self.model_config).to(self.model_dtype)  # bf16
   ```
   This only governs the *in-engine, bundled* multimodal path (which is anyway blocked for B.5'' by
   Limitation A). It does NOT mean TensorRT cannot run a quantized ViT.

**The deployable route is the standard automotive "vision-as-separate-engine" pattern**: quantize
`model.visual` with modelopt, run it as its own quantized module/engine, then feed the (FasterVLM-
pruned) embeddings into the LM engine via the same embedding-injection seam already used here for the
LM. So fp8/nvfp4 DO get a quantized ViT — the bench now reports L2 from a modelopt-quantized vision
tower (`stages.vision.quantized=true`, `dtype=fp8|nvfp4`).

**Latency honesty.** The bench runs the modelopt *fake-quant* ViT, which is accuracy-faithful (the L2
reflects quantized vision) but is NOT a vision speedup — fake-quant runs the same matmuls plus de/quant
ops, so `vision_ms` is ~bf16-equivalent. A real fp8 ViT TensorRT engine (modelopt-export + TRT build)
is the artifact that delivers the speedup; it is not booted in-session and is labeled
`latency_basis="fake-quant (accuracy-faithful); real fp8/nvfp4 ViT engine pending"` in the JSON.

**Consequence for FasterVLM.** When vision runs in-engine, `mm_encoder.forward` returns embeddings
directly into `fuse_input_embeds` inside the same `generate` call (line 1139-1161). There is **no
seam** to insert FasterVLM token pruning as an in-pipeline operator on the TRT side. FasterVLM is a
training-free op on the post-merger visual tokens; to deploy it as a real in-pipeline operator we
must run vision OUTSIDE the LM engine, prune, then feed the pruned embeddings to the LM. That is
exactly the mm-disagg embedding-injection seam.

## Chosen architecture (honest, deployable, fully measurable)

Given Limitation A, the bundled in-engine native multimodal path is not usable for this dual-modality
model in 1.3.0rc15, and it would also remove the FasterVLM insertion point. The HONEST design that
measures the COMPLETE path end-to-end, supports FasterVLM as a real operator, AND quantizes the ViT
(per the corrected Limitation B — the ViT IS quantizable via the vision-as-separate-engine route):

```
raw video frames + HD-map image
        │
        ▼  [STAGE 1: VISION]  HF Qwen3-VL ViT, GPU. bf16 for the bf16 run; modelopt fp8/nvfp4
        │                      (fake-quant, accuracy-faithful) for fp8/nvfp4 runs — see Lim. B.
        │                      get_video_features + get_image_features  — TIMED
        ▼  [STAGE 2: COMPRESS] FasterVLM compress_visual_tokens on the post-merger VIDEO tokens
        │                      (HD-map untouched) — in-pipeline operator — TIMED
        ▼  [STAGE 3: PREFILL ] LM in TRT (bf16 / fp8 / nvfp4) via embedding injection
        │                      (DisaggregatedParams + CPU SharedTensorContainer handles) — TIMED
        ▼  [STAGE 4: DECODE  ] greedy-decode the full 14-token trajectory in TRT — TIMED
        ▼  trajectory tokens -> waypoints -> L2 vs HF bf16 reference (N≈20-50 val samples)
```

What this measures that `bench_trt.py` did NOT:
- **vision_ms** — vision tower forward is now inside the timed end-to-end window (was precomputed once, untimed).
- **compress_ms** — FasterVLM runs as a timed in-pipeline stage (was applied to precomputed embeds, untimed).
- **full_traj_ms (END-TO-END)** = vision + compress + prefill + decode, not LM-only TTFT.
- **peak GPU mem across the whole stack** (vision tower + LM engine resident together).

### Honesty labels carried into the JSON
- `stages.vision.quantized`: **true** for fp8/nvfp4 (modelopt-quantized ViT), **false** for bf16
  (correct — bf16 vision is right for the bf16 run). `stages.vision.dtype` = `fp8`/`nvfp4`/`bf16`.
- For quantized vision: `stages.vision.quant_method = "modelopt FP8_DEFAULT_CFG / NVFP4_DEFAULT_CFG"`,
  `quant_target = "model.model.visual"`, and
  `latency_basis = "fake-quant (accuracy-faithful); real fp8/nvfp4 ViT engine pending"` — i.e. the L2
  reflects a genuinely quantized ViT, but `vision_ms` is NOT a measured quantized-ViT speedup.
- `stages.{prefill,decode}.quantized = (precision != bf16)`, `dtype = precision`.
- `native_vision_in_trt = false` with `reason` covering BOTH (1) the line-920 single-modality
  ValueError (the real blocker for the bundled in-engine path) and (2) the corrected statement that
  the line-823 ViT-quant-strip is a framework choice, not a TRT limit — the ViT IS quantizable.

### Why not patch TRT-LLM to do dual-modality in-engine?
That edits the installed site-package (not a deployable artifact) and removes the FasterVLM seam. It
would produce a LESS representative deploy number, not a more honest one. Rejected. (Note: this is
NOT because the ViT is unquantizable — it is quantizable; we quantize it via the
vision-as-separate-engine route, which also preserves the FasterVLM seam.)

## Limitations / what remains
- **Vision tower IS quantized for fp8/nvfp4** via modelopt (`quant_vision.py`, targeting
  `model.model.visual`) — accuracy-faithful fake-quant in this bench. What remains for full
  production deploy is exporting that quantized ViT to a real TensorRT engine (modelopt-export + TRT
  build) so the latency win is realized; the fake-quant ViT here is bf16-equivalent in latency and is
  labeled as such. bf16 run keeps the ViT bf16 (correct).
- The per-call HF vision forward uses the same CPU-shared-tensor injection workaround as
  `bench_trt.py` because CUDA-IPC handle restore is blocked by this container's seccomp
  (pidfd_getfd). Kept verbatim.
