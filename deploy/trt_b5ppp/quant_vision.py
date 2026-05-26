#!/usr/bin/env /venv/trt_llm/bin/python
"""B.5'' v2 vision-tower (ViT) PTQ helper — quantize `model.model.visual`.

WHY THIS EXISTS
---------------
The prior full-pipeline bench left the vision tower bf16 and the feasibility
note claimed the ViT is "unquantizable by design". That is WRONG. Quantizing a
ViT (Linear + attention) is a standard deployment operation. The ViT stayed
bf16 for TWO *fixable* reasons, both verified in source:

  1. Our PTQ scripts (quant_fp8.py / quant_nvfp4.py) call
       mtq.quantize(model.model.language_model, ...)
     ONLY — they never touch model.model.visual. modelopt can quantize the ViT
     just as well; it just was not asked to.

  2. TRT-LLM 1.3.0rc15's bundled Qwen3-VL class strips the ViT quant config on
     load:
       modeling_qwen3vl.py:823
         # Re-setting QuantConfig to exclude vision encoder weights ...
         self.model_config.quant_config = QuantConfig(
             kv_cache_quant_algo=...)
         self.visual = model_class(self.model_config).to(self.model_dtype)  # bf16
     That is a FRAMEWORK INTEGRATION CHOICE in the bundled multimodal path, NOT
     a TensorRT capability limit. The standard automotive "vision-as-separate-
     engine" deploy pattern sidesteps it: quantize model.visual with modelopt,
     run it as its own quantized module/engine, feed pruned embeddings to the LM
     engine. That is exactly the embedding-injection seam this bench already uses
     for the LM.

WHAT THIS MODULE DOES
---------------------
`quantize_vision_inplace(hf_model, calib_ds, precision, n_calib)` quantizes
`hf_model.model.visual` IN-PLACE with modelopt FP8_DEFAULT_CFG / NVFP4_DEFAULT_CFG,
calibrating on a handful of REAL multimodal samples through the SAME forward path
the bench uses (get_video_features / get_image_features -> .visual). The result
is a fake-quant ViT: numerically faithful to a real fp8/nvfp4 ViT for ACCURACY
(modelopt inserts quantize/dequantize around the real low-precision math), so the
bench's reported L2 reflects a genuinely quantized vision tower.

LATENCY HONESTY: fake-quant runs the underlying matmuls in the original dtype
with extra (de)quant ops, so it is NOT a speedup measurement. A real fp8 ViT
engine (modelopt-export + TRT) is the deploy artifact; producing/booting that is
out of scope for the in-session bench. Callers MUST label vision latency as
"fake-quant (accuracy-faithful); real fp8 ViT engine pending" — NOT as a measured
vision speedup. See bench_full_pipeline.py stages.vision.latency_basis.
"""
from __future__ import annotations

from typing import Optional


def _move_for_full_forward(sample, device, dtype):
    import torch
    out = {}
    for k, v in sample.items():
        if k.startswith("_meta_") or not isinstance(v, torch.Tensor):
            continue
        if k in ("input_ids", "attention_mask", "labels", "mm_token_type_ids"):
            v = v.unsqueeze(0)
        if k in ("image_grid_thw", "video_grid_thw") and v.ndim == 1:
            v = v.unsqueeze(0)
        out[k] = v.to(device, dtype=dtype) if v.dtype.is_floating_point else v.to(device)
    out.pop("labels", None)
    return out


def quantize_vision_inplace(*, hf_model, calib_ds, precision: str,
                            n_calib: int = 8, device: str = "cuda:0",
                            verbose: bool = True) -> dict:
    """Quantize hf_model.model.visual in-place with modelopt fake-quant.

    Args:
        hf_model:  loaded Qwen3-VL HF model (bf16). model.model.visual is the ViT.
        calib_ds:  dataset yielding multimodal samples (MultiModalPlanningDataset).
        precision: "fp8" or "nvfp4". (bf16 -> no-op, returns quantized=False.)
        n_calib:   number of real samples for activation calibration.

    Returns dict: {quantized, precision, cfg, n_calib_done, vision_module}.
    """
    import torch

    if precision == "bf16":
        if verbose:
            print("[visquant] precision=bf16 -> vision tower stays bf16 (no-op).")
        return {"quantized": False, "precision": "bf16", "cfg": None,
                "n_calib_done": 0}

    import modelopt.torch.quantization as mtq
    if precision == "fp8":
        from modelopt.torch.quantization import FP8_DEFAULT_CFG as CFG
        cfg_name = "FP8_DEFAULT_CFG"
    elif precision == "nvfp4":
        from modelopt.torch.quantization import NVFP4_DEFAULT_CFG as CFG
        cfg_name = "NVFP4_DEFAULT_CFG"
    else:
        raise ValueError(f"unsupported precision {precision!r}")

    # The ViT lives at model.model.visual (verified: quant_fp8.py:83 logs
    # type(model.model.visual)). It is Linear + attention -> modelopt-quantizable.
    visual = hf_model.model.visual
    dtype = next(hf_model.parameters()).dtype
    n_calib = min(int(n_calib), len(calib_ds))
    if verbose:
        print(f"[visquant] quantizing model.model.visual "
              f"({type(visual).__name__}) with {cfg_name} on {n_calib} real "
              f"multimodal samples (fake-quant; accuracy-faithful)")

    n_done = 0

    def forward_loop(_vis):
        """modelopt passes the (now wrapped) visual module as `_vis`. We drive
        calibration through the SAME entry the bench uses: get_video_features /
        get_image_features, which internally call hf_model.model.visual, so the
        activation observers on the ViT see exactly the bench's distribution."""
        nonlocal n_done
        hf_model.eval()
        with torch.inference_mode():
            for i in range(n_calib):
                try:
                    sample = calib_ds[i]
                except Exception as e:
                    print(f"[visquant] calib sample {i} failed: {e}; skip")
                    continue
                pv = sample.get("pixel_values")
                ig = sample.get("image_grid_thw")
                pvv = sample.get("pixel_values_videos")
                vg = sample.get("video_grid_thw")
                try:
                    if pvv is not None and vg is not None:
                        if vg.ndim == 1:
                            vg = vg.unsqueeze(0)
                        hf_model.get_video_features(
                            pixel_values_videos=pvv.to(device, dtype=dtype),
                            video_grid_thw=vg.to(device))
                    if pv is not None and ig is not None:
                        if ig.ndim == 1:
                            ig = ig.unsqueeze(0)
                        hf_model.get_image_features(
                            pixel_values=pv.to(device, dtype=dtype),
                            image_grid_thw=ig.to(device))
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print(f"[visquant] OOM on sample {i}; freed, continuing")
                    continue
                n_done += 1
                if verbose and (n_done % 4 == 0 or n_done == n_calib):
                    print(f"[visquant]   calib {n_done}/{n_calib}")

    t = __import__("time").perf_counter()
    mtq.quantize(visual, CFG, forward_loop)
    if verbose:
        dt = __import__("time").perf_counter() - t
        print(f"[visquant] mtq.quantize(visual) done in {dt:.1f}s "
              f"({n_done} calib forwards)")
        try:
            mtq.print_quant_summary(visual)
        except Exception as e:
            print(f"[visquant] print_quant_summary failed: {e} (non-fatal)")

    return {"quantized": True, "precision": precision, "cfg": cfg_name,
            "n_calib_done": n_done, "vision_module": type(visual).__name__}
