#!/usr/bin/env /venv/trt_llm/bin/python
"""ROUTE 1: REAL (compressed, not fake-quant) ViT latency bench for B.5'' Qwen3-VL-4B.

Times model.model.visual forward on the REAL 2800-token video input
(video_grid_thw [2,56,100]) at:
  - bf16 (baseline)
  - modelopt fp8 REAL compress (mtq.quantize FP8_DEFAULT_CFG -> mtq.compress
    -> enable_real_quant_gemm). The Linear layers then run torch._scaled_mm
    (real fp8 x fp8 GEMM). Attention (SDPA) stays bf16 (flash-attn broken in env).
  - modelopt nvfp4 REAL compress (if fp8 yields a real speedup).

Honesty: this is a Linear-only real fp8 GEMM speedup; attention math stays bf16.
We separately count Linear vs total FLOPs / time so the label is precise.

Usage:
  CUDA_VISIBLE_DEVICES=0 /venv/trt_llm/bin/python bench_vit_real_quant.py \
      [--precisions bf16,fp8,nvfp4] [--n-warmup 5] [--n-runs 10] [--calib-n 8]
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from _common import apply_venv_shims  # noqa: E402
apply_venv_shims()
from _common import (  # noqa: E402
    DEFAULT_CKPT, DEFAULT_CONFIG_YAML, build_calib_dataset,
    add_project_paths, reset_peak_mem, peak_mem_gb,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    p.add_argument("--config-yaml", default=DEFAULT_CONFIG_YAML)
    p.add_argument("--precisions", default="bf16,fp8,nvfp4")
    p.add_argument("--n-warmup", type=int, default=5)
    p.add_argument("--n-runs", type=int, default=10)
    p.add_argument("--calib-n", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default=str(_HERE / "results" / "vit_real_quant.json"))
    return p.parse_args()


def _get_video_inputs(sample, device, dtype):
    import torch
    pvv = sample.get("pixel_values_videos")
    vg = sample.get("video_grid_thw")
    assert pvv is not None and vg is not None, "sample has no video"
    if vg.ndim == 1:
        vg = vg.unsqueeze(0)
    return pvv.to(device, dtype=dtype), vg.to(device)


def _time_visual(visual, pvv, vg, n_warmup, n_runs):
    """Time visual.forward(pvv, vg). Returns (mean_ms, p50_ms, all_ms, out_shape)."""
    import torch
    visual.eval()
    out_shape = None
    with torch.inference_mode():
        for _ in range(n_warmup):
            out = visual(pvv, vg)
            out_shape = tuple(out.shape) if hasattr(out, "shape") else None
        torch.cuda.synchronize()
        times = []
        for _ in range(n_runs):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            visual(pvv, vg)
            torch.cuda.synchronize()
            times.append(1000.0 * (time.perf_counter() - t0))
    times.sort()
    mean = sum(times) / len(times)
    p50 = times[len(times) // 2]
    return mean, p50, times, out_shape


def _count_real_quant_linears(visual):
    import modelopt.torch.quantization as mtq
    from modelopt.torch.quantization.nn.modules.quant_linear import RealQuantLinear
    n_real = sum(1 for m in visual.modules() if isinstance(m, RealQuantLinear))
    n_fp8_w = 0
    for m in visual.modules():
        w = getattr(m, "weight", None)
        if w is not None and hasattr(w, "dtype") and getattr(w, "dtype", None) is not None:
            try:
                import torch
                # QTensorWrapper holds underlying fp8 storage
                if w.dtype == torch.float8_e4m3fn:
                    n_fp8_w += 1
            except Exception:
                pass
    return n_real, n_fp8_w


def main():
    args = parse_args()
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    add_project_paths()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    device = args.device
    precisions = [p.strip() for p in args.precisions.split(",") if p.strip()]
    results = {"ckpt": args.ckpt, "device": torch.cuda.get_device_name(0),
               "sm": torch.cuda.get_device_capability(0), "runs": {}}

    print(f"[vit] loading processor ...")
    processor = AutoProcessor.from_pretrained(args.ckpt)
    print(f"[vit] building 1 real val sample (video_grid_thw input) ...")
    val_ds = build_calib_dataset(processor=processor, n_samples=2, split="val",
                                 config_yaml=args.config_yaml)
    sample0 = val_ds[0]
    pvv, vg = _get_video_inputs(sample0, device, torch.bfloat16)
    print(f"[vit] video_grid_thw = {vg.tolist()}  pixel_values_videos = {tuple(pvv.shape)}")
    n_post_merge = int(vg.prod(dim=1).sum().item()) // 4  # spatial_merge 2x2
    print(f"[vit] expected post-merge video tokens ~= {n_post_merge}")

    for prec in precisions:
        print(f"\n========== precision = {prec} ==========")
        reset_peak_mem()
        # Fresh load each precision so quant of one doesn't contaminate next.
        print(f"[vit] loading model (bf16 weights, sdpa attn) ...")
        hf = AutoModelForImageTextToText.from_pretrained(
            args.ckpt, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        ).to(device).eval()
        visual = hf.model.visual

        meta = {"precision": prec}
        if prec == "bf16":
            meta["path"] = "bf16 baseline (no quant)"
        else:
            import copy
            import modelopt.torch.quantization as mtq
            if prec == "fp8":
                from modelopt.torch.quantization import FP8_DEFAULT_CFG as _BASE
                cfgname = "FP8_DEFAULT_CFG"
            elif prec == "nvfp4":
                from modelopt.torch.quantization import NVFP4_DEFAULT_CFG as _BASE
                cfgname = "NVFP4_DEFAULT_CFG"
            else:
                raise ValueError(prec)
            # Only Linear layers get a real low-bit GEMM after compress().
            # patch_embed.proj is a Conv3d (QuantConv3d) -> compress does NOT
            # convert convs to RealQuant, leaving a broken fp8-weight/fake-input
            # mix that crashes forward. Disable conv + attention-output quantizers
            # so we quantize ONLY the block Linears (qkv, proj, mlp fc1/fc2, merger).
            CFG = copy.deepcopy(_BASE)
            CFG["quant_cfg"]["*patch_embed*"] = {"enable": False}
            # nn.Conv3d quantizer (QuantConv3d) -> disable: not a real-GEMM target.
            CFG["quant_cfg"]["nn.Conv3d"] = {"*": {"enable": False}}

            # --- calibrate on a few real video samples through visual ---
            calib = build_calib_dataset(processor=processor, n_samples=args.calib_n,
                                        split="train", config_yaml=args.config_yaml)
            ncal = min(args.calib_n, len(calib))

            def forward_loop(_m):
                with torch.inference_mode():
                    for i in range(ncal):
                        try:
                            s = calib[i]
                        except Exception as e:
                            print(f"[vit]  calib {i} build fail {e}"); continue
                        cpvv = s.get("pixel_values_videos"); cvg = s.get("video_grid_thw")
                        if cpvv is None or cvg is None:
                            continue
                        if cvg.ndim == 1:
                            cvg = cvg.unsqueeze(0)
                        try:
                            visual(cpvv.to(device, dtype=torch.bfloat16), cvg.to(device))
                        except torch.cuda.OutOfMemoryError:
                            torch.cuda.empty_cache(); continue

            print(f"[vit] mtq.quantize(visual, {cfgname}) calibrating on {ncal} samples ...")
            t0 = time.perf_counter()
            mtq.quantize(visual, CFG, forward_loop)
            print(f"[vit] quantize (fake) done in {time.perf_counter()-t0:.1f}s")

            # --- REAL compress: convert weights to low-bit + enable real GEMM ---
            print(f"[vit] mtq.compress(visual) -> RealQuantLinear + real GEMM ...")
            t0 = time.perf_counter()
            mtq.compress(visual)
            print(f"[vit] compress done in {time.perf_counter()-t0:.1f}s")
            n_real, n_fp8w = _count_real_quant_linears(visual)
            from modelopt.torch.quantization.backends.gemm_registry import is_real_quant_gemm_enabled
            meta["path"] = f"modelopt {cfgname} -> compress -> real GEMM"
            meta["n_real_quant_linears"] = n_real
            meta["n_fp8_weight_tensors"] = n_fp8w
            meta["real_quant_gemm_enabled"] = bool(is_real_quant_gemm_enabled(visual))
            meta["cfg"] = cfgname
            print(f"[vit] RealQuantLinear count = {n_real}, fp8 weight tensors = {n_fp8w}, "
                  f"real_gemm_enabled = {meta['real_quant_gemm_enabled']}")

        # --- time it ---
        try:
            mean, p50, allms, oshape = _time_visual(visual, pvv, vg,
                                                    args.n_warmup, args.n_runs)
            meta["mean_ms"] = round(mean, 3)
            meta["p50_ms"] = round(p50, 3)
            meta["all_ms"] = [round(x, 3) for x in allms]
            meta["out_shape"] = oshape
            meta["peak_gb"] = round(peak_mem_gb(), 3)
            print(f"[vit] {prec}: mean={mean:.2f}ms p50={p50:.2f}ms out={oshape} "
                  f"peak={meta['peak_gb']}GB")
        except Exception as e:
            import traceback; traceback.print_exc()
            meta["error"] = str(e)
            print(f"[vit] {prec} FAILED: {e}")

        results["runs"][prec] = meta
        del hf, visual
        torch.cuda.empty_cache()

    # --- speedup summary ---
    if "bf16" in results["runs"] and "mean_ms" in results["runs"]["bf16"]:
        base = results["runs"]["bf16"]["mean_ms"]
        for prec, m in results["runs"].items():
            if prec != "bf16" and "mean_ms" in m:
                m["speedup_vs_bf16"] = round(base / m["mean_ms"], 3)
                print(f"[vit] SPEEDUP {prec}: {m['speedup_vs_bf16']}x "
                      f"({base:.2f} -> {m['mean_ms']:.2f} ms)")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[vit] saved -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
