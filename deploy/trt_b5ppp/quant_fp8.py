#!/usr/bin/env /venv/trt_llm/bin/python
"""B.5''' FP8 PTQ: quantize Qwen3-VL-4B language_model only (W8A8 + FP8 KV).

Vision tower (model.model.visual) stays bf16 — only the LM backbone is quantized.
Uses modelopt 0.43.0's mtq.quantize with FP8_DEFAULT_CFG over 256 train samples
from MultiModalPlanningDataset (same loader as training).

Output: writes a NEW sibling dir `quant_fp8/` next to the source ckpt — NEVER
overwrites or removes anything under `final/`.

Usage:
    /venv/trt_llm/bin/python quant_fp8.py \
        [--ckpt /path/to/final] \
        [--calib-n 256] \
        [--out /path/to/quant_fp8] \
        [--device cuda:0]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Shims MUST run before tensorrt_llm or modelopt imports
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from _common import apply_venv_shims  # noqa: E402
apply_venv_shims()

from _common import (  # noqa: E402
    DEFAULT_CKPT,
    DEFAULT_CONFIG_YAML,
    DEFAULT_PARENT,
    build_calib_dataset,
    copy_processor_and_tokenizer,
    load_hf_model_bf16,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FP8 PTQ for B.5''' Qwen3-VL-4B (LM only, vision stays bf16)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt", default=DEFAULT_CKPT,
                   help="Source HF ckpt (B.5''' SFT final/)")
    p.add_argument("--calib-n", type=int, default=256,
                   help="Number of calibration samples from train split")
    p.add_argument("--out", default=str(Path(DEFAULT_PARENT) / "quant_fp8"),
                   help="Output dir for quantized ckpt (sibling of final/)")
    p.add_argument("--config-yaml", default=DEFAULT_CONFIG_YAML,
                   help="Training yaml used to mirror dataset construction")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-new-tokens-loop", type=int, default=0,
                   help="If >0, calibration runs model.generate; else uses model(**batch)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out)

    print(f"[fp8] === STEP 1: validate paths ===")
    src = Path(args.ckpt)
    if not src.is_dir():
        print(f"[fp8] FATAL: source ckpt not found: {src}", file=sys.stderr)
        return 2
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"[fp8] WARN: out dir {out_dir} already populated; refusing to overwrite. "
              f"Move it aside and re-run.", file=sys.stderr)
        return 3
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[fp8] src = {src}")
    print(f"[fp8] out = {out_dir}")

    print(f"[fp8] === STEP 2: load HF model (bf16, sdpa) ===")
    import torch
    model, processor = load_hf_model_bf16(str(src), device=args.device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"[fp8] model params: {n_params:.2f}B")
    print(f"[fp8] vision tower: {type(model.model.visual).__name__} (will STAY bf16)")
    print(f"[fp8] language_model: {type(model.model.language_model).__name__} (target of FP8)")

    print(f"[fp8] === STEP 3: build calibration dataset ({args.calib_n} train samples) ===")
    t0 = time.perf_counter()
    calib_ds = build_calib_dataset(
        processor=processor, n_samples=args.calib_n,
        split="train", config_yaml=args.config_yaml,
    )
    n_actual = len(calib_ds)
    print(f"[fp8] calibration dataset ready: {n_actual} samples in {time.perf_counter()-t0:.1f}s")

    print(f"[fp8] === STEP 4: import modelopt + define forward_loop ===")
    import modelopt.torch.quantization as mtq
    from modelopt.torch.quantization import FP8_DEFAULT_CFG
    print(f"[fp8] modelopt config: FP8_DEFAULT_CFG (W8A8 FP8 + FP8 KV)")

    device = args.device
    dtype = torch.bfloat16

    def _move_to_device(sample: dict) -> dict:
        """Pin batch dim and move to device. Dataset returns 1-D tensors; LM
        forward expects (B=1, L). pixel_values/pixel_values_videos are already
        flat (n_patches, patch_dim) which Qwen3VL processor expects."""
        out = {}
        for k, v in sample.items():
            if k.startswith("_meta_"):
                continue
            if not isinstance(v, torch.Tensor):
                continue
            if k in ("input_ids", "attention_mask", "labels", "mm_token_type_ids"):
                v = v.unsqueeze(0)  # (L,) -> (1, L)
            # Qwen3-VL grid_thw MUST be 2-D (N, 3). MultiModalPlanningDataset
            # squeezes single-modality grids to 1-D (3,) (dataset L446-464), but
            # Qwen3-VL's fast_pos_embed_interpolate does `[row[0] for row in
            # grid_thw]` and raises "'int' object is not subscriptable" on a 1-D
            # grid. train_lora.collate_fn (L326-340) restores 2-D before forward;
            # mirror that here (same patch as bench_trt.py:532-533).
            if k in ("image_grid_thw", "video_grid_thw") and v.ndim == 1:
                v = v.unsqueeze(0)  # (3,) -> (1, 3)
            # Move to device w/ appropriate dtype
            if v.dtype.is_floating_point:
                out[k] = v.to(device, dtype=dtype)
            else:
                out[k] = v.to(device)
        # Drop labels: calibration is a forward-only pass; labels trigger loss
        # path which we don't need (and which can blow up VRAM with logits cat).
        out.pop("labels", None)
        return out

    n_calib = n_actual
    n_done = 0

    def forward_loop(_lm):
        """Run `n_calib` real multimodal samples through the full HF model.

        modelopt passes the QUANTIZED language_model as `_lm`. We ignore it and
        invoke `model.forward(...)` directly — the language_model attribute on
        `model` IS `_lm` (same object), so activation hooks fire correctly,
        AND we get the vision tower forward for free (bf16, no quant).
        """
        nonlocal n_done
        model.eval()
        with torch.inference_mode():
            for i in range(n_calib):
                try:
                    sample = calib_ds[i]
                except Exception as e:
                    print(f"[fp8] calib sample {i} failed: {e}; skipping")
                    continue
                batch = _move_to_device(sample)
                try:
                    _ = model(**batch)
                except torch.cuda.OutOfMemoryError:
                    print(f"[fp8] OOM on sample {i}; freeing and retrying once")
                    torch.cuda.empty_cache()
                    _ = model(**batch)
                n_done += 1
                if (n_done % 16) == 0 or n_done == n_calib:
                    print(f"[fp8]   calib progress: {n_done}/{n_calib}")

    print(f"[fp8] === STEP 5: mtq.quantize(model.model.language_model, FP8_DEFAULT_CFG) ===")
    print(f"[fp8] This will take ~10-30 min depending on calib_n. Streaming progress ...")
    t0 = time.perf_counter()
    mtq.quantize(model.model.language_model, FP8_DEFAULT_CFG, forward_loop)
    print(f"[fp8] mtq.quantize done in {time.perf_counter()-t0:.1f}s ({n_done} forward passes)")

    print(f"[fp8] === STEP 6: quant summary ===")
    try:
        mtq.print_quant_summary(model.model.language_model)
    except Exception as e:
        print(f"[fp8] print_quant_summary failed: {e} (non-fatal)")

    print(f"[fp8] === STEP 7: save quantized model → {out_dir} ===")
    t0 = time.perf_counter()
    model.save_pretrained(str(out_dir))
    print(f"[fp8] save_pretrained in {time.perf_counter()-t0:.1f}s")

    print(f"[fp8] === STEP 8: copy tokenizer + processor from source ===")
    copy_processor_and_tokenizer(str(src), str(out_dir))

    print(f"[fp8] === DONE ===")
    print(f"[fp8] FP8 ckpt written to: {out_dir}")
    print(f"[fp8] Next: /venv/trt_llm/bin/python build_engine.py --ckpt {out_dir} --precision fp8")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
