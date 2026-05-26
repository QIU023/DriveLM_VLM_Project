#!/usr/bin/env /venv/trt_llm/bin/python
"""B.5''' NVFP4 PTQ: quantize Qwen3-VL-4B language_model only (W4A4 NVFP4).

Vision tower stays bf16. NVFP4 is Blackwell-native (RTX 5090 SM_120 has HW
tensor-core support). Uses modelopt 0.43.0's NVFP4_DEFAULT_CFG over 256 train
samples from MultiModalPlanningDataset (same loader as training).

Output: NEW sibling dir `quant_nvfp4/` next to the source ckpt — NEVER
overwrites or removes anything under `final/`.

Usage:
    /venv/trt_llm/bin/python quant_nvfp4.py \
        [--ckpt /path/to/final] \
        [--calib-n 256] \
        [--out /path/to/quant_nvfp4] \
        [--awq]          # use NVFP4_AWQ_LITE_CFG instead of plain NVFP4_DEFAULT_CFG
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

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
        description="NVFP4 PTQ for B.5''' Qwen3-VL-4B (LM only, vision stays bf16)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    p.add_argument("--calib-n", type=int, default=256)
    p.add_argument("--out", default=str(Path(DEFAULT_PARENT) / "quant_nvfp4"))
    p.add_argument("--config-yaml", default=DEFAULT_CONFIG_YAML)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--awq", action="store_true",
                   help="Use NVFP4_AWQ_LITE_CFG (W4A4 + AWQ calibration) for better acc")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out)

    print(f"[nvfp4] === STEP 1: validate paths ===")
    src = Path(args.ckpt)
    if not src.is_dir():
        print(f"[nvfp4] FATAL: source ckpt not found: {src}", file=sys.stderr)
        return 2
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"[nvfp4] WARN: out dir {out_dir} already populated; refusing to overwrite. "
              f"Move it aside and re-run.", file=sys.stderr)
        return 3
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[nvfp4] src = {src}")
    print(f"[nvfp4] out = {out_dir}")

    print(f"[nvfp4] === STEP 2: load HF model (bf16, sdpa) ===")
    import torch
    model, processor = load_hf_model_bf16(str(src), device=args.device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"[nvfp4] model params: {n_params:.2f}B")
    print(f"[nvfp4] vision tower: {type(model.model.visual).__name__} (will STAY bf16)")
    print(f"[nvfp4] language_model: {type(model.model.language_model).__name__} (target of NVFP4)")

    print(f"[nvfp4] === STEP 3: build calibration dataset ({args.calib_n} train samples) ===")
    t0 = time.perf_counter()
    calib_ds = build_calib_dataset(
        processor=processor, n_samples=args.calib_n,
        split="train", config_yaml=args.config_yaml,
    )
    n_actual = len(calib_ds)
    print(f"[nvfp4] calibration dataset ready: {n_actual} samples in {time.perf_counter()-t0:.1f}s")

    print(f"[nvfp4] === STEP 4: import modelopt + select config ===")
    import modelopt.torch.quantization as mtq
    if args.awq:
        from modelopt.torch.quantization import NVFP4_AWQ_LITE_CFG as CFG
        cfg_name = "NVFP4_AWQ_LITE_CFG"
    else:
        from modelopt.torch.quantization import NVFP4_DEFAULT_CFG as CFG
        cfg_name = "NVFP4_DEFAULT_CFG"
    print(f"[nvfp4] modelopt config: {cfg_name} (W4A4 NVFP4 — Blackwell SM_120 native)")

    device = args.device
    dtype = torch.bfloat16

    def _move_to_device(sample: dict) -> dict:
        out = {}
        for k, v in sample.items():
            if k.startswith("_meta_"):
                continue
            if not isinstance(v, torch.Tensor):
                continue
            if k in ("input_ids", "attention_mask", "labels", "mm_token_type_ids"):
                v = v.unsqueeze(0)
            # Qwen3-VL grid_thw MUST be 2-D (N, 3). MultiModalPlanningDataset
            # squeezes single-modality grids to 1-D (3,) (dataset L446-464), but
            # Qwen3-VL's fast_pos_embed_interpolate does `[row[0] for row in
            # grid_thw]` and raises "'int' object is not subscriptable" on a 1-D
            # grid. train_lora.collate_fn (L326-340) restores 2-D before forward;
            # mirror that here (same patch as bench_trt.py:532-533).
            if k in ("image_grid_thw", "video_grid_thw") and v.ndim == 1:
                v = v.unsqueeze(0)
            if v.dtype.is_floating_point:
                out[k] = v.to(device, dtype=dtype)
            else:
                out[k] = v.to(device)
        out.pop("labels", None)
        return out

    n_calib = n_actual
    n_done = 0

    def forward_loop(_lm):
        nonlocal n_done
        model.eval()
        with torch.inference_mode():
            for i in range(n_calib):
                try:
                    sample = calib_ds[i]
                except Exception as e:
                    print(f"[nvfp4] calib sample {i} failed: {e}; skipping")
                    continue
                batch = _move_to_device(sample)
                try:
                    _ = model(**batch)
                except torch.cuda.OutOfMemoryError:
                    print(f"[nvfp4] OOM on sample {i}; freeing and retrying once")
                    torch.cuda.empty_cache()
                    _ = model(**batch)
                n_done += 1
                if (n_done % 16) == 0 or n_done == n_calib:
                    print(f"[nvfp4]   calib progress: {n_done}/{n_calib}")

    print(f"[nvfp4] === STEP 5: mtq.quantize(model.model.language_model, {cfg_name}) ===")
    print(f"[nvfp4] This will take ~10-30 min depending on calib_n. Streaming progress ...")
    t0 = time.perf_counter()
    mtq.quantize(model.model.language_model, CFG, forward_loop)
    print(f"[nvfp4] mtq.quantize done in {time.perf_counter()-t0:.1f}s ({n_done} forward passes)")

    print(f"[nvfp4] === STEP 6: quant summary ===")
    try:
        mtq.print_quant_summary(model.model.language_model)
    except Exception as e:
        print(f"[nvfp4] print_quant_summary failed: {e} (non-fatal)")

    print(f"[nvfp4] === STEP 7: EXPORT quantized model (modelopt) → {out_dir} ===")
    # CRITICAL FIX 2026-05-26 (see quant_fp8.py): save_pretrained drops modelopt's
    # quantization -> ckpt loads as bf16. export_hf_checkpoint writes the real
    # NVFP4-compressed safetensors (~3G) + hf_quant_config.json.
    from modelopt.torch.export import export_hf_checkpoint
    t0 = time.perf_counter()
    export_hf_checkpoint(model, export_dir=str(out_dir))
    print(f"[nvfp4] export_hf_checkpoint in {time.perf_counter()-t0:.1f}s")

    print(f"[nvfp4] === STEP 8: copy tokenizer + processor from source ===")
    copy_processor_and_tokenizer(str(src), str(out_dir))

    print(f"[nvfp4] === DONE ===")
    print(f"[nvfp4] NVFP4 ckpt written to: {out_dir}")
    print(f"[nvfp4] Next: /venv/trt_llm/bin/python build_engine.py --ckpt {out_dir} --precision nvfp4")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
