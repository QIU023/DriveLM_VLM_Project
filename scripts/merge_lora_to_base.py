"""Merge a Tier-1 LoRA adapter into the base Qwen2.5-VL weights to produce a
warm-init checkpoint for the Tier-2 full-SFT video-VLA training run.

This is structurally identical to `scripts/merge_lora.py` (which we keep for
backward-compat with the older Layer-3 GGUF pipeline) but uses
`AutoModelForImageTextToText` so it picks up the correct Qwen2.5-VL class on
transformers 5.x — and writes the processor / tokenizer alongside so the
output directory is a drop-in `model_id` for `train_lora.py`.

Usage
-----
  python scripts/merge_lora_to_base.py \
      --base   /workspace/models/Qwen2.5-VL-3B-Instruct \
      --lora   <path to single-image LoRA adapter dir, e.g. checkpoints_qwen25/baseline/checkpoint-46000> \
      --output /workspace/models/Qwen2.5-VL-3B-drivelm-merged

The `--lora` path must contain `adapter_config.json` + `adapter_model.safetensors`
(standard PEFT save layout). If you don't have a Tier-1 LoRA on this box yet,
skip this script — `gb200_vla.yaml` will fall through to the raw base model and
the VLA SFT will just take longer to converge.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Path or HF repo of the base Qwen2.5-VL model")
    ap.add_argument("--lora", required=True, help="Path to LoRA adapter dir (contains adapter_config.json)")
    ap.add_argument("--output", required=True, help="Output dir for the merged HF model")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"],
                    help="Output weight dtype. bf16 matches Tier-2 training default.")
    ap.add_argument("--device", default="cpu",
                    help="Where to load for the merge. 'cpu' avoids OOM on small GPUs.")
    args = ap.parse_args()

    # Resolve / validate
    base = args.base
    lora = args.lora
    out = args.output
    if not os.path.isabs(out):
        out = os.path.abspath(out)
    if not os.path.exists(lora):
        print(f"ERROR: --lora path does not exist: {lora}", file=sys.stderr)
        sys.exit(2)
    if not os.path.exists(os.path.join(lora, "adapter_config.json")):
        print(
            f"ERROR: {lora} does not contain adapter_config.json. "
            f"Pass a PEFT-style LoRA adapter directory.",
            file=sys.stderr,
        )
        sys.exit(2)

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    # Read adapter cfg to print expected base model + sanity-check
    with open(os.path.join(lora, "adapter_config.json")) as f:
        adapter_cfg = json.load(f)
    expected_base = adapter_cfg.get("base_model_name_or_path", "")
    if expected_base and expected_base != base:
        print(f"NOTE: adapter expects base={expected_base!r}; you passed {base!r}.")
        print("      Continuing (the LoRA tensors are shape-checked at load time).")

    print("=" * 64)
    print(f"  Base:    {base}")
    print(f"  LoRA:    {lora}    (r={adapter_cfg.get('r')} alpha={adapter_cfg.get('lora_alpha')})")
    print(f"  Output:  {out}")
    print(f"  dtype:   {args.dtype}    device: {args.device}")
    print("=" * 64)

    os.makedirs(out, exist_ok=True)

    print("[1/4] Loading base model...")
    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(
        base, torch_dtype=dtype, device_map={"": args.device},
    )
    print(f"  done in {time.time() - t0:.1f}s")

    print("[2/4] Loading + attaching LoRA adapter...")
    t0 = time.time()
    model = PeftModel.from_pretrained(model, lora, torch_dtype=dtype)
    print(f"  done in {time.time() - t0:.1f}s")

    print("[3/4] merge_and_unload()...")
    t0 = time.time()
    model = model.merge_and_unload()
    print(f"  done in {time.time() - t0:.1f}s")

    print(f"[4/4] Saving merged model + processor to {out}...")
    t0 = time.time()
    model.save_pretrained(out, safe_serialization=True)
    processor = AutoProcessor.from_pretrained(base)
    processor.save_pretrained(out)
    print(f"  done in {time.time() - t0:.1f}s")

    # Summary
    total = sum(
        os.path.getsize(os.path.join(out, f))
        for f in os.listdir(out)
        if f.endswith((".safetensors", ".bin"))
    )
    meta = {
        "base_model": base,
        "lora_adapter": lora,
        "lora_r": adapter_cfg.get("r"),
        "lora_alpha": adapter_cfg.get("lora_alpha"),
        "target_modules": adapter_cfg.get("target_modules"),
        "dtype": args.dtype,
        "model_size_gb": round(total / 1024 ** 3, 3),
    }
    with open(os.path.join(out, "merge_info.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print()
    print(f"DONE. Merged model: {out}  ({total / 1024 ** 3:.2f} GB)")
    print("Use as a Tier-2 warm-init by setting in your YAML:")
    print(f"  model_id: \"{out}\"")


if __name__ == "__main__":
    main()
