"""Batch merge multiple LoRA adapters into separate full models.

Each LoRA is merged with the base model independently and saved as a
standalone HuggingFace model directory (loadable by vLLM, llama.cpp, etc.).

Runs on CPU — no GPU required. Base model files are cached after first load.

Usage:
    # Merge all three LoRAs
    python scripts/batch_merge.py \
        baseline=checkpoints_qwen25/checkpoint-46000 \
        fastervlm_c4=checkpoints_qwen25/fastervlm_c4/checkpoint-best \
        prumerge_c4=checkpoints_qwen25/prumerge_c4/checkpoint-best

    # Skip already-merged outputs (default) or force re-merge
    python scripts/batch_merge.py baseline=... --force

    # Custom base model or dtype
    python scripts/batch_merge.py ... --base Qwen/Qwen2.5-VL-3B-Instruct --dtype float16

Output:
    models/qwen25vl-3b-drivelm-{name}-merged/
"""

import argparse
import gc
import json
import os
import sys
import time

import torch
from peft import PeftModel
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def merge_one(base_model_name, lora_path, output_path, target_dtype):
    """Load base model + LoRA, merge, save full model."""
    # Load adapter config
    adapter_cfg_path = os.path.join(lora_path, "adapter_config.json")
    if not os.path.exists(adapter_cfg_path):
        print(f"  ERROR: adapter_config.json not found in {lora_path}")
        return False

    with open(adapter_cfg_path, "r", encoding="utf-8") as f:
        adapter_cfg = json.load(f)

    detected_base = adapter_cfg.get("base_model_name_or_path", base_model_name)
    print(f"  Base: {detected_base}")
    print(f"  LoRA: r={adapter_cfg.get('r')}, alpha={adapter_cfg.get('lora_alpha')}")

    # [1/4] Load base model on CPU
    t0 = time.time()
    print(f"  [1/4] Loading base model on CPU...")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        detected_base, torch_dtype=target_dtype, device_map="cpu",
    )
    print(f"         {time.time() - t0:.1f}s")

    # [2/4] Load LoRA adapter and merge
    t0 = time.time()
    print(f"  [2/4] Loading LoRA + merge_and_unload...")
    model = PeftModel.from_pretrained(base_model, lora_path, torch_dtype=target_dtype)
    model = model.merge_and_unload()
    print(f"         {time.time() - t0:.1f}s")

    # [3/4] Save merged model
    t0 = time.time()
    os.makedirs(output_path, exist_ok=True)
    print(f"  [3/4] Saving merged model to {output_path}...")
    model.save_pretrained(output_path, safe_serialization=True)
    print(f"         {time.time() - t0:.1f}s")

    # [4/4] Save processor (tokenizer + image processor)
    t0 = time.time()
    print(f"  [4/4] Saving processor...")
    processor = AutoProcessor.from_pretrained(detected_base, trust_remote_code=True)
    processor.save_pretrained(output_path)
    print(f"         {time.time() - t0:.1f}s")

    # Model size
    total_size = sum(
        os.path.getsize(os.path.join(output_path, f))
        for f in os.listdir(output_path)
        if f.endswith((".safetensors", ".bin"))
    )

    # Save merge metadata
    meta = {
        "base_model": detected_base,
        "lora_checkpoint": lora_path,
        "lora_r": adapter_cfg.get("r"),
        "lora_alpha": adapter_cfg.get("lora_alpha"),
        "target_modules": adapter_cfg.get("target_modules"),
        "output_dtype": str(target_dtype).split(".")[-1],
        "model_size_gb": round(total_size / 1024**3, 2),
    }
    with open(os.path.join(output_path, "merge_info.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"  Done! {total_size / 1024**3:.2f} GB")

    # Free memory for next merge
    del model, base_model
    gc.collect()
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Batch merge LoRA adapters into base model",
        epilog="Example: python scripts/batch_merge.py baseline=checkpoints_qwen25/checkpoint-46000 fastervlm_c4=checkpoints_qwen25/fastervlm_c4/checkpoint-best",
    )
    parser.add_argument("merges", nargs="+", metavar="NAME=LORA_PATH",
                        help="name=lora_checkpoint_path pairs")
    parser.add_argument("--base", default="Qwen/Qwen2.5-VL-3B-Instruct",
                        help="Base model (default: auto-detect from adapter_config.json)")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"],
                        help="Output dtype (float16 for widest compatibility)")
    parser.add_argument("--output-dir", default="models",
                        help="Parent output directory (default: models/)")
    parser.add_argument("--force", action="store_true",
                        help="Re-merge even if output already exists")
    args = parser.parse_args()

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    target_dtype = dtype_map[args.dtype]

    # Parse name=path pairs
    jobs = []
    for spec in args.merges:
        if "=" not in spec:
            print(f"ERROR: expected NAME=LORA_PATH, got: {spec}")
            sys.exit(1)
        name, lora = spec.split("=", 1)
        lora_abs = os.path.join(BASE_DIR, lora) if not os.path.isabs(lora) else lora
        output = os.path.join(BASE_DIR, args.output_dir, f"qwen25vl-3b-drivelm-{name}-merged")
        jobs.append((name, lora_abs, output))

    # Summary
    print(f"{'=' * 60}")
    print(f"Batch LoRA merge: {len(jobs)} model(s)")
    print(f"Base: {args.base} | dtype: {args.dtype}")
    print(f"{'=' * 60}")
    for name, lora, out in jobs:
        exists = os.path.exists(os.path.join(out, "model.safetensors"))
        tag = " [EXISTS]" if exists else ""
        print(f"  {name}: {lora}")
        print(f"    -> {out}{tag}")
    print()

    # Merge each
    success, skipped = 0, 0
    for name, lora_path, output_path in jobs:
        print(f"\n{'=' * 60}")
        print(f"[{name}]")

        if not args.force and os.path.exists(os.path.join(output_path, "model.safetensors")):
            print(f"  SKIP: already exists (use --force to overwrite)")
            skipped += 1
            continue

        if not os.path.isdir(lora_path):
            print(f"  ERROR: LoRA path not found: {lora_path}")
            continue

        if merge_one(args.base, lora_path, output_path, target_dtype):
            success += 1

    # Final summary
    print(f"\n{'=' * 60}")
    print(f"Complete: {success} merged, {skipped} skipped, {len(jobs) - success - skipped} failed")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
