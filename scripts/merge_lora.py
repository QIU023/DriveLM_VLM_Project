"""
LoRA Merge Script: merge LoRA adapter into base model and export full weights.

Usage:
    python scripts/merge_lora.py \
        --lora checkpoints/checkpoint-46000 \
        --output models/qwen25vl-3b-drivelm-merged \
        [--base Qwen/Qwen2.5-VL-3B-Instruct] \
        [--dtype float16]

Runs on CPU to avoid VRAM constraints. Output is a standard HuggingFace model
directory that can be loaded by vLLM, llama.cpp, AutoAWQ, AutoGPTQ, etc.
"""

import argparse
import os
import sys
import time
import json

import torch
from peft import PeftModel
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--lora", required=True, help="Path to LoRA checkpoint dir")
    parser.add_argument("--output", required=True, help="Output directory for merged model")
    parser.add_argument("--base", default=None, help="Base model name/path (auto-detected from adapter_config.json)")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"],
                        help="Output dtype (default: float16 for widest compatibility)")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lora_path = os.path.join(project_root, args.lora) if not os.path.isabs(args.lora) else args.lora
    output_path = os.path.join(project_root, args.output) if not os.path.isabs(args.output) else args.output

    # Auto-detect base model from adapter config
    adapter_config_path = os.path.join(lora_path, "adapter_config.json")
    if not os.path.exists(adapter_config_path):
        print(f"ERROR: adapter_config.json not found in {lora_path}")
        sys.exit(1)

    with open(adapter_config_path, "r", encoding="utf-8") as f:
        adapter_config = json.load(f)

    base_model_name = args.base or adapter_config.get("base_model_name_or_path")
    if not base_model_name:
        print("ERROR: Cannot determine base model. Specify --base explicitly.")
        sys.exit(1)

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    target_dtype = dtype_map[args.dtype]

    print(f"Base model:  {base_model_name}")
    print(f"LoRA path:   {lora_path}")
    print(f"Output path: {output_path}")
    print(f"Output dtype: {args.dtype}")
    print(f"LoRA config: r={adapter_config.get('r')}, alpha={adapter_config.get('lora_alpha')}")
    print(f"Target modules: {adapter_config.get('target_modules')}")
    print()

    # Step 1: Load base model on CPU
    print("[1/5] Loading base model on CPU...")
    t0 = time.time()
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_name,
        torch_dtype=target_dtype,
        device_map="cpu",
    )
    print(f"  Done in {time.time() - t0:.1f}s")

    # Step 2: Load LoRA adapter
    print("[2/5] Loading LoRA adapter...")
    t0 = time.time()
    model = PeftModel.from_pretrained(base_model, lora_path, torch_dtype=target_dtype)
    print(f"  Done in {time.time() - t0:.1f}s")

    # Step 3: Merge and unload
    print("[3/5] Merging LoRA weights into base model...")
    t0 = time.time()
    model = model.merge_and_unload()
    print(f"  Done in {time.time() - t0:.1f}s")

    # Step 4: Save merged model
    print(f"[4/5] Saving merged model to {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    t0 = time.time()
    model.save_pretrained(output_path, safe_serialization=True)
    print(f"  Done in {time.time() - t0:.1f}s")

    # Step 5: Save processor/tokenizer
    print("[5/5] Saving processor (tokenizer + image processor)...")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(base_model_name, trust_remote_code=True)
    processor.save_pretrained(output_path)
    print(f"  Done in {time.time() - t0:.1f}s")

    # Summary
    total_size = sum(
        os.path.getsize(os.path.join(output_path, f))
        for f in os.listdir(output_path)
        if f.endswith((".safetensors", ".bin"))
    )
    print()
    print("=" * 60)
    print(f"Merge complete!")
    print(f"  Output: {output_path}")
    print(f"  Model size: {total_size / 1024**3:.2f} GB")
    print(f"  Files: {os.listdir(output_path)}")
    print("=" * 60)

    # Save merge metadata
    meta = {
        "base_model": base_model_name,
        "lora_checkpoint": lora_path,
        "lora_r": adapter_config.get("r"),
        "lora_alpha": adapter_config.get("lora_alpha"),
        "target_modules": adapter_config.get("target_modules"),
        "output_dtype": args.dtype,
        "model_size_gb": round(total_size / 1024**3, 2),
    }
    with open(os.path.join(output_path, "merge_info.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  Metadata saved to merge_info.json")


if __name__ == "__main__":
    main()
