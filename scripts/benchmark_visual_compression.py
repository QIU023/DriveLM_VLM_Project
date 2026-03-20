"""Benchmark visual token compression impact on inference latency.

Measures the REAL benefit of token compression: fewer visual tokens → faster
prefill → lower TTFT for image inputs. Compares baseline (full tokens) vs
compressed (fastervlm / prumerge) on the same images.

Runs on GH200 (bf16, full precision) with base model + LoRA adapter.

Metrics per sample:
  - n_visual_tokens: before and after compression
  - t_vision_encoder: time to run ViT
  - t_compress: time for token compression (0 for baseline)
  - t_prefill: time from inputs ready to first output token
  - t_decode: total decode time
  - t_total: end-to-end inference time
  - n_output_tokens: number of generated tokens
  - gpu_memory: peak GPU memory (GB)

Usage (on GH200):
    python scripts/benchmark_visual_compression.py \
        --config configs/gh200.yaml \
        --experiments baseline=checkpoints/baseline/checkpoint-46000 \
                      fastervlm=checkpoints/fastervlm/final:fastervlm:4 \
                      prumerge=checkpoints/prumerge/final:prumerge:4 \
        --n 30 --max-tokens 128

    Format: name=lora_path[:compress_method:compress_ratio]
    If compress_method is omitted, defaults to "none" (no compression).
"""

import argparse
import json
import os
import random
import sys
import time
import yaml

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

VAL_FILE = os.path.join(BASE_DIR, "data_processed", "val.json")

PROMPTS = [
    "What are the important objects in the current scene? Those objects will be considered for the future planning.",
    "What is the moving status of the object directly ahead of the ego vehicle?",
    "Is there a traffic light visible? What is its current status?",
    "What is the safe action for the ego vehicle to take in this situation?",
    "Describe the weather and road conditions visible in this driving scene.",
]


def select_image_samples(data, n=30, seed=42):
    """Pick samples that have images, spread across categories."""
    rng = random.Random(seed)
    with_images = []
    for d in data:
        for msg in d["messages"]:
            if isinstance(msg.get("content"), list):
                for part in msg["content"]:
                    if part.get("type") == "image":
                        img_path = part["image"].replace("file://", "")
                        if os.path.exists(img_path):
                            with_images.append((d, img_path))
                            break
                break
    rng.shuffle(with_images)
    return with_images[:n]


def get_base_model(model):
    """Unwrap PEFT to get the original Qwen2.5-VL model."""
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        return model.base_model.model
    return model


def inference_with_timing(model, processor, image_path, prompt, compress_method,
                          compress_ratio, max_tokens, image_token_id):
    """Run single inference and return detailed timing breakdown."""
    from visual_compress import compress_visual_tokens

    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": prompt},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

    input_ids = inputs["input_ids"]
    pixel_values = inputs["pixel_values"]
    image_grid_thw = inputs["image_grid_thw"]

    base = get_base_model(model)
    n_visual_before = int((image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]).sum().item())

    torch.cuda.synchronize()

    # --- Vision encoder ---
    t0 = time.perf_counter()
    vis_dtype = next(base.visual.parameters()).dtype
    with torch.no_grad():
        image_embeds = base.visual(pixel_values.to(vis_dtype), grid_thw=image_grid_thw)
    torch.cuda.synchronize()
    t_vision = time.perf_counter() - t0

    # --- Compression ---
    t0 = time.perf_counter()
    if compress_method != "none":
        compressed, new_grid_thw = compress_visual_tokens(
            image_embeds, image_grid_thw, compress_method, compress_ratio)
    else:
        compressed, new_grid_thw = image_embeds, image_grid_thw
    torch.cuda.synchronize()
    t_compress = time.perf_counter() - t0

    n_visual_after = int((new_grid_thw[:, 0] * new_grid_thw[:, 1] * new_grid_thw[:, 2]).sum().item())

    # --- Build inputs_embeds with compressed tokens ---
    if compress_method != "none":
        # Adjust input_ids: remove excess image placeholders
        orig_count = n_visual_before
        new_count = n_visual_after
        img_positions = (input_ids[0] == image_token_id).nonzero(as_tuple=True)[0]
        if len(img_positions) > new_count:
            remove_pos = img_positions[new_count:]
            keep = torch.ones(input_ids.shape[1], dtype=torch.bool, device=input_ids.device)
            keep[remove_pos] = False
            new_input_ids = input_ids[:, keep]
            new_attn = inputs["attention_mask"][:, keep]
        else:
            new_input_ids = input_ids
            new_attn = inputs["attention_mask"]

        inputs_embeds = base.model.embed_tokens(new_input_ids)
        img_mask = new_input_ids == image_token_id
        inputs_embeds[img_mask] = compressed.to(inputs_embeds.dtype)

        gen_inputs = {
            "input_ids": new_input_ids,
            "inputs_embeds": inputs_embeds,
            "attention_mask": new_attn,
            "image_grid_thw": new_grid_thw,
        }
    else:
        # No compression: standard forward with pixel_values
        gen_inputs = {
            "input_ids": input_ids,
            "attention_mask": inputs["attention_mask"],
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }

    # --- Prefill + Decode ---
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **gen_inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
        )
    torch.cuda.synchronize()
    t_generate = time.perf_counter() - t0

    n_input = gen_inputs["input_ids"].shape[1]
    n_output = output_ids.shape[1] - n_input
    # rough split: prefill ≈ proportional to input tokens
    # actual split not measurable without hooks, so report total generate time
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3

    return {
        "n_visual_before": n_visual_before,
        "n_visual_after": n_visual_after,
        "compression_ratio": round(n_visual_before / max(n_visual_after, 1), 2),
        "n_input_tokens": n_input,
        "n_output_tokens": n_output,
        "t_vision_ms": round(t_vision * 1000, 1),
        "t_compress_ms": round(t_compress * 1000, 1),
        "t_generate_ms": round(t_generate * 1000, 1),
        "t_total_ms": round((t_vision + t_compress + t_generate) * 1000, 1),
        "decode_tps": round(n_output / t_generate, 1) if t_generate > 0 else 0,
        "gpu_peak_gb": round(peak_mem, 2),
    }


def parse_experiment(spec):
    """Parse 'name=lora_path[:method:ratio]' into dict."""
    name, rest = spec.split("=", 1)
    parts = rest.split(":")
    lora_path = parts[0]
    method = parts[1] if len(parts) > 1 else "none"
    ratio = int(parts[2]) if len(parts) > 2 else 1
    if not os.path.isabs(lora_path):
        lora_path = os.path.join(BASE_DIR, lora_path)
    return {"name": name, "lora_path": lora_path, "method": method, "ratio": ratio}


def main():
    parser = argparse.ArgumentParser(description="Benchmark visual token compression latency")
    parser.add_argument("--config", required=True, help="YAML config for model loading")
    parser.add_argument("--experiments", nargs="+", required=True,
                        help="name=lora_path[:method:ratio] (e.g. baseline=ckpt/baseline/ckpt-46000)")
    parser.add_argument("--n", type=int, default=30, help="Number of image samples")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    # Support base_config inheritance
    if "base_config" in cfg:
        base_path = cfg.pop("base_config")
        if not os.path.isabs(base_path):
            base_path = os.path.join(os.path.dirname(args.config), base_path)
        with open(base_path, "r") as f:
            base_cfg = yaml.safe_load(f)
        base_cfg.update(cfg)
        cfg = base_cfg

    model_id = cfg["model_id"]
    dtype_str = cfg.get("dtype", "bfloat16")
    compute_dtype = getattr(torch, dtype_str)
    min_pixels = cfg.get("min_pixels", 256 * 28 * 28)
    max_pixels = cfg.get("max_pixels", 512 * 28 * 28)

    experiments = [parse_experiment(s) for s in args.experiments]

    # Load val data with images
    print(f"Loading val data from {VAL_FILE}...")
    with open(VAL_FILE, "r", encoding="utf-8") as f:
        val_data = json.load(f)
    samples = select_image_samples(val_data, n=args.n, seed=args.seed)
    print(f"  Selected {len(samples)} samples with images\n")

    # Get image_token_id
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    if hasattr(processor, "image_processor") and processor.image_processor is not None:
        processor.image_processor.min_pixels = min_pixels
        processor.image_processor.max_pixels = max_pixels
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    print(f"  image_token_id = {image_token_id}\n")

    all_results = {}

    for exp in experiments:
        name = exp["name"]
        print(f"{'=' * 60}")
        print(f"Experiment: {name}")
        print(f"  LoRA:     {exp['lora_path']}")
        print(f"  Compress: {exp['method']} (ratio={exp['ratio']})")
        print(f"{'=' * 60}")

        # Load base model fresh for each experiment
        print("  Loading base model...")
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=compute_dtype, device_map="auto")
        print("  Loading LoRA adapter...")
        model = PeftModel.from_pretrained(model, exp["lora_path"])
        model.eval()

        gpu_mem = torch.cuda.memory_allocated() / 1024**3
        print(f"  Model loaded. GPU: {gpu_mem:.1f} GB")

        results = []
        for i, (sample, img_path) in enumerate(samples):
            prompt = PROMPTS[i % len(PROMPTS)]
            torch.cuda.reset_peak_memory_stats()

            r = inference_with_timing(
                model, processor, img_path, prompt,
                exp["method"], exp["ratio"], args.max_tokens, image_token_id)

            results.append(r)
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i+1}/{len(samples)}] visual: {r['n_visual_before']}→{r['n_visual_after']} "
                      f"| vision={r['t_vision_ms']:.0f}ms compress={r['t_compress_ms']:.0f}ms "
                      f"gen={r['t_generate_ms']:.0f}ms total={r['t_total_ms']:.0f}ms "
                      f"| {r['decode_tps']:.0f} tok/s")

        # Aggregate
        import statistics
        agg = {
            "name": name,
            "method": exp["method"],
            "ratio": exp["ratio"],
            "n_samples": len(results),
            "visual_tokens_before": round(statistics.mean([r["n_visual_before"] for r in results])),
            "visual_tokens_after": round(statistics.mean([r["n_visual_after"] for r in results])),
            "compression_ratio": round(statistics.mean([r["compression_ratio"] for r in results]), 2),
            "n_input_tokens_mean": round(statistics.mean([r["n_input_tokens"] for r in results])),
            "t_vision_ms_mean": round(statistics.mean([r["t_vision_ms"] for r in results]), 1),
            "t_compress_ms_mean": round(statistics.mean([r["t_compress_ms"] for r in results]), 1),
            "t_generate_ms_mean": round(statistics.mean([r["t_generate_ms"] for r in results]), 1),
            "t_total_ms_mean": round(statistics.mean([r["t_total_ms"] for r in results]), 1),
            "t_total_ms_p50": round(statistics.median([r["t_total_ms"] for r in results]), 1),
            "decode_tps_mean": round(statistics.mean([r["decode_tps"] for r in results]), 1),
            "gpu_peak_gb": round(max([r["gpu_peak_gb"] for r in results]), 2),
        }
        all_results[name] = {"aggregate": agg, "samples": results}

        print(f"\n  --- {name} Summary ---")
        print(f"  Visual tokens: {agg['visual_tokens_before']} → {agg['visual_tokens_after']} "
              f"({agg['compression_ratio']}x)")
        print(f"  Input tokens (mean): {agg['n_input_tokens_mean']}")
        print(f"  Vision encoder: {agg['t_vision_ms_mean']:.0f} ms")
        print(f"  Compression:    {agg['t_compress_ms_mean']:.0f} ms")
        print(f"  Generate:       {agg['t_generate_ms_mean']:.0f} ms")
        print(f"  Total:          {agg['t_total_ms_mean']:.0f} ms (P50: {agg['t_total_ms_p50']:.0f} ms)")
        print(f"  Decode speed:   {agg['decode_tps_mean']:.0f} tok/s")
        print(f"  GPU peak:       {agg['gpu_peak_gb']:.2f} GB")

        # Cleanup
        del model
        torch.cuda.empty_cache()
        import gc; gc.collect()
        print()

    # ============ Comparison Table ============
    print(f"\n{'=' * 70}")
    print("COMPARISON TABLE")
    print(f"{'=' * 70}")
    exps = list(all_results.values())
    baseline_total = exps[0]["aggregate"]["t_total_ms_mean"] if exps else 1

    print(f"{'Model':<14} {'Tokens':>12} {'Input':>7} {'Vision':>8} {'Compr':>7} "
          f"{'Generate':>9} {'Total':>8} {'Speedup':>8} {'GPU':>6}")
    print("-" * 85)
    for e in exps:
        a = e["aggregate"]
        speedup = baseline_total / a["t_total_ms_mean"] if a["t_total_ms_mean"] > 0 else 0
        token_str = f"{a['visual_tokens_before']}→{a['visual_tokens_after']}"
        print(f"{a['name']:<14} {token_str:>12} {a['n_input_tokens_mean']:>7} "
              f"{a['t_vision_ms_mean']:>7.0f}ms {a['t_compress_ms_mean']:>5.0f}ms "
              f"{a['t_generate_ms_mean']:>7.0f}ms {a['t_total_ms_mean']:>6.0f}ms "
              f"{speedup:>7.2f}x {a['gpu_peak_gb']:>5.1f}G")

    # Save
    output_path = args.output or os.path.join(BASE_DIR, "benchmark_visual_compression.json")
    report = {
        "config": args.config,
        "n_samples": args.n,
        "max_tokens": args.max_tokens,
        "experiments": {k: v["aggregate"] for k, v in all_results.items()},
        "details": {k: v["samples"] for k, v in all_results.items()},
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
