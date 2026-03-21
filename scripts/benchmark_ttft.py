"""Benchmark Time-to-First-Token (TTFT) for visual token compression.

Isolates prefill latency by generating only 1 token (max_new_tokens=1).
This reveals the real benefit of token compression: fewer input tokens → faster prefill.

Unlike the full benchmark (benchmark_visual_compression.py), decode time is negligible
here, so the compression speedup is clearly visible.

Usage:
    python scripts/benchmark_ttft.py \
        --config configs/gh200.yaml \
        --experiments \
            baseline=checkpoints_qwen25/default/checkpoint-46000 \
            fastervlm=checkpoints_qwen25/fastervlm_c4/final:fastervlm:4 \
            prumerge=checkpoints_qwen25/prumerge_c4/final:prumerge:4 \
            pyramiddrop=checkpoints_qwen25/pyramiddrop_c4/final:pyramiddrop:4 \
        --n 30
"""

import argparse
import gc
import json
import os
import random
import statistics
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
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        return model.base_model.model
    return model


def parse_experiment(spec):
    name, rest = spec.split("=", 1)
    parts = rest.split(":")
    lora_path = parts[0]
    method = parts[1] if len(parts) > 1 else "none"
    ratio = int(parts[2]) if len(parts) > 2 else 1
    if not os.path.isabs(lora_path):
        lora_path = os.path.join(BASE_DIR, lora_path)
    return {"name": name, "lora_path": lora_path, "method": method, "ratio": ratio}


def measure_ttft(model, processor, image_path, prompt, compress_method,
                 compress_ratio, image_token_id):
    """Measure TTFT by generating exactly 1 token."""
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
    merge_size = getattr(base.model.visual, "spatial_merge_size", 2)
    post_grid = image_grid_thw.clone()
    post_grid[:, 1] = image_grid_thw[:, 1] // merge_size
    post_grid[:, 2] = image_grid_thw[:, 2] // merge_size
    n_visual_before = int((post_grid[:, 0] * post_grid[:, 1] * post_grid[:, 2]).sum().item())

    torch.cuda.synchronize()

    # --- Vision encoder ---
    t0 = time.perf_counter()
    vis_dtype = next(base.model.visual.parameters()).dtype
    with torch.no_grad():
        vis_output = base.model.visual(pixel_values.to(vis_dtype), grid_thw=image_grid_thw)
        image_embeds = vis_output.pooler_output if hasattr(vis_output, "pooler_output") and vis_output.pooler_output is not None else vis_output.last_hidden_state
        if isinstance(image_embeds, (tuple, list)):
            image_embeds = image_embeds[0]
    torch.cuda.synchronize()
    t_vision = time.perf_counter() - t0

    # --- Compression ---
    t0 = time.perf_counter()
    if compress_method != "none":
        compressed, new_grid_thw = compress_visual_tokens(
            image_embeds, post_grid, compress_method, compress_ratio)
    else:
        compressed, new_grid_thw = image_embeds, post_grid
    torch.cuda.synchronize()
    t_compress = time.perf_counter() - t0

    n_visual_after = int((new_grid_thw[:, 0] * new_grid_thw[:, 1] * new_grid_thw[:, 2]).sum().item())

    # --- Build inputs (always use inputs_embeds path for fair comparison) ---
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

    inputs_embeds = base.model.language_model.embed_tokens(new_input_ids)
    img_mask = new_input_ids == image_token_id
    inputs_embeds[img_mask] = compressed.to(inputs_embeds.dtype)

    n_input = new_input_ids.shape[1]

    # --- Generate 1 token (= prefill + 1 decode step) ---
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        model.generate(
            input_ids=new_input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=new_attn,
            image_grid_thw=new_grid_thw,
            max_new_tokens=1,
            do_sample=False,
        )
    torch.cuda.synchronize()
    t_prefill = time.perf_counter() - t0

    t_ttft = t_vision + t_compress + t_prefill

    return {
        "n_visual_before": n_visual_before,
        "n_visual_after": n_visual_after,
        "compression_ratio": round(n_visual_before / max(n_visual_after, 1), 2),
        "n_input_tokens": n_input,
        "t_vision_ms": round(t_vision * 1000, 1),
        "t_compress_ms": round(t_compress * 1000, 1),
        "t_prefill_ms": round(t_prefill * 1000, 1),
        "t_ttft_ms": round(t_ttft * 1000, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark TTFT for visual token compression")
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiments", nargs="+", required=True,
                        help="name=lora_path[:method:ratio]")
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
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

    print(f"Loading val data from {VAL_FILE}...")
    with open(VAL_FILE, "r", encoding="utf-8") as f:
        val_data = json.load(f)
    samples = select_image_samples(val_data, n=args.n, seed=args.seed)
    print(f"  Selected {len(samples)} samples with images\n")

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    if hasattr(processor, "image_processor") and processor.image_processor is not None:
        processor.image_processor.min_pixels = min_pixels
        processor.image_processor.max_pixels = max_pixels
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")

    all_results = {}

    for exp in experiments:
        name = exp["name"]
        print(f"{'=' * 60}")
        print(f"Experiment: {name}")
        print(f"  LoRA:     {exp['lora_path']}")
        print(f"  Compress: {exp['method']} (ratio={exp['ratio']})")
        print(f"{'=' * 60}")

        print("  Loading model...")
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=compute_dtype, device_map="auto")
        model = PeftModel.from_pretrained(model, exp["lora_path"])
        model.eval()

        # Warmup
        print(f"  Warming up ({args.warmup} iterations)...")
        for w in range(args.warmup):
            _, img_path = samples[0]
            measure_ttft(model, processor, img_path, PROMPTS[0],
                         exp["method"], exp["ratio"], image_token_id)

        torch.cuda.reset_peak_memory_stats()
        results = []
        for i, (sample, img_path) in enumerate(samples):
            prompt = PROMPTS[i % len(PROMPTS)]
            r = measure_ttft(model, processor, img_path, prompt,
                             exp["method"], exp["ratio"], image_token_id)
            results.append(r)
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i+1}/{len(samples)}] tokens: {r['n_visual_before']}→{r['n_visual_after']} "
                      f"| vision={r['t_vision_ms']:.0f}ms compress={r['t_compress_ms']:.0f}ms "
                      f"prefill={r['t_prefill_ms']:.0f}ms TTFT={r['t_ttft_ms']:.0f}ms")

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
            "t_vision_ms_p50": round(statistics.median([r["t_vision_ms"] for r in results]), 1),
            "t_compress_ms_mean": round(statistics.mean([r["t_compress_ms"] for r in results]), 1),
            "t_prefill_ms_mean": round(statistics.mean([r["t_prefill_ms"] for r in results]), 1),
            "t_prefill_ms_p50": round(statistics.median([r["t_prefill_ms"] for r in results]), 1),
            "t_ttft_ms_mean": round(statistics.mean([r["t_ttft_ms"] for r in results]), 1),
            "t_ttft_ms_p50": round(statistics.median([r["t_ttft_ms"] for r in results]), 1),
            "gpu_peak_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        }
        all_results[name] = {"aggregate": agg, "samples": results}

        print(f"\n  --- {name} Summary ---")
        print(f"  Visual tokens: {agg['visual_tokens_before']} → {agg['visual_tokens_after']} ({agg['compression_ratio']}x)")
        print(f"  Input tokens:  {agg['n_input_tokens_mean']}")
        print(f"  Vision:   {agg['t_vision_ms_mean']:.1f} ms (P50: {agg['t_vision_ms_p50']:.1f})")
        print(f"  Compress: {agg['t_compress_ms_mean']:.1f} ms")
        print(f"  Prefill:  {agg['t_prefill_ms_mean']:.1f} ms (P50: {agg['t_prefill_ms_p50']:.1f})")
        print(f"  TTFT:     {agg['t_ttft_ms_mean']:.1f} ms (P50: {agg['t_ttft_ms_p50']:.1f})")

        del model
        torch.cuda.empty_cache()
        gc.collect()
        print()

    # ============ Comparison Table ============
    print(f"\n{'=' * 80}")
    print("TTFT COMPARISON TABLE")
    print(f"{'=' * 80}")
    exps = list(all_results.values())
    baseline_ttft = exps[0]["aggregate"]["t_ttft_ms_mean"] if exps else 1
    baseline_prefill = exps[0]["aggregate"]["t_prefill_ms_mean"] if exps else 1

    print(f"{'Model':<14} {'Tokens':>12} {'Input':>7} {'Vision':>8} {'Compr':>7} "
          f"{'Prefill':>9} {'TTFT':>8} {'Speedup':>8}")
    print("-" * 80)
    for e in exps:
        a = e["aggregate"]
        ttft_speedup = baseline_ttft / a["t_ttft_ms_mean"] if a["t_ttft_ms_mean"] > 0 else 0
        prefill_speedup = baseline_prefill / a["t_prefill_ms_mean"] if a["t_prefill_ms_mean"] > 0 else 0
        token_str = f"{a['visual_tokens_before']}→{a['visual_tokens_after']}"
        print(f"{a['name']:<14} {token_str:>12} {a['n_input_tokens_mean']:>7} "
              f"{a['t_vision_ms_mean']:>6.0f}ms {a['t_compress_ms_mean']:>5.1f}ms "
              f"{a['t_prefill_ms_mean']:>7.1f}ms {a['t_ttft_ms_mean']:>6.0f}ms "
              f"{ttft_speedup:>7.2f}x")

    # Prefill-only speedup
    print(f"\nPrefill-only speedup (isolating compression benefit):")
    for e in exps:
        a = e["aggregate"]
        prefill_speedup = baseline_prefill / a["t_prefill_ms_mean"] if a["t_prefill_ms_mean"] > 0 else 0
        print(f"  {a['name']:<14} prefill={a['t_prefill_ms_mean']:.1f}ms → {prefill_speedup:.2f}x")

    # Save
    output_path = args.output or os.path.join(BASE_DIR, "results", "benchmark_ttft.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    report = {
        "config": args.config,
        "n_samples": args.n,
        "warmup": args.warmup,
        "experiments": {k: v["aggregate"] for k, v in all_results.items()},
        "details": {k: v["samples"] for k, v in all_results.items()},
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
