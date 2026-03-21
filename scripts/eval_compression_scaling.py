"""Evaluate accuracy vs compression ratio scaling curve.

Uses a SINGLE checkpoint (baseline, trained without compression) and applies
FasterVLM compression at inference time with varying ratios.
Shows how accuracy degrades as we compress more aggressively.

Usage:
    python scripts/eval_compression_scaling.py \
        --config configs/gh200.yaml \
        --lora checkpoints_qwen25/default/checkpoint-46000 \
        --ratios 1,2,4,8,16 \
        --samples-per-cat 100
"""

import argparse
import gc
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


def get_base_model(model):
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        return model.base_model.model
    return model


def select_samples_by_category(data, per_cat=100, seed=42):
    """Select samples evenly across categories, ensuring images exist."""
    rng = random.Random(seed)
    by_cat = {}
    for d in data:
        cat = d.get("metadata", {}).get("category", "unknown")
        img_path = None
        for msg in d["messages"]:
            if isinstance(msg.get("content"), list):
                for part in msg["content"]:
                    if part.get("type") == "image":
                        img_path = part["image"].replace("file://", "")
                        break
                break
        if img_path and os.path.exists(img_path):
            by_cat.setdefault(cat, []).append(d)

    selected = {}
    for cat in sorted(by_cat.keys()):
        pool = by_cat[cat]
        rng.shuffle(pool)
        selected[cat] = pool[:per_cat]
    return selected


def extract_parts(sample):
    """Extract image path, question, ground truth from a sample."""
    image_path = question = ground_truth = system_prompt = None
    for msg in sample["messages"]:
        if msg["role"] == "system":
            system_prompt = msg["content"]
        elif msg["role"] == "user":
            for part in msg["content"]:
                if part.get("type") == "image":
                    image_path = part["image"].replace("file://", "")
                elif part.get("type") == "text":
                    question = part["text"]
        elif msg["role"] == "assistant":
            ground_truth = msg["content"]
    return image_path, question, ground_truth, system_prompt


def run_inference(model, processor, image_path, question, system_prompt,
                  compress_method, compress_ratio, image_token_id, max_tokens):
    """Run inference with optional compression. Returns prediction and token count."""
    from visual_compress import compress_visual_tokens

    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    msgs.append({"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": question},
    ]})

    image = Image.open(image_path).convert("RGB")
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

    input_ids = inputs["input_ids"]
    image_grid_thw = inputs["image_grid_thw"]

    base = get_base_model(model)
    merge_size = getattr(base.model.visual, "spatial_merge_size", 2)
    post_grid = image_grid_thw.clone()
    post_grid[:, 1] = image_grid_thw[:, 1] // merge_size
    post_grid[:, 2] = image_grid_thw[:, 2] // merge_size
    n_visual_before = int((post_grid[:, 0] * post_grid[:, 1] * post_grid[:, 2]).sum().item())

    if compress_method != "none" and compress_ratio > 1:
        # Manual vision encoder + compression
        vis_dtype = next(base.model.visual.parameters()).dtype
        with torch.no_grad():
            vis_out = base.model.visual(inputs["pixel_values"].to(vis_dtype), grid_thw=image_grid_thw)
            image_embeds = vis_out.pooler_output if hasattr(vis_out, "pooler_output") and vis_out.pooler_output is not None else vis_out.last_hidden_state
            if isinstance(image_embeds, (tuple, list)):
                image_embeds = image_embeds[0]

        compressed, new_grid_thw = compress_visual_tokens(
            image_embeds, post_grid, compress_method, compress_ratio)
        n_visual_after = int((new_grid_thw[:, 0] * new_grid_thw[:, 1] * new_grid_thw[:, 2]).sum().item())

        # Adjust input_ids
        img_positions = (input_ids[0] == image_token_id).nonzero(as_tuple=True)[0]
        if len(img_positions) > n_visual_after:
            remove_pos = img_positions[n_visual_after:]
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

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=new_input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=new_attn,
                image_grid_thw=new_grid_thw,
                max_new_tokens=max_tokens,
                do_sample=False,
            )
        n_input = new_input_ids.shape[1]
    else:
        # Standard forward
        n_visual_after = n_visual_before
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
            )
        n_input = input_ids.shape[1]

    new_tokens = output_ids[:, n_input:]
    prediction = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()

    return prediction, n_visual_before, n_visual_after


def main():
    parser = argparse.ArgumentParser(description="Compression ratio vs accuracy scaling curve")
    parser.add_argument("--config", required=True)
    parser.add_argument("--lora", required=True, help="Path to baseline LoRA checkpoint")
    parser.add_argument("--method", default="fastervlm", help="Compression method")
    parser.add_argument("--ratios", default="1,2,4,8,16", help="Comma-separated ratios")
    parser.add_argument("--samples-per-cat", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    ratios = [int(r) for r in args.ratios.split(",")]

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
    compute_dtype = getattr(torch, cfg.get("dtype", "bfloat16"))
    min_pixels = cfg.get("min_pixels", 256 * 28 * 28)
    max_pixels = cfg.get("max_pixels", 512 * 28 * 28)

    lora_path = args.lora
    if not os.path.isabs(lora_path):
        lora_path = os.path.join(BASE_DIR, lora_path)

    # Load data
    print(f"Loading val data from {VAL_FILE}...")
    with open(VAL_FILE, "r", encoding="utf-8") as f:
        val_data = json.load(f)
    by_cat = select_samples_by_category(val_data, per_cat=args.samples_per_cat, seed=args.seed)
    total = sum(len(v) for v in by_cat.values())
    print(f"  Categories: {list(by_cat.keys())}")
    print(f"  Total samples: {total}\n")

    # Load model once
    print(f"Loading {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    if hasattr(processor, "image_processor") and processor.image_processor is not None:
        processor.image_processor.min_pixels = min_pixels
        processor.image_processor.max_pixels = max_pixels
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")

    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=compute_dtype, device_map="auto")
    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    print(f"  Model loaded. GPU: {torch.cuda.memory_allocated()/1024**3:.1f} GB\n")

    all_ratio_results = {}

    for ratio in ratios:
        method = "none" if ratio <= 1 else args.method
        print(f"{'=' * 60}")
        print(f"Ratio: {ratio}x  (method: {method})")
        print(f"{'=' * 60}")

        cat_results = {}
        total_correct = 0
        total_count = 0
        total_vis_tokens = 0

        for cat in sorted(by_cat.keys()):
            samples = by_cat[cat]
            correct = 0
            vis_tokens_sum = 0

            for i, sample in enumerate(samples):
                image_path, question, ground_truth, system_prompt = extract_parts(sample)
                if not image_path or not os.path.exists(image_path):
                    continue

                pred, n_before, n_after = run_inference(
                    model, processor, image_path, question, system_prompt,
                    method, ratio, image_token_id, args.max_tokens)

                is_match = pred.strip().lower() == ground_truth.strip().lower()
                if is_match:
                    correct += 1
                vis_tokens_sum += n_after

                if (i + 1) % 50 == 0:
                    print(f"  [{cat}] {i+1}/{len(samples)} ... acc={correct/(i+1)*100:.1f}%")

            n = len(samples)
            acc = correct / max(n, 1) * 100
            avg_tokens = vis_tokens_sum / max(n, 1)
            cat_results[cat] = {"n": n, "correct": correct, "accuracy": round(acc, 1), "avg_visual_tokens": round(avg_tokens)}
            total_correct += correct
            total_count += n
            total_vis_tokens += vis_tokens_sum
            print(f"  {cat}: {correct}/{n} = {acc:.1f}%")

        overall_acc = total_correct / max(total_count, 1) * 100
        avg_vis = total_vis_tokens / max(total_count, 1)
        cat_results["overall"] = {
            "n": total_count, "correct": total_correct,
            "accuracy": round(overall_acc, 1), "avg_visual_tokens": round(avg_vis),
        }
        all_ratio_results[str(ratio)] = cat_results
        print(f"  OVERALL: {total_correct}/{total_count} = {overall_acc:.1f}% | avg tokens: {avg_vis:.0f}\n")

    # ============ Summary Table ============
    print(f"\n{'=' * 70}")
    print(f"SCALING CURVE: {args.method} compression on baseline checkpoint")
    print(f"{'=' * 70}")
    cats = sorted([c for c in list(by_cat.keys())])
    header = f"{'Ratio':>6} {'Tokens':>8}"
    for c in cats:
        header += f" {c:>12}"
    header += f" {'OVERALL':>10}"
    print(header)
    print("-" * len(header))

    for ratio in ratios:
        r = all_ratio_results[str(ratio)]
        tokens = r["overall"]["avg_visual_tokens"]
        row = f"{ratio:>5}x {tokens:>7}"
        for c in cats:
            row += f" {r[c]['accuracy']:>11.1f}%"
        row += f" {r['overall']['accuracy']:>9.1f}%"
        print(row)

    # Save
    output_path = args.output or os.path.join(BASE_DIR, "results", "compression_scaling.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    report = {
        "config": args.config,
        "lora": args.lora,
        "method": args.method,
        "samples_per_cat": args.samples_per_cat,
        "ratios": {k: v for k, v in all_ratio_results.items()},
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
