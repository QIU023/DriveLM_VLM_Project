"""Demo inference: run LoRA-finetuned Qwen2.5-VL-3B on DriveLM val samples.

Loads the base model + LoRA adapter, runs inference on diverse samples
(across perception/prediction/planning/behavior categories), and prints
model output vs ground truth side-by-side.

Usage:
  python demo_inference.py --config configs/gh200.yaml                         # final checkpoint
  python demo_inference.py --config configs/gh200.yaml --lora checkpoints_qwen25/checkpoint-500
  python demo_inference.py --config configs/gh200.yaml --n 20                  # more samples
  python demo_inference.py --config configs/gh200.yaml --no-lora               # base model only
"""
import argparse
import json
import os
import random
import time
import yaml
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from PIL import Image

# ============ Config ============
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VAL_FILE = os.path.join(BASE_DIR, "data_processed", "val.json")


def select_diverse_samples(data, n=10, seed=42):
    """Pick samples evenly across DriveLM categories."""
    rng = random.Random(seed)
    by_cat = {}
    for d in data:
        cat = d["metadata"]["category"]
        by_cat.setdefault(cat, []).append(d)

    selected = []
    per_cat = max(1, n // len(by_cat))
    for cat in sorted(by_cat.keys()):
        pool = by_cat[cat]
        rng.shuffle(pool)
        selected.extend(pool[:per_cat])

    rng.shuffle(selected)
    return selected[:n]


def extract_parts(sample):
    """Extract image path, question, ground truth from a DriveLM sample."""
    messages = sample["messages"]
    image_path = None
    question = None
    ground_truth = None
    system_prompt = None

    for msg in messages:
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


def build_messages(question, system_prompt):
    """Build chat messages for the model."""
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    msgs.append({
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": question},
        ],
    })
    return msgs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--lora", type=str, default=None,
                        help="Path to LoRA checkpoint (default: checkpoints_qwen25/final)")
    parser.add_argument("--n", type=int, default=10, help="Number of samples")
    parser.add_argument("--no-lora", action="store_true", help="Base model only")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ============ Load config ============
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    model_id = cfg["model_id"]
    quantize = cfg.get("quantize", False)
    dtype_str = cfg.get("dtype", "bfloat16")
    compute_dtype = getattr(torch, dtype_str)
    min_pixels = cfg.get("min_pixels", 256 * 28 * 28)
    max_pixels = cfg.get("max_pixels", 512 * 28 * 28)

    lora_path = args.lora if args.lora else os.path.join(BASE_DIR, "checkpoints_qwen25", "final")

    # ============ Load model ============
    print(f"Loading {model_id} | quantize={quantize} | dtype={dtype_str}")
    load_kwargs = {"device_map": "auto"}
    if quantize:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=(cfg.get("quant_bits", 4) == 4),
            load_in_8bit=(cfg.get("quant_bits", 4) == 8),
            bnb_4bit_use_double_quant=cfg.get("double_quant", True),
            bnb_4bit_quant_type=cfg.get("quant_type", "nf4"),
            bnb_4bit_compute_dtype=compute_dtype,
        )
        load_kwargs["quantization_config"] = bnb_config
    else:
        load_kwargs["torch_dtype"] = compute_dtype

    model = AutoModelForImageTextToText.from_pretrained(model_id, **load_kwargs)
    processor = AutoProcessor.from_pretrained(model_id)
    if hasattr(processor, "image_processor") and processor.image_processor is not None:
        processor.image_processor.min_pixels = min_pixels
        processor.image_processor.max_pixels = max_pixels

    if not args.no_lora:
        print(f"Loading LoRA adapter from {lora_path}...")
        model = PeftModel.from_pretrained(model, lora_path)
        model.eval()
        tag = f"LoRA ({os.path.basename(lora_path)})"
    else:
        model.eval()
        tag = "Base"

    gpu_mem = torch.cuda.memory_allocated() / 1024**3
    print(f"Model loaded [{tag}]. GPU: {gpu_mem:.1f} GB\n")

    # ============ Load val data ============
    print(f"Loading val data from {VAL_FILE}...")
    with open(VAL_FILE) as f:
        val_data = json.load(f)
    samples = select_diverse_samples(val_data, n=args.n, seed=args.seed)
    print(f"Selected {len(samples)} samples across categories\n")

    # ============ Run inference ============
    results = []
    for i, sample in enumerate(samples):
        image_path, question, ground_truth, system_prompt = extract_parts(sample)
        cat = sample["metadata"]["category"]

        if not os.path.exists(image_path):
            print(f"[{i+1}/{args.n}] SKIP - image not found: {image_path}")
            continue

        messages = build_messages(question, system_prompt)
        image = Image.open(image_path).convert("RGB")

        # Qwen2.5-VL: no thinking mode
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text], images=[image], return_tensors="pt",
        ).to(model.device)

        t0 = time.time()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                do_sample=False,
            )
        elapsed = time.time() - t0

        # Decode only the new tokens
        new_tokens = output_ids[:, inputs["input_ids"].shape[1]:]
        answer = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
        num_tokens = new_tokens.shape[1]

        results.append({
            "category": cat,
            "question": question,
            "ground_truth": ground_truth,
            "prediction": answer,
        })

        # Print result
        print(f"{'='*70}")
        print(f"  [{i+1}/{args.n}] Category: {cat}")
        print(f"  Image: ...{image_path[-55:]}")
        print(f"  Q: {question}")
        print(f"  GT:     {ground_truth}")
        print(f"  Answer: {answer}")
        print(f"  ({num_tokens} tokens, {elapsed:.1f}s)")
        print()

    # ============ Summary ============
    print("=" * 70)
    print(f"SUMMARY ({tag} model, {len(results)} samples)")
    print("=" * 70)

    # Simple exact-match accuracy by category
    by_cat = {}
    for r in results:
        cat = r["category"]
        by_cat.setdefault(cat, {"total": 0, "exact": 0})
        by_cat[cat]["total"] += 1
        if r["prediction"].strip().lower() == r["ground_truth"].strip().lower():
            by_cat[cat]["exact"] += 1

    total_exact = 0
    total_count = 0
    for cat in sorted(by_cat.keys()):
        s = by_cat[cat]
        acc = s["exact"] / s["total"] * 100
        print(f"  {cat:12s}: {s['exact']}/{s['total']} exact match ({acc:.0f}%)")
        total_exact += s["exact"]
        total_count += s["total"]

    overall = total_exact / max(total_count, 1) * 100
    print(f"  {'overall':12s}: {total_exact}/{total_count} exact match ({overall:.0f}%)")
    print()

    # Save results to JSON
    out_path = os.path.join(BASE_DIR, "demo_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
