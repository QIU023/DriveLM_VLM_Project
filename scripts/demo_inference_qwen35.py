"""Demo inference: run LoRA-finetuned Qwen3.5-4B on 10 DriveLM val samples.

Loads the base model + LoRA adapter, runs inference on 10 diverse samples
(across perception/prediction/planning/behavior categories), and prints
model output vs ground truth side-by-side.

Usage:
  python demo_inference.py                  # default: 10 samples
  python demo_inference.py --n 20           # more samples
  python demo_inference.py --no-lora        # base model only (compare)
"""
import argparse
import json
import os
import random
import time
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from PIL import Image

# ============ Config ============
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_ID = "Qwen/Qwen3.5-4B"
LORA_PATH = os.path.join(BASE_DIR, "checkpoints", "final")
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
    """Build chat messages for the model (image passed separately to processor)."""
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
    parser.add_argument("--n", type=int, default=10, help="Number of samples")
    parser.add_argument("--no-lora", action="store_true", help="Base model only")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ============ Load model ============
    print("Loading base model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    if not args.no_lora:
        print(f"Loading LoRA adapter from {LORA_PATH}...")
        model = PeftModel.from_pretrained(model, LORA_PATH)
        model.eval()
        tag = "LoRA"
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

        # Two-step: template → text, then processor handles tokenization + image
        # Disable thinking: DriveLM answers are short factual responses
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
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
        raw_output = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
        num_tokens = new_tokens.shape[1]

        # Extract answer after </think> if present (Qwen3.5 thinking mode)
        if "</think>" in raw_output:
            answer = raw_output.split("</think>")[-1].strip()
        else:
            answer = raw_output

        results.append({
            "category": cat,
            "question": question,
            "ground_truth": ground_truth,
            "prediction": answer,
            "raw_output": raw_output,
        })

        # Print result
        print(f"{'='*70}")
        print(f"  [{i+1}/{args.n}] Category: {cat}")
        print(f"  Image: ...{image_path[-55:]}")
        print(f"  Q: {question}")
        print(f"  GT:     {ground_truth}")
        print(f"  Answer: {answer}")
        if "</think>" in raw_output:
            # Show abbreviated thinking
            think_part = raw_output.split("</think>")[0].strip()
            preview = think_part[:120].replace("\n", " ") + "..."
            print(f"  (think: {preview})")
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
