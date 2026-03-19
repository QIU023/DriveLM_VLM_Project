"""Full evaluation on DriveLM val set with resume support.

Runs LoRA (or base) model on val.json, computes per-category metrics,
saves results incrementally so it can resume after interruption.

Usage:
  # Eval LoRA checkpoint on all val data (resume-safe)
  python scripts/eval_full.py --config configs/4070ti.yaml --lora checkpoints/checkpoint-46000

  # Eval base model
  python scripts/eval_full.py --config configs/4070ti.yaml --no-lora

  # Limit samples per category (faster, still representative)
  python scripts/eval_full.py --config configs/4070ti.yaml --lora checkpoints/checkpoint-46000 --max-per-cat 200

  # Resume interrupted run (auto-detected from output file)
  python scripts/eval_full.py --config configs/4070ti.yaml --lora checkpoints/checkpoint-46000 --resume
"""
import argparse
import json
import os
import time
import collections
import yaml
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VAL_FILE = os.path.join(BASE_DIR, "data_processed", "val.json")


def load_model(cfg, lora_path=None):
    """Load base model with optional quantization and LoRA."""
    model_id = cfg["model_id"]
    quantize = cfg.get("quantize", False)
    dtype_str = cfg.get("dtype", "bfloat16")
    compute_dtype = getattr(torch, dtype_str)
    min_pixels = cfg.get("min_pixels", 256 * 28 * 28)
    max_pixels = cfg.get("max_pixels", 512 * 28 * 28)

    quant_bits = cfg.get("quant_bits", 4)
    print(f"Loading {model_id} | quantize={quantize} | bits={quant_bits} | dtype={dtype_str}")
    load_kwargs = {"device_map": "auto"}
    if quantize and quant_bits == 4:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=cfg.get("double_quant", True),
            bnb_4bit_quant_type=cfg.get("quant_type", "nf4"),
            bnb_4bit_compute_dtype=compute_dtype,
        )
        load_kwargs["quantization_config"] = bnb_config
    elif quantize and quant_bits == 8:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        load_kwargs["quantization_config"] = bnb_config
    else:
        load_kwargs["torch_dtype"] = compute_dtype

    model = AutoModelForImageTextToText.from_pretrained(model_id, **load_kwargs)
    processor = AutoProcessor.from_pretrained(model_id)
    if hasattr(processor, "image_processor") and processor.image_processor is not None:
        processor.image_processor.min_pixels = min_pixels
        processor.image_processor.max_pixels = max_pixels

    tag = "Base"
    if lora_path:
        print(f"Loading LoRA from {lora_path}...")
        model = PeftModel.from_pretrained(model, lora_path)
        tag = f"LoRA({os.path.basename(lora_path)})"

    model.eval()
    gpu_mem = torch.cuda.memory_allocated() / 1024**3
    print(f"Model loaded [{tag}]. GPU: {gpu_mem:.1f} GB\n")
    return model, processor, tag


def extract_parts(sample):
    """Extract image path, question, ground truth, system prompt."""
    messages = sample["messages"]
    image_path = question = ground_truth = system_prompt = None
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


def run_inference(model, processor, image_path, question, system_prompt, max_tokens=512):
    """Run single-sample inference, return answer string and timing."""
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    msgs.append({
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": question}],
    })

    image = Image.open(image_path).convert("RGB")
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    elapsed = time.time() - t0

    new_tokens = output_ids[:, inputs["input_ids"].shape[1]:]
    answer = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
    return answer, new_tokens.shape[1], elapsed


def compute_metrics(results):
    """Compute per-category and overall metrics."""
    by_cat = collections.defaultdict(lambda: {"total": 0, "exact": 0, "tokens": 0, "time": 0.0})

    for r in results:
        cat = r["category"]
        by_cat[cat]["total"] += 1
        by_cat[cat]["tokens"] += r["num_tokens"]
        by_cat[cat]["time"] += r["elapsed"]
        if r["prediction"].strip().lower() == r["ground_truth"].strip().lower():
            by_cat[cat]["exact"] += 1

    # Build summary
    summary = {}
    total_exact = total_count = total_tokens = total_time = 0
    for cat in sorted(by_cat.keys()):
        s = by_cat[cat]
        acc = s["exact"] / s["total"] * 100
        avg_tok = s["tokens"] / s["total"]
        avg_time = s["time"] / s["total"]
        summary[cat] = {
            "n": s["total"], "exact_match": s["exact"], "accuracy": round(acc, 1),
            "avg_tokens": round(avg_tok, 1), "avg_time_s": round(avg_time, 2),
        }
        total_exact += s["exact"]
        total_count += s["total"]
        total_tokens += s["tokens"]
        total_time += s["time"]

    summary["overall"] = {
        "n": total_count, "exact_match": total_exact,
        "accuracy": round(total_exact / max(total_count, 1) * 100, 1),
        "avg_tokens": round(total_tokens / max(total_count, 1), 1),
        "avg_time_s": round(total_time / max(total_count, 1), 2),
        "total_time_min": round(total_time / 60, 1),
    }
    return summary


def print_summary(summary, tag):
    """Pretty-print evaluation summary."""
    print("\n" + "=" * 70)
    print(f"EVALUATION SUMMARY — {tag}")
    print("=" * 70)
    print(f"  {'Category':<14} {'N':>6} {'Exact':>6} {'Acc%':>7} {'AvgTok':>7} {'AvgTime':>8}")
    print(f"  {'-'*14} {'-'*6} {'-'*6} {'-'*7} {'-'*7} {'-'*8}")
    for cat in sorted(k for k in summary if k != "overall"):
        s = summary[cat]
        print(f"  {cat:<14} {s['n']:>6} {s['exact_match']:>6} {s['accuracy']:>6.1f}% {s['avg_tokens']:>7.1f} {s['avg_time_s']:>7.2f}s")
    s = summary["overall"]
    print(f"  {'-'*14} {'-'*6} {'-'*6} {'-'*7} {'-'*7} {'-'*8}")
    print(f"  {'OVERALL':<14} {s['n']:>6} {s['exact_match']:>6} {s['accuracy']:>6.1f}% {s['avg_tokens']:>7.1f} {s['avg_time_s']:>7.2f}s")
    print(f"\n  Total time: {s['total_time_min']:.1f} min")
    print(f"  GPU memory: {torch.cuda.memory_allocated()/1024**3:.1f} GB")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Full DriveLM evaluation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--lora", type=str, default=None)
    parser.add_argument("--no-lora", action="store_true")
    parser.add_argument("--max-per-cat", type=int, default=None,
                        help="Max samples per category (None=all)")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--save-every", type=int, default=50, help="Save results every N samples")
    args = parser.parse_args()

    # Config
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Determine output path
    if args.output:
        out_path = args.output
    else:
        lora_name = os.path.basename(args.lora) if args.lora else "base"
        suffix = f"_max{args.max_per_cat}" if args.max_per_cat else "_full"
        out_path = os.path.join(BASE_DIR, f"eval_results_{lora_name}{suffix}.json")

    # Load existing results if resuming
    completed_ids = set()
    results = []
    if args.resume and os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            saved = json.load(f)
        results = saved.get("results", [])
        completed_ids = {r["sample_id"] for r in results}
        print(f"Resuming: {len(completed_ids)} samples already done")

    # Load val data
    print(f"Loading val data from {VAL_FILE}...")
    with open(VAL_FILE, encoding="utf-8") as f:
        val_data = json.load(f)

    # Assign stable IDs and group by category
    by_cat = collections.defaultdict(list)
    for i, s in enumerate(val_data):
        s["_id"] = i
        by_cat[s["metadata"]["category"]].append(s)

    # Select samples
    samples = []
    for cat in sorted(by_cat.keys()):
        pool = by_cat[cat]
        if args.max_per_cat and len(pool) > args.max_per_cat:
            # Deterministic sampling
            step = len(pool) / args.max_per_cat
            pool = [pool[int(i * step)] for i in range(args.max_per_cat)]
        samples.extend(pool)

    total = len(samples)
    remaining = [s for s in samples if s["_id"] not in completed_ids]
    print(f"Total: {total} | Already done: {total - len(remaining)} | Remaining: {len(remaining)}\n")

    if not remaining:
        print("All samples already evaluated!")
        with open(out_path, encoding="utf-8") as f:
            saved = json.load(f)
        print_summary(saved["summary"], saved.get("tag", ""))
        return

    # Load model
    lora_path = None if args.no_lora else (args.lora or os.path.join(BASE_DIR, "checkpoints_qwen25", "final"))
    model, processor, tag = load_model(cfg, lora_path)

    # Run evaluation
    skipped = 0
    t_start = time.time()
    for idx, sample in enumerate(remaining):
        sid = sample["_id"]
        image_path, question, ground_truth, system_prompt = extract_parts(sample)
        cat = sample["metadata"]["category"]

        if not image_path or not os.path.exists(image_path):
            skipped += 1
            continue

        try:
            answer, num_tokens, elapsed = run_inference(
                model, processor, image_path, question, system_prompt, args.max_tokens
            )
        except Exception as e:
            print(f"  ERROR sample {sid}: {e}")
            skipped += 1
            continue

        results.append({
            "sample_id": sid,
            "category": cat,
            "question": question,
            "ground_truth": ground_truth,
            "prediction": answer,
            "num_tokens": num_tokens,
            "elapsed": round(elapsed, 3),
        })

        done = len(results)
        total_elapsed = time.time() - t_start
        avg_per_sample = total_elapsed / (idx + 1)
        eta_min = avg_per_sample * (len(remaining) - idx - 1) / 60
        exact = 1 if answer.strip().lower() == ground_truth.strip().lower() else 0

        # Progress line
        if done % 10 == 0 or done == 1:
            print(f"  [{done}/{total}] cat={cat:<12s} exact={exact} tok={num_tokens:>3} "
                  f"t={elapsed:.1f}s | ETA: {eta_min:.0f}min")

        # Periodic save
        if done % args.save_every == 0:
            summary = compute_metrics(results)
            payload = {"tag": tag, "config": os.path.basename(args.config),
                       "summary": summary, "results": results}
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)

    # Final save
    summary = compute_metrics(results)
    payload = {
        "tag": tag,
        "config": os.path.basename(args.config),
        "lora": args.lora,
        "total_samples": len(results),
        "skipped": skipped,
        "summary": summary,
        "results": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print_summary(summary, tag)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
