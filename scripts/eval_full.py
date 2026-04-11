"""Full evaluation on DriveLM val set with resume support.

Runs LoRA (or base) model on val.json, computes per-category metrics,
saves results incrementally so it can resume after interruption.
Supports visual token compression methods (fastervlm, prumerge, etc.)

Usage:
  # Eval LoRA checkpoint on all val data (resume-safe)
  python scripts/eval_full.py --config configs/4070ti.yaml --lora checkpoints/checkpoint-46000

  # Eval base model
  python scripts/eval_full.py --config configs/4070ti.yaml --no-lora

  # Eval with compression (reads from config)
  python scripts/eval_full.py --config configs/fastervlm_c4.yaml --lora checkpoints_qwen25/fastervlm_c4/final --max-per-cat 100

  # Limit samples per category (faster, still representative)
  python scripts/eval_full.py --config configs/4070ti.yaml --lora checkpoints/checkpoint-46000 --max-per-cat 200

  # Resume interrupted run (auto-detected from output file)
  python scripts/eval_full.py --config configs/4070ti.yaml --lora checkpoints/checkpoint-46000 --resume
"""
import argparse
import json
import os
import re
import sys
import time
import collections
import yaml
import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from PIL import Image
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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


def get_base_model(model):
    """Unwrap PEFT to get the original Qwen2.5-VL model."""
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        return model.base_model.model
    return model


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


def run_inference_compressed(model, processor, image_path, question, system_prompt,
                             compress_method, compress_ratio, image_token_id, max_tokens=512):
    """Run inference with visual token compression applied."""
    from visual_compress import compress_visual_tokens

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
        base = get_base_model(model)

        # 1. Run vision encoder
        pixel_values = inputs["pixel_values"]
        image_grid_thw = inputs["image_grid_thw"]
        vis_dtype = next(base.model.visual.parameters()).dtype
        vis_out = base.model.visual(pixel_values.to(vis_dtype), grid_thw=image_grid_thw)
        image_embeds = vis_out.pooler_output if hasattr(vis_out, "pooler_output") else vis_out
        if isinstance(image_embeds, (tuple, list)):
            image_embeds = image_embeds[0]

        # 2. Compress visual tokens
        raw_grid = image_grid_thw
        merge_size = getattr(base.model.visual, "spatial_merge_size", 2)
        post_grid = raw_grid.clone()
        post_grid[:, 1] = raw_grid[:, 1] // merge_size
        post_grid[:, 2] = raw_grid[:, 2] // merge_size
        compressed, new_grid_thw = compress_visual_tokens(
            image_embeds, post_grid, compress_method, compress_ratio
        )

        # 3. Remove excess image placeholder tokens
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        orig_count = int((post_grid[:, 0] * post_grid[:, 1] * post_grid[:, 2]).sum())
        new_count = int((new_grid_thw[:, 0] * new_grid_thw[:, 1] * new_grid_thw[:, 2]).sum())

        img_pos = (input_ids[0] == image_token_id).nonzero(as_tuple=True)[0]
        n_remove = len(img_pos) - new_count

        if n_remove > 0:
            remove_pos = img_pos[new_count:]
            keep = torch.ones(input_ids.shape[1], dtype=torch.bool, device=input_ids.device)
            keep[remove_pos] = False
            input_ids = input_ids[:, keep]
            attention_mask = attention_mask[:, keep]

        # 4. Build inputs_embeds with compressed visual tokens
        inputs_embeds = base.model.language_model.embed_tokens(input_ids).clone()
        img_mask = input_ids[0] == image_token_id
        inputs_embeds[0, img_mask] = compressed.to(inputs_embeds.dtype)

        # 5. Generate
        output_ids = model.generate(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            image_grid_thw=new_grid_thw,
            max_new_tokens=max_tokens,
            do_sample=False,
        )
    elapsed = time.time() - t0

    new_tokens = output_ids[:, input_ids.shape[1]:]
    answer = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
    return answer, new_tokens.shape[1], elapsed


def infer_tag(category, question):
    """Infer official DriveLM evaluation tag from category and question pattern.

    Tag mapping (from challenge/test_eval.json):
      0 = accuracy (exact match)  — behavior, closed-choice perception/prediction
      1 = GPT/language eval       — planning (open reasoning)
      2 = language (BLEU/ROUGE)   — perception open-ended descriptions
      3 = match (coordinate F1)   — prediction with <cX,CAM,...> coordinates
    """
    q = question.strip().lower() if question else ""
    if category == "behavior":
        return 0
    elif category == "planning":
        return 1
    elif category == "perception":
        # Closed-choice: "Please select" or "What is the moving status"
        if "please select" in q or "what is the moving status" in q or "what is the observed status" in q:
            return 0
        return 2
    elif category == "prediction":
        # Coordinate-heavy: "What object should the ego vehicle notice first..."
        if "notice first" in q or "notice second" in q or "notice third" in q:
            return 3
        return 0
    return 0


def compute_bleu_rouge(pred, gt):
    """Compute BLEU-4 and ROUGE-L for a single (prediction, ground_truth) pair."""
    smooth = SmoothingFunction().method1
    pred_tokens = pred.lower().split()
    gt_tokens = gt.lower().split()
    bleu = sentence_bleu([gt_tokens], pred_tokens, smoothing_function=smooth)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge = scorer.score(gt, pred)["rougeL"].fmeasure
    return bleu, rouge


def compute_match_f1(pred, gt, threshold=16.0):
    """Compute coordinate-based F1 (official DriveLM match metric).

    Extracts all float-pairs from pred/GT, matches by L1 distance < threshold.
    """
    pred_nums = re.findall(r'\d+\.\d+', pred)
    gt_nums = re.findall(r'\d+\.\d+', gt)
    if len(pred_nums) % 2 != 0:
        pred_nums = pred_nums[:-1]
    if len(gt_nums) % 2 != 0:
        gt_nums = gt_nums[:-1]
    if not gt_nums:
        return 1.0 if not pred_nums else 0.0

    pred_pts = np.array([float(x) for x in pred_nums]).reshape(-1, 2)
    gt_pts = np.array([float(x) for x in gt_nums]).reshape(-1, 2)
    n_gt = len(gt_pts)
    gt_remaining = list(range(n_gt))

    tp = 0
    for p in pred_pts:
        best_dist = float("inf")
        best_idx = -1
        for i in gt_remaining:
            d = np.sum(np.abs(p - gt_pts[i]))
            if d < best_dist:
                best_dist = d
                best_idx = i
        if best_dist < threshold and best_idx >= 0:
            tp += 1
            gt_remaining.remove(best_idx)

    fp = len(pred_pts) - tp
    fn = n_gt - tp
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return f1


def compute_metrics(results):
    """Compute per-category and overall metrics using official DriveLM multi-metric eval.

    Metric assignment by tag (inferred from category + question):
      Tag 0 (accuracy):  exact string match          — behavior, closed-choice
      Tag 1 (language):  BLEU + ROUGE_L average       — planning (open reasoning)
      Tag 2 (language):  BLEU + ROUGE_L average       — perception open descriptions
      Tag 3 (match):     coordinate F1                — prediction with coordinates

    Final score = weighted average across all metrics.
    """
    by_cat = collections.defaultdict(lambda: {
        "total": 0, "tokens": 0, "time": 0.0,
        # Tag-specific accumulators
        "exact_n": 0, "exact_correct": 0,       # tag 0
        "lang_n": 0, "bleu_sum": 0.0, "rouge_sum": 0.0,  # tag 1, 2
        "match_n": 0, "f1_sum": 0.0,            # tag 3
    })

    for r in results:
        cat = r["category"]
        by_cat[cat]["total"] += 1
        by_cat[cat]["tokens"] += r["num_tokens"]
        by_cat[cat]["time"] += r["elapsed"]

        tag = infer_tag(cat, r.get("question", ""))
        pred = r["prediction"].strip()
        gt = r["ground_truth"].strip()

        if tag == 0:
            by_cat[cat]["exact_n"] += 1
            if pred.lower() == gt.lower():
                by_cat[cat]["exact_correct"] += 1
        elif tag in (1, 2):
            by_cat[cat]["lang_n"] += 1
            bleu, rouge = compute_bleu_rouge(pred, gt)
            by_cat[cat]["bleu_sum"] += bleu
            by_cat[cat]["rouge_sum"] += rouge
        elif tag == 3:
            by_cat[cat]["match_n"] += 1
            f1 = compute_match_f1(pred, gt)
            by_cat[cat]["f1_sum"] += f1

    # Build summary
    summary = {}
    totals = {"n": 0, "exact_n": 0, "exact_correct": 0,
              "lang_n": 0, "bleu_sum": 0.0, "rouge_sum": 0.0,
              "match_n": 0, "f1_sum": 0.0, "tokens": 0, "time": 0.0}

    for cat in sorted(by_cat.keys()):
        s = by_cat[cat]
        n = s["total"]
        cat_summary = {"n": n}

        # Exact match score (tag 0)
        if s["exact_n"] > 0:
            cat_summary["exact_match"] = s["exact_correct"]
            cat_summary["exact_match_n"] = s["exact_n"]
            cat_summary["accuracy"] = round(s["exact_correct"] / s["exact_n"] * 100, 1)

        # Language score (tag 1, 2)
        if s["lang_n"] > 0:
            cat_summary["lang_n"] = s["lang_n"]
            cat_summary["bleu"] = round(s["bleu_sum"] / s["lang_n"] * 100, 1)
            cat_summary["rouge_l"] = round(s["rouge_sum"] / s["lang_n"] * 100, 1)
            cat_summary["language_score"] = round(
                (s["bleu_sum"] + s["rouge_sum"]) / (2 * s["lang_n"]) * 100, 1)

        # Match F1 score (tag 3)
        if s["match_n"] > 0:
            cat_summary["match_n"] = s["match_n"]
            cat_summary["match_f1"] = round(s["f1_sum"] / s["match_n"] * 100, 1)

        # Combined category score: weighted avg of available metrics
        scores, weights = [], []
        if s["exact_n"] > 0:
            scores.append(s["exact_correct"] / s["exact_n"])
            weights.append(s["exact_n"])
        if s["lang_n"] > 0:
            scores.append((s["bleu_sum"] + s["rouge_sum"]) / (2 * s["lang_n"]))
            weights.append(s["lang_n"])
        if s["match_n"] > 0:
            scores.append(s["f1_sum"] / s["match_n"])
            weights.append(s["match_n"])
        cat_summary["combined_score"] = round(
            sum(sc * w for sc, w in zip(scores, weights)) / max(sum(weights), 1) * 100, 1)

        cat_summary["avg_tokens"] = round(s["tokens"] / n, 1)
        cat_summary["avg_time_s"] = round(s["time"] / n, 2)
        summary[cat] = cat_summary

        for k in totals:
            if k in s:
                totals[k] += s[k]
        totals["n"] += n

    # Overall
    overall_scores, overall_weights = [], []
    if totals["exact_n"] > 0:
        overall_scores.append(totals["exact_correct"] / totals["exact_n"])
        overall_weights.append(totals["exact_n"])
    if totals["lang_n"] > 0:
        overall_scores.append(
            (totals["bleu_sum"] + totals["rouge_sum"]) / (2 * totals["lang_n"]))
        overall_weights.append(totals["lang_n"])
    if totals["match_n"] > 0:
        overall_scores.append(totals["f1_sum"] / totals["match_n"])
        overall_weights.append(totals["match_n"])

    combined = sum(sc * w for sc, w in zip(overall_scores, overall_weights)) / max(sum(overall_weights), 1) * 100

    summary["overall"] = {
        "n": totals["n"],
        "combined_score": round(combined, 1),
        "exact_match_accuracy": round(totals["exact_correct"] / max(totals["exact_n"], 1) * 100, 1),
        "language_score": round(
            (totals["bleu_sum"] + totals["rouge_sum"]) / max(2 * totals["lang_n"], 1) * 100, 1) if totals["lang_n"] else None,
        "match_f1": round(totals["f1_sum"] / max(totals["match_n"], 1) * 100, 1) if totals["match_n"] else None,
        "avg_tokens": round(totals["tokens"] / max(totals["n"], 1), 1),
        "avg_time_s": round(totals["time"] / max(totals["n"], 1), 2),
        "total_time_min": round(totals["time"] / 60, 1),
    }
    return summary


def print_summary(summary, tag):
    """Pretty-print evaluation summary with multi-metric results."""
    print("\n" + "=" * 85)
    print(f"EVALUATION SUMMARY — {tag}")
    print("=" * 85)
    print(f"  {'Category':<14} {'N':>5} {'ExactAcc':>9} {'BLEU':>7} {'ROUGE':>7} {'MatchF1':>8} {'Combined':>9}")
    print(f"  {'-'*14} {'-'*5} {'-'*9} {'-'*7} {'-'*7} {'-'*8} {'-'*9}")
    for cat in sorted(k for k in summary if k != "overall"):
        s = summary[cat]
        acc = f"{s['accuracy']:.1f}%" if "accuracy" in s else "   —  "
        bleu = f"{s['bleu']:.1f}%" if "bleu" in s else "   — "
        rouge = f"{s['rouge_l']:.1f}%" if "rouge_l" in s else "   — "
        f1 = f"{s['match_f1']:.1f}%" if "match_f1" in s else "   —  "
        comb = f"{s['combined_score']:.1f}%"
        print(f"  {cat:<14} {s['n']:>5} {acc:>9} {bleu:>7} {rouge:>7} {f1:>8} {comb:>9}")
    s = summary["overall"]
    print(f"  {'-'*14} {'-'*5} {'-'*9} {'-'*7} {'-'*7} {'-'*8} {'-'*9}")
    acc = f"{s['exact_match_accuracy']:.1f}%" if s.get("exact_match_accuracy") else "   —  "
    lang = f"{s['language_score']:.1f}%" if s.get("language_score") else "   — "
    f1 = f"{s['match_f1']:.1f}%" if s.get("match_f1") else "   —  "
    comb = f"{s['combined_score']:.1f}%"
    print(f"  {'OVERALL':<14} {s['n']:>5} {acc:>9} {lang:>14} {f1:>8} {comb:>9}")
    print(f"\n  Total time: {s['total_time_min']:.1f} min")
    print(f"  GPU memory: {torch.cuda.memory_allocated()/1024**3:.1f} GB")
    print("=" * 85)


def load_config(config_path):
    """Load YAML config with optional base_config inheritance."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    if "base_config" in cfg:
        base_path = cfg.pop("base_config")
        if not os.path.isabs(base_path):
            base_path = os.path.join(os.path.dirname(config_path), base_path)
        base_cfg = load_config(base_path)
        base_cfg.update(cfg)
        cfg = base_cfg
    return cfg


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

    # Config — support base_config inheritance
    cfg = load_config(args.config)

    # Compression settings
    compress_method = cfg.get("compress_method", "none")
    compress_ratio = cfg.get("compress_ratio", 1)
    experiment = cfg.get("experiment", "default")
    use_compression = compress_method != "none" and compress_ratio > 1
    print(f"Experiment: {experiment} | Compression: {compress_method} ratio={compress_ratio}")

    # Determine output path
    if args.output:
        out_path = args.output
    else:
        # Use experiment name from lora path: checkpoints_qwen25/<experiment>/final → <experiment>
        if args.lora:
            lora_parts = os.path.normpath(args.lora).split(os.sep)
            # Find the part after checkpoints_qwen25
            if "checkpoints_qwen25" in lora_parts:
                idx = lora_parts.index("checkpoints_qwen25")
                exp_name = lora_parts[idx + 1] if idx + 1 < len(lora_parts) else os.path.basename(args.lora)
            else:
                exp_name = os.path.basename(os.path.dirname(args.lora))
        else:
            exp_name = "base"
        suffix = f"_max{args.max_per_cat}" if args.max_per_cat else "_full"
        eval_dir = os.path.join(BASE_DIR, "eval_results")
        os.makedirs(eval_dir, exist_ok=True)
        out_path = os.path.join(eval_dir, f"{exp_name}{suffix}.json")

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

    # Get image token id for compression
    image_token_id = None
    if use_compression:
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        print(f"Using compression: {compress_method} ratio={compress_ratio} | image_token_id={image_token_id}")

    # Update tag with compression info
    if use_compression:
        tag = f"{tag} [{compress_method} {compress_ratio}x]"

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
            if use_compression:
                answer, num_tokens, elapsed = run_inference_compressed(
                    model, processor, image_path, question, system_prompt,
                    compress_method, compress_ratio, image_token_id, args.max_tokens
                )
            else:
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
        tag = infer_tag(cat, question)
        if tag == 0:
            score_str = f"exact={'Y' if answer.strip().lower() == ground_truth.strip().lower() else 'N'}"
        elif tag in (1, 2):
            _b, _r = compute_bleu_rouge(answer.strip(), ground_truth.strip())
            score_str = f"bleu={_b:.2f} rouge={_r:.2f}"
        elif tag == 3:
            _f1 = compute_match_f1(answer.strip(), ground_truth.strip())
            score_str = f"f1={_f1:.2f}"
        else:
            score_str = ""

        # Progress line
        if done % 10 == 0 or done == 1:
            print(f"  [{done}/{total}] cat={cat:<12s} {score_str} tok={num_tokens:>3} "
                  f"t={elapsed:.1f}s | ETA: {eta_min:.0f}min")

        # Periodic save
        if done % args.save_every == 0:
            summary = compute_metrics(results)
            payload = {"tag": tag, "config": os.path.basename(args.config),
                       "compress_method": compress_method, "compress_ratio": compress_ratio,
                       "summary": summary, "results": results}
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)

    # Final save
    summary = compute_metrics(results)
    payload = {
        "tag": tag,
        "config": os.path.basename(args.config),
        "experiment": experiment,
        "compress_method": compress_method,
        "compress_ratio": compress_ratio,
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
