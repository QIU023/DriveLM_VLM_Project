"""Tier-1 video VLM smoke test.

Loads a video-mode YAML config (e.g. configs/gb200_video.yaml), instantiates the
processor + Qwen2.5-VL model, pulls 1-2 samples from the new DriveLMDataset
video branch, runs a *single forward pass* + ~32-token generate to confirm the
end-to-end pipeline works.

Hard constraints:
  * NO optimizer step, NO .backward(), NO checkpoint write — read-only.
  * If `model_id` in the YAML points to a non-existent local path, fall back to
    a smaller variant; never auto-download a 32B model.

Usage:
  python scripts/smoke_test_video.py --config configs/gb200_video.yaml
  python scripts/smoke_test_video.py --config configs/gb200_video.yaml --num-samples 2
"""
import argparse
import os
import sys
import traceback

import torch
import yaml
from transformers import AutoModelForImageTextToText, AutoProcessor

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_lora import DriveLMDataset, collate_fn, load_config  # noqa: E402


# Smallest -> largest fall-back chain. We only fall *down*, never up — protects disk.
FALLBACK_MODELS = [
    "/workspace/models/Qwen2.5-VL-3B-Instruct",
    "/workspace/models/Qwen2.5-VL-7B-Instruct",
    "/workspace/models/Qwen2.5-VL-32B-Instruct",
]


def resolve_model_id(requested):
    """If requested model path exists, use it. Otherwise pick the smallest available."""
    if os.path.exists(requested) or not requested.startswith("/"):
        # HF repo id (no leading slash) is fine to leave as-is — transformers will
        # try to download, which respects HF_HOME cache.
        return requested
    print(f"[smoke] model_id '{requested}' missing on disk; searching local fall-backs...")
    for cand in FALLBACK_MODELS:
        if os.path.exists(cand):
            print(f"[smoke] using fall-back: {cand}")
            return cand
    raise FileNotFoundError(
        f"No Qwen2.5-VL weights found locally. Configured '{requested}' missing and "
        f"no fall-back in {FALLBACK_MODELS}. To proceed, download via:\n"
        f"  hf download Qwen/Qwen2.5-VL-3B-Instruct --local-dir /workspace/models/Qwen2.5-VL-3B-Instruct\n"
        f"NOTE: do NOT auto-download the 32B variant (~64 GB) without checking disk."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to video YAML config")
    ap.add_argument("--num-samples", type=int, default=1, help="Samples to run forward on")
    ap.add_argument("--gen-tokens", type=int, default=32, help="Tokens to generate")
    ap.add_argument("--data-path", default=None, help="Override data_path_video from config")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if not cfg.get("video_mode", False):
        print("ERROR: config is not in video_mode — refusing to run video smoke test.", file=sys.stderr)
        sys.exit(2)

    num_frames = cfg.get("num_frames", 4)
    video_fps = cfg.get("video_fps", 2.0)
    min_pixels = cfg.get("min_pixels", 25088)
    max_pixels = cfg.get("max_pixels", 100352)
    max_length = cfg.get("max_length", 2048)

    model_id = resolve_model_id(cfg["model_id"])
    data_path = args.data_path or cfg.get("data_path_video",
                                          f"data_processed/v1_1_video_n{num_frames}.json")
    if not os.path.isabs(data_path):
        data_path = os.path.join(_BASE_DIR, data_path)
    if not os.path.exists(data_path):
        print(f"ERROR: video data file missing: {data_path}", file=sys.stderr)
        print(f"Run: python scripts/convert_data_video.py --num-frames {num_frames}", file=sys.stderr)
        sys.exit(2)

    dtype_str = cfg.get("dtype", "bfloat16")
    compute_dtype = getattr(torch, dtype_str)

    print(f"[smoke] config       : {args.config}")
    print(f"[smoke] model_id     : {model_id}")
    print(f"[smoke] data         : {data_path}")
    print(f"[smoke] num_frames   : {num_frames}  fps={video_fps}")
    print(f"[smoke] pixels/frame : min={min_pixels}  max={max_pixels}")
    print(f"[smoke] dtype        : {dtype_str}")

    print(f"[smoke] loading processor...")
    processor = AutoProcessor.from_pretrained(model_id)
    if hasattr(processor, "image_processor") and processor.image_processor is not None:
        processor.image_processor.min_pixels = min_pixels
        processor.image_processor.max_pixels = max_pixels
    if hasattr(processor, "video_processor") and processor.video_processor is not None:
        vp = processor.video_processor
        if hasattr(vp, "size") and vp.size is not None:
            if hasattr(vp.size, "shortest_edge"):
                setattr(vp.size, "shortest_edge", min_pixels)
            if hasattr(vp.size, "longest_edge"):
                setattr(vp.size, "longest_edge", max_pixels)
        for attr, val in (("min_pixels", min_pixels), ("max_pixels", max_pixels)):
            if hasattr(vp, attr):
                setattr(vp, attr, val)
        print(f"[smoke] video processor size override -> {vp.size}")

    print(f"[smoke] loading model (this may take a minute)...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=compute_dtype, device_map="auto", attn_implementation="sdpa",
    )
    model.eval()
    print(f"[smoke] model device : {model.device}")
    print(f"[smoke] GPU memory   : {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    ds = DriveLMDataset(
        data_path, processor, max_length=max_length,
        video_mode=True, num_frames=num_frames, video_fps=video_fps,
    )
    print(f"[smoke] dataset size : {len(ds)}")
    n = min(args.num_samples, len(ds))
    if n == 0:
        print("ERROR: dataset empty.", file=sys.stderr)
        sys.exit(2)

    samples = [ds[i] for i in range(n)]
    batch = collate_fn(samples)
    print(f"[smoke] batch keys   : {sorted(k for k in batch.keys() if k != 'image_names')}")
    print(f"[smoke] input_ids    : {tuple(batch['input_ids'].shape)}")
    if "pixel_values_videos" in batch:
        print(f"[smoke] pixel_values_videos : {tuple(batch['pixel_values_videos'].shape)} "
              f"dtype={batch['pixel_values_videos'].dtype}")
    if "video_grid_thw" in batch:
        print(f"[smoke] video_grid_thw      : {batch['video_grid_thw'].tolist()}")
    if "second_per_grid_ts" in batch:
        print(f"[smoke] second_per_grid_ts  : {batch['second_per_grid_ts'].tolist()}")

    # Move tensors to model.device
    fwd_batch = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            if k in ("pixel_values_videos", "pixel_values"):
                fwd_batch[k] = v.to(model.device, dtype=compute_dtype)
            else:
                fwd_batch[k] = v.to(model.device)
        elif k == "image_names":
            continue

    # ============ Forward pass (no grad) ============
    print("[smoke] running forward pass (no grad, no backward)...")
    with torch.no_grad():
        out = model(**fwd_batch)
    print(f"[smoke] forward OK   : logits {tuple(out.logits.shape)}  loss={out.loss.item() if out.loss is not None else 'N/A'}")

    # ============ Generate ============
    print(f"[smoke] generating {args.gen_tokens} tokens...")
    # For generation we need a prompt-shaped input (drop labels, drop trailing assistant text).
    # The simplest reliable smoke is to run generate on the dataset's full input_ids — this
    # extends past the answer, but it confirms the autoregressive path works end-to-end.
    gen_inputs = {k: v for k, v in fwd_batch.items() if k != "labels"}
    with torch.no_grad():
        gen = model.generate(
            **gen_inputs,
            max_new_tokens=args.gen_tokens,
            do_sample=False,
        )
    new_tokens = gen[:, fwd_batch["input_ids"].shape[1]:]
    decoded = processor.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    for i, t in enumerate(decoded):
        print(f"[smoke] gen[{i}]: {t!r}")

    print("[smoke] DONE — pipeline works end-to-end. NO optimizer.step / NO .backward / NO checkpoint write.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
