"""Tier-2 video-VLA smoke test.

Loads a VLA YAML config (e.g. configs/gb200_vla.yaml), instantiates the
processor + Qwen2.5-VL model, pulls 1 sample from a VLA-augmented dataset
(`v1_1_video_n4_with_traj.json`), runs:

  1. A single forward pass with the trajectory tokens. Confirms loss is finite.
  2. A single greedy generate of 16 new tokens. Decodes the output via
     `trajectory_tokenizer.TrajectoryTokenizer.decode` and prints the resulting
     xy waypoints.

Hard constraints:
  * NO optimizer step, NO .backward(), NO checkpoint write.
  * If `model_id` in the YAML doesn't exist on disk, fall back to the smallest
    available local Qwen2.5-VL weights (see FALLBACK_MODELS); never auto-DL 32B.
  * Run on the merged Tier-1 LoRA if present (config `model_id`); otherwise on
    the raw base. Both produce a sensible loss; only the merged warm-init is
    expected to emit *meaningful* trajectory tokens on the first generate.
"""
from __future__ import annotations

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
from trajectory_tokenizer import (  # noqa: E402
    TrajectoryTokenizer,
    TrajectoryTokenizerConfig,
    register_with_tokenizer,
)


FALLBACK_MODELS = [
    "/workspace/models/Qwen2.5-VL-3B-drivelm-merged",
    "/workspace/models/Qwen2.5-VL-3B-Instruct",
    "/workspace/models/Qwen2.5-VL-7B-Instruct",
]


def resolve_model_id(requested: str) -> str:
    if os.path.exists(requested) or not requested.startswith("/"):
        return requested
    print(f"[smoke-vla] model_id '{requested}' missing; trying fall-backs...")
    for cand in FALLBACK_MODELS:
        if os.path.exists(cand):
            print(f"[smoke-vla] using fall-back: {cand}")
            return cand
    raise FileNotFoundError(
        f"No Qwen2.5-VL weights found. Configured '{requested}' missing; "
        f"checked fall-backs {FALLBACK_MODELS}."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to VLA YAML config")
    ap.add_argument("--gen-tokens", type=int, default=16,
                    help="Tokens to generate (default 16 — enough for 6 waypoints + end)")
    ap.add_argument("--data-path", default=None,
                    help="Override data_path_vla from config")
    ap.add_argument("--sample-idx", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if not cfg.get("vla_mode", False):
        print("ERROR: config is not in vla_mode — refusing to run VLA smoke.", file=sys.stderr)
        sys.exit(2)
    if not cfg.get("video_mode", False):
        print("ERROR: VLA smoke currently expects video_mode=true.", file=sys.stderr)
        sys.exit(2)

    num_frames = cfg.get("num_frames", 4)
    video_fps = cfg.get("video_fps", 2.0)
    min_pixels = cfg.get("min_pixels", 25088)
    max_pixels = cfg.get("max_pixels", 100352)
    max_length = cfg.get("max_length", 2560)
    vla_loss_mode = cfg.get("vla_loss_mode", "answer_and_traj")

    model_id = resolve_model_id(cfg["model_id"])
    data_path = args.data_path or cfg.get(
        "data_path_vla", f"data_processed/v1_1_video_n{num_frames}_with_traj.json"
    )
    if not os.path.isabs(data_path):
        data_path = os.path.join(_BASE_DIR, data_path)
    if not os.path.exists(data_path):
        print(f"ERROR: VLA data file missing: {data_path}", file=sys.stderr)
        print(f"Run: python scripts/extract_ego_trajectory.py --input data_processed/v1_1_video_n{num_frames}.json",
              file=sys.stderr)
        sys.exit(2)

    dtype_str = cfg.get("dtype", "bfloat16")
    compute_dtype = getattr(torch, dtype_str)

    print(f"[smoke-vla] config       : {args.config}")
    print(f"[smoke-vla] model_id     : {model_id}")
    print(f"[smoke-vla] data         : {data_path}")
    print(f"[smoke-vla] num_frames   : {num_frames}  fps={video_fps}")
    print(f"[smoke-vla] vla_loss_mode: {vla_loss_mode}")

    # --- processor (no token registration needed — we use raw ids that already
    # live in the spare embedding rows; the tokenizer doesn't need to *render*
    # them as text since DriveLMDataset splices the action ids in post-tokenize).
    print(f"[smoke-vla] loading processor...")
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

    print(f"[smoke-vla] loading model (this may take a minute)...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=compute_dtype, device_map="auto", attn_implementation="sdpa",
    )
    model.eval()
    print(f"[smoke-vla] model device : {model.device}")
    print(f"[smoke-vla] GPU memory   : {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    # Trajectory tokenizer
    traj_cfg = TrajectoryTokenizerConfig()
    traj_tok = TrajectoryTokenizer(traj_cfg)
    # Sanity: model's embedding matrix is at least big enough for our token ids
    emb_rows = model.get_input_embeddings().num_embeddings
    if emb_rows <= traj_cfg.traj_end_id:
        print(f"ERROR: model embedding rows={emb_rows} but traj_end_id={traj_cfg.traj_end_id}. "
              f"Need to resize_token_embeddings on this checkpoint.", file=sys.stderr)
        sys.exit(2)
    print(f"[smoke-vla] embed rows   : {emb_rows}  (traj ids fit: bin {traj_cfg.bin_base}.."
          f"{traj_cfg.bin_base+traj_cfg.num_bins-1}, start/end {traj_cfg.traj_start_id}/{traj_cfg.traj_end_id})")

    ds = DriveLMDataset(
        data_path, processor, max_length=max_length,
        video_mode=True, num_frames=num_frames, video_fps=video_fps,
        vla_mode=True, vla_loss_mode=vla_loss_mode,
        traj_start_id=traj_cfg.traj_start_id, traj_end_id=traj_cfg.traj_end_id,
    )
    print(f"[smoke-vla] dataset size : {len(ds)}")
    if len(ds) == 0:
        print("ERROR: dataset empty.", file=sys.stderr)
        sys.exit(2)

    sample = ds[args.sample_idx]
    batch = collate_fn([sample])
    print(f"[smoke-vla] batch keys   : {sorted(k for k in batch.keys() if k != 'image_names')}")
    print(f"[smoke-vla] input_ids    : {tuple(batch['input_ids'].shape)}")
    if "pixel_values_videos" in batch:
        print(f"[smoke-vla] pixel_values_videos : {tuple(batch['pixel_values_videos'].shape)}")
    if "video_grid_thw" in batch:
        print(f"[smoke-vla] video_grid_thw      : {batch['video_grid_thw'].tolist()}")

    # Show how many action tokens are embedded in this batch + the ground-truth waypoints
    ids_list = batch["input_ids"][0].tolist()
    action_ids_in_batch = [i for i in ids_list
                           if i == traj_cfg.traj_start_id or i == traj_cfg.traj_end_id
                           or traj_tok.token_id_to_bin(i) >= 0]
    print(f"[smoke-vla] action tokens in batch: {len(action_ids_in_batch)}")
    gt_wp = traj_tok.decode(action_ids_in_batch)
    print(f"[smoke-vla] GT waypoints (decoded from input_ids): {gt_wp.tolist()}")

    # Move to device
    fwd_batch = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            if k in ("pixel_values_videos", "pixel_values"):
                fwd_batch[k] = v.to(model.device, dtype=compute_dtype)
            else:
                fwd_batch[k] = v.to(model.device)

    # ============ Forward (no grad) ============
    print("[smoke-vla] forward pass (no grad)...")
    with torch.no_grad():
        out = model(**fwd_batch)
    loss = out.loss.item() if out.loss is not None else float("nan")
    print(f"[smoke-vla] forward OK   : logits {tuple(out.logits.shape)}  loss={loss:.4f}")
    if not torch.isfinite(torch.tensor(loss)):
        print("ERROR: loss is not finite.", file=sys.stderr)
        sys.exit(3)

    # ============ Generate ============
    # For generate we keep only the prompt portion: strip everything from the
    # last <|im_start|>assistant\n onwards (so the model has to *predict* the
    # answer + trajectory). This is closer to real inference.
    im_start_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
    asst_marker = processor.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    full_ids = batch["input_ids"][0].tolist()
    cut = None
    for i in range(len(full_ids) - len(asst_marker) + 1):
        if full_ids[i : i + len(asst_marker)] == asst_marker:
            cut = i + len(asst_marker)
    if cut is None:
        print("[smoke-vla] WARNING: could not locate assistant marker; generating from full prompt")
        cut = len(full_ids)
    prompt_ids = torch.tensor([full_ids[:cut]], device=model.device, dtype=batch["input_ids"].dtype)
    prompt_mask = torch.ones_like(prompt_ids)

    gen_inputs = {"input_ids": prompt_ids, "attention_mask": prompt_mask}
    for k in ("pixel_values_videos", "video_grid_thw", "second_per_grid_ts",
              "pixel_values", "image_grid_thw"):
        if k in fwd_batch:
            gen_inputs[k] = fwd_batch[k]

    print(f"[smoke-vla] generating {args.gen_tokens} tokens from {prompt_ids.shape[1]} prompt tokens...")
    with torch.no_grad():
        gen = model.generate(
            **gen_inputs, max_new_tokens=args.gen_tokens, do_sample=False,
        )
    new_tokens = gen[0, prompt_ids.shape[1]:].tolist()
    print(f"[smoke-vla] new token ids: {new_tokens}")
    # Try to decode as text...
    text = processor.tokenizer.decode(new_tokens, skip_special_tokens=False)
    print(f"[smoke-vla] new text     : {text!r}")
    # ...and as a trajectory.
    wp = traj_tok.decode(new_tokens)
    print(f"[smoke-vla] gen waypoints (xy m): {wp.tolist()}")

    print("[smoke-vla] DONE — pipeline works end-to-end. NO optimizer / NO backward / NO ckpt write.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
