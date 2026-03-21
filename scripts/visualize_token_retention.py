"""Visualize which spatial positions are retained by each compression method.

Maps kept token indices back to 2D grid positions and overlays a heatmap
on the original driving image. Shows what the model "sees" after compression.

Usage:
    python scripts/visualize_token_retention.py \
        --config configs/gh200.yaml \
        --methods fastervlm,prumerge,pyramiddrop \
        --ratio 4 --n 5
"""

import argparse
import json
import os
import random
import sys
import yaml

import torch
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

VAL_FILE = os.path.join(BASE_DIR, "data_processed", "val.json")


def get_base_model(model):
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        return model.base_model.model
    return model


def get_kept_indices(tokens, method, ratio):
    """Replicate compression selection logic but return kept LINEAR indices.

    Args:
        tokens: (n, dim) visual tokens
        method: compression method name
        ratio: compression ratio

    Returns:
        kept_idx: 1D tensor of kept linear indices (sorted)
        k: number of kept tokens
    """
    n = tokens.shape[0]
    k = max(1, n // ratio)

    if method == "fastervlm":
        _, idx = tokens.norm(dim=-1).topk(k)
        return idx.sort().values, k

    elif method == "prumerge":
        _, idx = tokens.norm(dim=-1).topk(k)
        return idx.sort().values, k

    elif method == "pyramiddrop":
        mid_k = max(k, n // 2)
        _, mid_idx = tokens.norm(dim=-1).topk(mid_k)
        mid_tokens = tokens[mid_idx]
        if mid_k > k:
            _, fine_idx = mid_tokens.norm(dim=-1).topk(k)
            # Map fine_idx back to original indices
            kept = mid_idx[fine_idx]
        else:
            kept = mid_idx
        return kept.sort().values, k

    else:
        raise ValueError(f"Unknown method: {method}")


def create_retention_heatmap(kept_idx, h, w):
    """Create a 2D retention mask from linear indices."""
    mask = np.zeros((h, w), dtype=np.float32)
    for idx in kept_idx.cpu().numpy():
        row = idx // w
        col = idx % w
        if row < h and col < w:
            mask[row, col] = 1.0
    return mask


def select_image_samples(data, n=5, seed=42):
    """Pick diverse samples with images."""
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


def main():
    # Must set backend before importing pyplot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from PIL import Image

    parser = argparse.ArgumentParser(description="Visualize token retention heatmaps")
    parser.add_argument("--config", required=True)
    parser.add_argument("--methods", default="fastervlm,prumerge,pyramiddrop")
    parser.add_argument("--ratio", type=int, default=4)
    parser.add_argument("--n", type=int, default=5, help="Number of images")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",")]
    output_dir = args.output_dir or os.path.join(BASE_DIR, "visualizations")
    os.makedirs(output_dir, exist_ok=True)

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

    # Load data
    print(f"Loading val data...")
    with open(VAL_FILE, "r", encoding="utf-8") as f:
        val_data = json.load(f)
    samples = select_image_samples(val_data, n=args.n, seed=args.seed)
    print(f"  Selected {len(samples)} images\n")

    # Load model (base only, no LoRA needed for visualization)
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"Loading {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    if hasattr(processor, "image_processor") and processor.image_processor is not None:
        processor.image_processor.min_pixels = min_pixels
        processor.image_processor.max_pixels = max_pixels

    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=compute_dtype, device_map="auto")
    model.eval()
    print(f"  Model loaded.\n")

    # Custom colormap: red (dropped) → green (kept)
    cmap = LinearSegmentedColormap.from_list("retention", [
        (0.0, "#1a1a2e"),    # dark blue-black (dropped)
        (1.0, "#00ff88"),    # bright green (kept)
    ])

    for img_idx, (sample, img_path) in enumerate(samples):
        print(f"[{img_idx+1}/{len(samples)}] {os.path.basename(img_path)}")
        image = Image.open(img_path).convert("RGB")

        # Process image to get visual tokens
        msgs = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe this scene."},
        ]}]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

        image_grid_thw = inputs["image_grid_thw"]
        merge_size = getattr(model.model.visual, "spatial_merge_size", 2)
        post_grid = image_grid_thw.clone()
        post_grid[:, 1] = image_grid_thw[:, 1] // merge_size
        post_grid[:, 2] = image_grid_thw[:, 2] // merge_size
        t = int(post_grid[0, 0])
        h = int(post_grid[0, 1])
        w = int(post_grid[0, 2])
        n_tokens = t * h * w

        # Run vision encoder
        vis_dtype = next(model.model.visual.parameters()).dtype
        with torch.no_grad():
            vis_out = model.model.visual(inputs["pixel_values"].to(vis_dtype), grid_thw=image_grid_thw)
            image_embeds = vis_out.pooler_output if hasattr(vis_out, "pooler_output") and vis_out.pooler_output is not None else vis_out.last_hidden_state
            if isinstance(image_embeds, (tuple, list)):
                image_embeds = image_embeds[0]

        # For single frame, take first t*h*w tokens
        tokens = image_embeds[:n_tokens]

        # Create figure: original + one per method
        n_cols = 1 + len(methods)
        fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
        if n_cols == 1:
            axes = [axes]

        # Original image
        axes[0].imshow(image)
        axes[0].set_title(f"Original\n{image.size[0]}x{image.size[1]}", fontsize=11)
        axes[0].axis("off")

        # Token importance heatmap (L2 norm) for reference
        norms = tokens.norm(dim=-1).float().cpu().numpy().reshape(h, w)
        norm_min, norm_max = norms.min(), norms.max()

        for m_idx, method in enumerate(methods):
            ax = axes[1 + m_idx]
            kept_idx, k = get_kept_indices(tokens, method, args.ratio)
            mask = create_retention_heatmap(kept_idx, h, w)

            # Upsample mask to image size
            img_w, img_h = image.size
            mask_upsampled = np.kron(mask, np.ones((img_h // h, img_w // w)))
            # Handle remainder
            if mask_upsampled.shape[0] < img_h or mask_upsampled.shape[1] < img_w:
                pad_h = img_h - mask_upsampled.shape[0]
                pad_w = img_w - mask_upsampled.shape[1]
                mask_upsampled = np.pad(mask_upsampled, ((0, pad_h), (0, pad_w)), mode='edge')
            mask_upsampled = mask_upsampled[:img_h, :img_w]

            # Show image with overlay
            ax.imshow(image)
            ax.imshow(mask_upsampled, alpha=0.55, cmap=cmap, vmin=0, vmax=1)
            ax.set_title(f"{method} ({args.ratio}x)\n{k}/{n_tokens} tokens kept", fontsize=11)
            ax.axis("off")

        plt.tight_layout()
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(output_dir, f"retention_{img_name}_r{args.ratio}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {out_path} (grid {h}x{w} = {n_tokens} tokens)")

    # Also create a combined importance norm visualization for one image
    _, img_path = samples[0]
    image = Image.open(img_path).convert("RGB")
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe."}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    image_grid_thw = inputs["image_grid_thw"]
    post_grid = image_grid_thw.clone()
    post_grid[:, 1] = image_grid_thw[:, 1] // merge_size
    post_grid[:, 2] = image_grid_thw[:, 2] // merge_size
    h, w = int(post_grid[0, 1]), int(post_grid[0, 2])
    with torch.no_grad():
        vis_out = model.model.visual(inputs["pixel_values"].to(vis_dtype), grid_thw=image_grid_thw)
        emb = vis_out.pooler_output if hasattr(vis_out, "pooler_output") and vis_out.pooler_output is not None else vis_out.last_hidden_state
        if isinstance(emb, (tuple, list)):
            emb = emb[0]

    norms = emb[:h*w].norm(dim=-1).float().cpu().numpy().reshape(h, w)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    img_w, img_h = image.size
    ax.imshow(image)
    norm_up = np.kron(norms, np.ones((img_h // h, img_w // w)))
    if norm_up.shape[0] < img_h or norm_up.shape[1] < img_w:
        norm_up = np.pad(norm_up, ((0, img_h - norm_up.shape[0]), (0, img_w - norm_up.shape[1])), mode='edge')
    norm_up = norm_up[:img_h, :img_w]
    im = ax.imshow(norm_up, alpha=0.6, cmap="hot")
    ax.set_title(f"Token Importance (L2 norm) — {h}x{w} grid", fontsize=12)
    ax.axis("off")
    plt.colorbar(im, ax=ax, shrink=0.8, label="L2 norm")
    plt.tight_layout()
    out_path = os.path.join(output_dir, "token_importance_heatmap.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nImportance heatmap: {out_path}")

    print(f"\nAll visualizations saved to {output_dir}/")


if __name__ == "__main__":
    main()
