"""Precompute CRP importance scores for all DriveLM images.

Pipeline (single pass per image):
  1. Parse DriveLM data → unique images + object references
  2. Load Qwen2.5-VL ViT (monkey-patch fullatt layers for attention extraction)
  3. For each image: ViT forward → attention maps → patch labels → CRP importance
  4. Pool importance to post-merger resolution, save to .pt file

Output: precomputed/crp_importance.pt  ({image_name: (N_post,) importance})

Usage:
  python scripts/precompute_crp_data.py --config configs/gb200.yaml
  python scripts/precompute_crp_data.py --config configs/gb200.yaml --mini
"""

import argparse
import importlib
import json
import math
import os
import re
import sys
import torch
from collections import defaultdict
from PIL import Image
from tqdm import tqdm

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crp_importance import crp_importance, pool_importance_to_post_merger

# Object reference pattern in DriveLM: <c3,CAM_FRONT,1490.1,588.3>
OBJ_PATTERN = re.compile(r"<(c\d+),(\w+),([\d.]+),([\d.]+)>")

FULLATT_LAYERS = [7, 15, 23, 31]


# ===================== Data parsing =====================

def parse_drivelm_objects(data):
    """Parse DriveLM data → {image_name: [(class_str, x, y), ...]}."""
    image_objects = defaultdict(list)
    for item in data:
        image_name = None
        for msg in item["messages"]:
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "image":
                        path = part["image"]
                        if path.startswith("file://"):
                            path = path[7:]
                        image_name = os.path.basename(path)
            text = ""
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
            for m in OBJ_PATTERN.finditer(text):
                cls, cam, x, y = m.groups()
                if image_name:
                    image_objects[image_name].append((cls, float(x), float(y)))
    return dict(image_objects)


def get_unique_images(data):
    """Get {image_name: full_path} for unique images in the dataset."""
    paths = {}
    for item in data:
        for msg in item["messages"]:
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "image":
                        path = part["image"]
                        if path.startswith("file://"):
                            path = path[7:]
                        paths[os.path.basename(path)] = path
    return paths


def compute_patch_labels(obj_refs, h, w, orig_h, orig_w, proc_h, proc_w, patch_size=14):
    """Convert object point references to patch-level labels at pre-merger resolution.

    Returns:
        labels: (h * w,) int tensor — 0=bg, 1..C=foreground
    """
    labels = torch.zeros(h, w, dtype=torch.long)
    class_map = {}

    for cls, x, y in obj_refs:
        x_proc = x * proc_w / orig_w
        y_proc = y * proc_h / orig_h
        px = max(0, min(w - 1, int(x_proc / patch_size)))
        py = max(0, min(h - 1, int(y_proc / patch_size)))

        if cls not in class_map:
            class_map[cls] = len(class_map) + 1

        # Mark a small region around the object point (radius=1 patch)
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                ny, nx = py + dy, px + dx
                if 0 <= ny < h and 0 <= nx < w:
                    labels[ny, nx] = class_map[cls]

    return labels.flatten()


# ===================== ViT attention extraction =====================

def _get_apply_rotary():
    """Import apply_rotary_pos_emb_vision from transformers (version-robust)."""
    for path in [
        "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl",
        "transformers.models.qwen2_vl.modeling_qwen2_vl",
    ]:
        try:
            mod = importlib.import_module(path)
            fn = getattr(mod, "apply_rotary_pos_emb_vision", None)
            if fn is not None:
                return fn
        except ImportError:
            continue
    return None


class ViTAttentionExtractor:
    """Monkey-patches Qwen2.5-VL ViT fullatt layers to capture attention weights.

    Compatible with transformers 5.x API:
      forward(hidden_states, cu_seqlens, rotary_pos_emb=None,
              position_embeddings=None, **kwargs)
    where position_embeddings = (cos, sin) replaces the old rotary_pos_emb tensor.
    apply_rotary_pos_emb_vision(q, k, cos, sin) — four separate args.
    """

    def __init__(self, visual_model):
        self.visual = visual_model
        self.store = {}
        self._originals = {}
        self._apply_rope = _get_apply_rotary()
        self._patch()

    def _patch(self):
        for idx in FULLATT_LAYERS:
            attn_mod = self.visual.blocks[idx].attn
            self._originals[idx] = attn_mod.forward
            attn_mod.forward = self._make_forward(attn_mod, idx)

    def _make_forward(self, attn_mod, layer_idx):
        store = self.store
        apply_rope = self._apply_rope

        def forward(hidden_states, cu_seqlens,
                    rotary_pos_emb=None, position_embeddings=None, **kwargs):
            seq_length = hidden_states.shape[0]
            # (seq_len, 3, num_heads, head_dim) → unbind → (seq_len, num_heads, head_dim)
            q, k, v = (
                attn_mod.qkv(hidden_states)
                .reshape(seq_length, 3, attn_mod.num_heads, -1)
                .permute(1, 0, 2, 3)
                .unbind(0)
            )

            if apply_rope is not None:
                if position_embeddings is not None:
                    # transformers 5.x: position_embeddings = (cos, sin)
                    cos, sin = position_embeddings
                    q, k = apply_rope(q, k, cos, sin)
                elif rotary_pos_emb is not None:
                    # transformers 4.x: rotary_pos_emb was a tensor; try old call
                    try:
                        q = apply_rope(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
                        k = apply_rope(k.unsqueeze(0), rotary_pos_emb).squeeze(0)
                    except TypeError:
                        pass  # skip RoPE if API mismatch

            head_dim = q.shape[-1]
            q_h = q.transpose(0, 1)  # (H, N, D)
            k_h = k.transpose(0, 1)

            attn_weights = torch.bmm(q_h, k_h.transpose(1, 2)) / math.sqrt(head_dim)
            attn_weights = attn_weights.softmax(dim=-1)
            store[layer_idx] = attn_weights.detach().cpu().float()

            # Normal output
            v_h = v.transpose(0, 1)
            out = torch.bmm(attn_weights.to(v_h.dtype), v_h)
            out = out.transpose(0, 1).reshape(seq_length, -1)
            out = attn_mod.proj(out)
            return out

        return forward

    def get_and_clear(self):
        result = dict(self.store)
        self.store.clear()
        return result

    def unpatch(self):
        for idx, orig in self._originals.items():
            self.visual.blocks[idx].attn.forward = orig


# ===================== Main =====================

def main():
    parser = argparse.ArgumentParser(description="Precompute CRP importance scores")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mini", action="store_true", help="Use train_mini.json")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    # Load config
    from train_lora import load_config
    cfg = load_config(args.config)

    model_id = cfg["model_id"]
    min_pixels = cfg.get("min_pixels", 256 * 28 * 28)
    max_pixels = cfg.get("max_pixels", 512 * 28 * 28)

    # Load data
    data_file = os.path.join(
        _BASE_DIR, "data_processed", "train_mini.json" if args.mini else "train.json"
    )
    print(f"Loading data: {data_file}")
    with open(data_file) as f:
        data = json.load(f)

    image_objects = parse_drivelm_objects(data)
    image_paths = get_unique_images(data)
    print(f"Unique images: {len(image_paths)} | Images with object refs: {len(image_objects)}")

    # Load model
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"Loading model: {model_id}")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    processor = AutoProcessor.from_pretrained(model_id)
    if hasattr(processor, "image_processor") and processor.image_processor is not None:
        processor.image_processor.min_pixels = min_pixels
        processor.image_processor.max_pixels = max_pixels
    model.eval()

    visual = model.model.visual
    merge_size = getattr(visual, "spatial_merge_size", 2)
    extractor = ViTAttentionExtractor(visual)

    # Process each unique image
    output_dir = os.path.join(_BASE_DIR, "precomputed")
    os.makedirs(output_dir, exist_ok=True)

    importance_dict = {}
    patch_labels_dict = {}
    dummy_text = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>hi<|im_end|>"

    for name, path in tqdm(image_paths.items(), desc="Computing CRP importance"):
        if not os.path.exists(path):
            continue
        try:
            image = Image.open(path).convert("RGB")
            orig_w, orig_h = image.size

            inputs = processor(text=[dummy_text], images=[image], return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(model.device, dtype=torch.bfloat16)
            grid_thw = inputs["image_grid_thw"].to(model.device)

            t = int(grid_thw[0, 0])
            h = int(grid_thw[0, 1])
            w = int(grid_thw[0, 2])

            # ViT forward (attention captured by hooks)
            with torch.no_grad():
                visual(pixel_values, grid_thw=grid_thw)

            attn_maps = extractor.get_and_clear()

            # Patch labels at pre-merger resolution
            proc_h = h * 14  # pixels = grid * patch_size
            proc_w = w * 14
            obj_refs = image_objects.get(name, [])
            labels = compute_patch_labels(obj_refs, h, w, orig_h, orig_w, proc_h, proc_w)

            # CRP importance at pre-merger resolution
            imp = crp_importance(attn_maps, labels)

            # Pool to post-merger resolution
            imp_post = pool_importance_to_post_merger(imp, t, h, w, merge_size)
            importance_dict[name] = imp_post

            # Pool patch labels to post-merger resolution (max = preserve foreground)
            lbl = labels.view(t, h, w)
            new_h = (h // merge_size) * merge_size
            new_w = (w // merge_size) * merge_size
            lbl = lbl[:, :new_h, :new_w].reshape(
                t, new_h // merge_size, merge_size, new_w // merge_size, merge_size
            ).amax(dim=(2, 4)).flatten()
            patch_labels_dict[name] = lbl

        except Exception as e:
            tqdm.write(f"[WARN] {name}: {e}")
            continue

    # Save importance scores (for visual token compression)
    out_path = args.output or os.path.join(output_dir, "crp_importance.pt")
    torch.save(importance_dict, out_path)
    print(f"\nSaved CRP importance for {len(importance_dict)} images → {out_path}")

    # Save patch labels (for region-aware distillation)
    labels_path = os.path.join(output_dir, "crp_patch_labels.pt")
    torch.save(patch_labels_dict, labels_path)
    print(f"Saved CRP patch labels for {len(patch_labels_dict)} images → {labels_path}")

    extractor.unpatch()


if __name__ == "__main__":
    main()
