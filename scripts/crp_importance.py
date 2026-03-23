"""Class-Region Pooled (CRP) attention importance scoring.

Computes per-token importance using ViT self-attention weighted by
class-region information from object annotations (SATS-style).

Used by:
  - precompute_crp_data.py (offline precomputation)
  - visual_compress.py (crp / crp_merge methods)
"""

import torch


def crp_importance(attn_maps, patch_labels):
    """Compute CRP importance from attention maps and patch labels.

    Args:
        attn_maps: dict {layer_idx: (H, N, N)} attention weights per fullatt layer
        patch_labels: (N,) int tensor — 0=background, 1..C=foreground classes

    Returns:
        importance: (N,) float tensor, normalized to [0, 1]
    """
    N = patch_labels.shape[0]
    importance = torch.zeros(N)
    fg_mask = patch_labels > 0

    if not fg_mask.any():
        # No foreground objects: fall back to mean received attention
        for attn in attn_maps.values():
            importance += attn.float().mean(dim=0).sum(dim=0)
        return importance / (importance.max() + 1e-8)

    classes = patch_labels[fg_mask].unique()

    for attn in attn_maps.values():
        attn_avg = attn.float().mean(dim=0)  # (N, N) average over heads
        for c in classes:
            c_mask = patch_labels == c
            # Class-region pooled attention (SATS formula 1):
            # average attention FROM class-region tokens TO all tokens
            pooled = attn_avg[c_mask].mean(dim=0)  # (N,)
            importance += pooled

    return importance / (importance.max() + 1e-8)


def pool_importance_to_post_merger(importance, t, h, w, merge_size=2):
    """Average-pool importance from pre-merger to post-merger resolution.

    Args:
        importance: (t*h*w,) float tensor at pre-merger resolution
        t, h, w: pre-merger grid dimensions
        merge_size: spatial merge factor (default 2 for Qwen2.5-VL)

    Returns:
        pooled: (t * h//merge_size * w//merge_size,) float tensor
    """
    imp = importance.view(t, h, w)
    # Crop to exact multiple
    new_h = (h // merge_size) * merge_size
    new_w = (w // merge_size) * merge_size
    imp = imp[:, :new_h, :new_w]
    # Pool
    imp = imp.reshape(t, new_h // merge_size, merge_size, new_w // merge_size, merge_size)
    imp = imp.mean(dim=(2, 4))  # (t, post_h, post_w)
    return imp.flatten()
