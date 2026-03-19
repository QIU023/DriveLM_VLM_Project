"""Visual token compression methods for Qwen2.5-VL.

Compresses visual tokens between vision encoder output and LLM input:
- avg_pool: 2D spatial average pooling (preserves grid structure)
- fastervlm: importance-based token selection by L2 norm
- prumerge: prune low-importance + merge into nearest kept tokens
- pyramiddrop: two-stage progressive dropping by importance

References:
- FasterVLM (Chen et al., 2024)
- LLaVA-PruMerge (Shang et al., 2024)
- PyramidDrop (Xia et al., 2024)
"""

import math
import torch
import torch.nn.functional as F


def compress_visual_tokens(image_embeds, grid_thw, method, ratio):
    """Compress visual tokens.

    Args:
        image_embeds: (total_tokens, hidden_dim) concatenated visual tokens
        grid_thw: (num_images, 3) — (temporal, height, width) per image
        method: "avg_pool" | "fastervlm" | "prumerge" | "pyramiddrop"
        ratio: compression ratio (4 = keep 1/4 of tokens)

    Returns:
        compressed: (total_compressed, hidden_dim)
        new_grid_thw: (num_images, 3)
    """
    if method == "none" or ratio <= 1:
        return image_embeds, grid_thw

    fn = {
        "avg_pool": _avg_pool,
        "fastervlm": _fastervlm,
        "prumerge": _prumerge,
        "pyramiddrop": _pyramiddrop,
    }
    if method not in fn:
        raise ValueError(f"Unknown compress method: {method}")
    return fn[method](image_embeds, grid_thw, ratio)


def _avg_pool(image_embeds, grid_thw, ratio):
    """2x2 (or NxN) spatial average pooling on the visual token grid."""
    pool_size = int(math.sqrt(ratio))
    assert pool_size * pool_size == ratio, f"ratio={ratio} must be perfect square for avg_pool"

    results, new_thws = [], []
    offset = 0
    for i in range(grid_thw.shape[0]):
        t, h, w = int(grid_thw[i, 0]), int(grid_thw[i, 1]), int(grid_thw[i, 2])
        n = t * h * w
        tokens = image_embeds[offset : offset + n]
        dim = tokens.shape[-1]

        new_h, new_w = h // pool_size, w // pool_size
        # crop to exact multiple, reshape, pool
        tokens = tokens.view(t, h, w, dim)[:, : new_h * pool_size, : new_w * pool_size, :]
        tokens = tokens.reshape(t, new_h, pool_size, new_w, pool_size, dim).mean(dim=(2, 4))
        results.append(tokens.reshape(-1, dim))
        new_thws.append([t, new_h, new_w])
        offset += n

    return torch.cat(results), torch.tensor(new_thws, dtype=grid_thw.dtype, device=grid_thw.device)


def _fastervlm(image_embeds, grid_thw, ratio):
    """Keep top-K tokens by L2 norm (importance proxy for CLS-attention)."""
    results, new_thws = [], []
    offset = 0
    for i in range(grid_thw.shape[0]):
        t, h, w = int(grid_thw[i, 0]), int(grid_thw[i, 1]), int(grid_thw[i, 2])
        n = t * h * w
        tokens = image_embeds[offset : offset + n]
        k = max(1, n // ratio)

        _, idx = tokens.norm(dim=-1).topk(k)
        idx = idx.sort().values  # preserve spatial order
        results.append(tokens[idx])
        new_thws.append([t, 1, k])
        offset += n

    return torch.cat(results), torch.tensor(new_thws, dtype=grid_thw.dtype, device=grid_thw.device)


def _prumerge(image_embeds, grid_thw, ratio):
    """Prune low-importance tokens, merge them into nearest kept token."""
    results, new_thws = [], []
    offset = 0
    for i in range(grid_thw.shape[0]):
        t, h, w = int(grid_thw[i, 0]), int(grid_thw[i, 1]), int(grid_thw[i, 2])
        n = t * h * w
        tokens = image_embeds[offset : offset + n]
        k = max(1, n // ratio)

        _, topk_idx = tokens.norm(dim=-1).topk(k)
        keep_mask = torch.zeros(n, dtype=torch.bool, device=tokens.device)
        keep_mask[topk_idx] = True

        kept = tokens[keep_mask].clone()
        pruned = tokens[~keep_mask]

        if pruned.shape[0] > 0 and kept.shape[0] > 0:
            # cosine similarity → assign each pruned token to nearest kept
            sim = torch.mm(F.normalize(pruned, dim=-1), F.normalize(kept, dim=-1).t())
            assignments = sim.argmax(dim=-1)
            for j in range(k):
                mask_j = assignments == j
                if mask_j.any():
                    kept[j] = torch.cat([kept[j : j + 1], pruned[mask_j]]).mean(dim=0)

        results.append(kept)
        new_thws.append([t, 1, k])
        offset += n

    return torch.cat(results), torch.tensor(new_thws, dtype=grid_thw.dtype, device=grid_thw.device)


def _pyramiddrop(image_embeds, grid_thw, ratio):
    """Two-stage progressive dropping by importance (simplified pre-LLM version).

    Full PyramidDrop operates inside LLM layers for layer-wise dropping.
    This version applies two rounds of norm-based selection at the
    vision-LLM boundary to approximate the progressive behaviour.
    """
    results, new_thws = [], []
    offset = 0
    for i in range(grid_thw.shape[0]):
        t, h, w = int(grid_thw[i, 0]), int(grid_thw[i, 1]), int(grid_thw[i, 2])
        n = t * h * w
        tokens = image_embeds[offset : offset + n]
        k = max(1, n // ratio)

        # stage 1: coarse — keep top 50 %
        mid_k = max(k, n // 2)
        _, mid_idx = tokens.norm(dim=-1).topk(mid_k)
        mid_tokens = tokens[mid_idx]

        # stage 2: fine — keep target k
        if mid_k > k:
            _, fine_idx = mid_tokens.norm(dim=-1).topk(k)
            fine_idx = fine_idx.sort().values
            results.append(mid_tokens[fine_idx])
        else:
            results.append(mid_tokens)

        new_thws.append([t, 1, k])
        offset += n

    return torch.cat(results), torch.tensor(new_thws, dtype=grid_thw.dtype, device=grid_thw.device)
