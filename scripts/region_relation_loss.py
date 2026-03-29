"""Region-Aware Relation Distillation (RRD) loss for VLM knowledge distillation.

Distills inter-region attention relationships from teacher to student.
- LLaVA-KD baseline (no labels): cosine similarity on O(N²) token self-correlation
- SATS RRD (with labels): cosine similarity on O(C²) region relation matrix

Used by train_distill.py.
"""

import math
import torch
import torch.nn.functional as F


def compute_visual_attention(Q, K, vis_idx, num_heads, num_kv_heads, head_dim):
    """Compute head-averaged attention for visual tokens from Q, K projections.

    Args:
        Q: (B, seq_len, num_heads * head_dim)
        K: (B, seq_len, num_kv_heads * head_dim)
        vis_idx: (N_vis,) indices of visual tokens in the sequence
        num_heads, num_kv_heads, head_dim: attention config

    Returns:
        attn: (B, N_vis, N_vis) head-averaged attention matrix (softmax probs)
    """
    B = Q.shape[0]
    Q_vis = Q[:, vis_idx, :]  # (B, N_vis, H * D)
    K_vis = K[:, vis_idx, :]  # (B, N_vis, H_kv * D)
    N_vis = Q_vis.shape[1]

    Q_vis = Q_vis.view(B, N_vis, num_heads, head_dim).transpose(1, 2)
    K_vis = K_vis.view(B, N_vis, num_kv_heads, head_dim).transpose(1, 2)

    # GQA: expand K to match num_heads
    if num_kv_heads != num_heads:
        K_vis = K_vis.repeat_interleave(num_heads // num_kv_heads, dim=1)

    logits = torch.matmul(Q_vis, K_vis.transpose(-1, -2)) / math.sqrt(head_dim)
    attn = logits.softmax(dim=-1)  # (B, H, N_vis, N_vis)
    return attn.mean(dim=1)        # (B, N_vis, N_vis)


def region_pooled_attention(vis_attn, patch_labels, num_classes):
    """Pool token-level attention to region-level relation matrix.

    Args:
        vis_attn: (N_vis, N_vis) attention matrix
        patch_labels: (N_vis,) int — 0=bg, 1..C=foreground
        num_classes: number of foreground classes C

    Returns:
        R: (C, C) region-level relation matrix
    """
    C = num_classes
    R = torch.zeros(C, C, device=vis_attn.device, dtype=vis_attn.dtype)

    for ci in range(C):
        mi = (patch_labels == ci + 1)
        if mi.sum() == 0:
            continue
        for cj in range(C):
            mj = (patch_labels == cj + 1)
            if mj.sum() == 0:
                continue
            R[ci, cj] = vis_attn[mi][:, mj].mean()

    return R


def _cosine_loss(R_s, R_t):
    """Cosine similarity loss: 1 - cos(R_s, R_t). Range [0, 2].

    Following LLaVA-KD Eq.5: L_rel = 1 - Cos(R_v^s, R_v^t)
    Flatten matrices to vectors for cosine similarity.
    """
    return 1 - F.cosine_similarity(R_s.flatten().unsqueeze(0),
                                    R_t.flatten().unsqueeze(0)).squeeze()


def _kl_loss(R_s, R_t, temperature=1.0):
    """KL divergence loss on row-wise softmax attention distributions.

    Each row of C×C matrix is a distribution over regions.
    KL(teacher || student) per row, averaged over rows.
    """
    t_prob = F.softmax(R_t / temperature, dim=-1)
    s_log_prob = F.log_softmax(R_s / temperature, dim=-1)
    # sum over columns, mean over rows; clamp to avoid float precision negatives
    kl_per_row = F.kl_div(s_log_prob, t_prob, reduction='none').sum(dim=-1)
    return kl_per_row.clamp(min=0).mean() * (temperature ** 2)


def region_relation_distill_loss(teacher_store, student_store,
                                  layer_map, input_ids, image_token_id,
                                  teacher_cfg, student_cfg,
                                  patch_labels_batch=None,
                                  rrd_loss_type="cosine"):
    """Compute RRD loss between teacher and student.

    Both paths use cosine similarity (following LLaVA-KD):
    - With patch_labels: O(C²) region-pooled relation matrix (SATS RRD)
    - Without patch_labels: O(N²) full token attention matrix (LLaVA-KD RDist)

    Args:
        teacher_store: dict with keys 'q_{layer}', 'k_{layer}' (detached)
        student_store: dict with keys 'q_{layer}', 'k_{layer}' (with grad)
        layer_map: dict {teacher_layer_idx: student_layer_idx}
        input_ids: (B, seq_len)
        image_token_id: int
        teacher_cfg: dict with num_attention_heads, num_key_value_heads, hidden_size
        student_cfg: dict with same keys
        patch_labels_batch: optional list of (N_vis,) int tensors per sample

    Returns:
        loss: scalar
    """
    t_heads = teacher_cfg["num_attention_heads"]
    t_kv = teacher_cfg["num_key_value_heads"]
    t_dim = teacher_cfg["hidden_size"] // t_heads

    s_heads = student_cfg["num_attention_heads"]
    s_kv = student_cfg["num_key_value_heads"]
    s_dim = student_cfg["hidden_size"] // s_heads

    B = input_ids.shape[0]
    loss = 0.0
    count = 0

    for t_layer, s_layer in layer_map.items():
        t_Q = teacher_store[f"q_{t_layer}"]
        t_K = teacher_store[f"k_{t_layer}"]
        s_Q = student_store[f"q_{s_layer}"]
        s_K = student_store[f"k_{s_layer}"]

        for b in range(B):
            vis_idx = (input_ids[b] == image_token_id).nonzero(as_tuple=True)[0]
            if len(vis_idx) == 0:
                continue

            # Compute visual attention per sample
            t_attn = compute_visual_attention(
                t_Q[b:b+1], t_K[b:b+1], vis_idx, t_heads, t_kv, t_dim
            ).squeeze(0)  # (N_vis, N_vis)

            s_attn = compute_visual_attention(
                s_Q[b:b+1], s_K[b:b+1], vis_idx, s_heads, s_kv, s_dim
            ).squeeze(0)

            if patch_labels_batch is not None and patch_labels_batch[b] is not None:
                labels = patch_labels_batch[b]
                labels = labels[:len(vis_idx)]
                n_cls = int(labels.max().item())
                if n_cls > 0:
                    # SATS: O(C²) region relation matrix
                    R_t = region_pooled_attention(t_attn.detach(), labels, n_cls)
                    R_s = region_pooled_attention(s_attn, labels, n_cls)
                    if rrd_loss_type == "kl":
                        loss += _kl_loss(R_s, R_t)
                    else:
                        loss += _cosine_loss(R_s, R_t)
                else:
                    # No foreground: fallback to O(N²) full cosine
                    loss += _cosine_loss(s_attn, t_attn.detach())
            else:
                # LLaVA-KD RDist: O(N²) full token cosine similarity
                loss += _cosine_loss(s_attn, t_attn.detach())

            count += 1

    return loss / max(count, 1)