"""Region-Aware Relation Distillation (RRD) loss for VLM knowledge distillation.

Distills inter-region attention relationships from teacher to student using
class-region pooling: compresses O(N^2) token relations to O(C^2) region relations.

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
        attn: (B, N_vis, N_vis) head-averaged attention matrix
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
    # Return both logits and probs for flexible loss computation
    attn = logits.softmax(dim=-1)  # (B, H, N_vis, N_vis)
    return attn.mean(dim=1), logits.mean(dim=1)  # (B, N_vis, N_vis) each


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


def _row_kl(s_mat, t_mat):
    """Row-wise KL divergence: treat each row as a distribution.

    Works for both (N_vis, N_vis) attention logits and (C, C) region matrices.
    Adds small eps before log to avoid log(0) on sparse region matrices.
    """
    eps = 1e-8
    t_prob = F.softmax(t_mat, dim=-1)
    s_log_prob = F.log_softmax(s_mat + eps, dim=-1)
    return F.kl_div(s_log_prob, t_prob, reduction="batchmean")


def region_relation_distill_loss(teacher_store, student_store,
                                  layer_map, input_ids, image_token_id,
                                  teacher_cfg, student_cfg,
                                  patch_labels_batch=None):
    """Compute RRD loss between teacher and student.

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
            t_attn, t_logits = compute_visual_attention(
                t_Q[b:b+1], t_K[b:b+1], vis_idx, t_heads, t_kv, t_dim
            )
            t_attn, t_logits = t_attn.squeeze(0), t_logits.squeeze(0)

            s_attn, s_logits = compute_visual_attention(
                s_Q[b:b+1], s_K[b:b+1], vis_idx, s_heads, s_kv, s_dim
            )
            s_attn, s_logits = s_attn.squeeze(0), s_logits.squeeze(0)

            if patch_labels_batch is not None and patch_labels_batch[b] is not None:
                labels = patch_labels_batch[b]
                labels = labels[:len(vis_idx)]
                n_cls = int(labels.max().item())
                if n_cls > 0:
                    # SATS core: O(C²) region relation KL divergence
                    R_t = region_pooled_attention(t_attn.detach(), labels, n_cls)
                    R_s = region_pooled_attention(s_attn, labels, n_cls)
                    # R is already positive (avg of softmax probs),
                    # normalize rows to distributions then KL
                    eps = 1e-8
                    R_t_norm = R_t / (R_t.sum(dim=-1, keepdim=True) + eps)
                    R_s_norm = R_s / (R_s.sum(dim=-1, keepdim=True) + eps)
                    loss += F.kl_div(
                        (R_s_norm + eps).log(), R_t_norm,
                        reduction="batchmean"
                    )
                else:
                    # No foreground regions: fallback to full O(N²) token KL
                    loss += _row_kl(s_logits, t_logits.detach())
            else:
                # No labels available: fallback to full O(N²) token KL
                loss += _row_kl(s_logits, t_logits.detach())

            count += 1

    return loss / max(count, 1)
