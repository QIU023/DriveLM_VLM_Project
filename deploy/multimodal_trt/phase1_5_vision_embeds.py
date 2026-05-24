"""Phase 1.5 — Assemble pre-computed multimodal_embedding tensor for TRT.

Produces a single (mm_total, hidden * (1 + n_deepstack)) tensor by:
  1. Running HF Qwen3-VL vision tower via get_image_features / get_video_features
  2. Stacking [pooler_output] + deepstack_features along dim=1 per modality
  3. Concatenating in INPUT-IDS ORDER (B.5'' = video first, image second)

This matches TRT-LLM 1.3.0rc15's expected format at
modeling_qwen3vl.py:932-945 (post `split_mm_embeds` reverses this stacking).

The output tensor is what we inject as `multimodal_data["multimodal_embedding"]`
to hit the cache-hit path in `get_multimodal_embeddings` and skip the vision
encoder forward (thus bypassing the single-modality assertion at line 920).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch


def assemble_mm_embedding(
    *,
    hf_model,
    pixel_values: Optional[torch.Tensor],          # image patches (n_img_patches, patch_dim)
    image_grid_thw: Optional[torch.Tensor],        # (n_img, 3)
    pixel_values_videos: Optional[torch.Tensor],   # video patches (n_vid_patches, patch_dim)
    video_grid_thw: Optional[torch.Tensor],        # (n_vid, 3)
    modality_order: Tuple[str, ...] = ("video", "image"),
) -> Dict[str, torch.Tensor]:
    """Returns dict with:
      - mm_embedding: (mm_total_tokens, lm_hidden * (1 + n_deepstack))  bf16
      - image_mm_tokens: int (post-merger token count for image branch)
      - video_mm_tokens: int (post-merger token count for video branch)
    """
    device = next(hf_model.parameters()).device
    dtype = next(hf_model.parameters()).dtype  # bf16

    image_concat: Optional[torch.Tensor] = None
    video_concat: Optional[torch.Tensor] = None
    image_n_tokens = 0
    video_n_tokens = 0

    with torch.no_grad():
        if pixel_values is not None and image_grid_thw is not None:
            if image_grid_thw.dim() == 1:
                image_grid_thw = image_grid_thw.unsqueeze(0)
            out = hf_model.get_image_features(
                pixel_values=pixel_values.to(device, dtype=dtype),
                image_grid_thw=image_grid_thw.to(device),
            )
            # pooler_output is a tuple-of-tensors (one per image), each (n_tokens, hidden)
            po = out.pooler_output
            if isinstance(po, (list, tuple)):
                base = torch.cat(list(po), dim=0)
            else:
                base = po
            ds = out.deepstack_features  # list of n_deepstack tensors, each (n_tokens, hidden)
            image_concat = torch.cat([base] + list(ds), dim=1)  # (n_img_tokens, hidden * (1+n_ds))
            image_n_tokens = base.shape[0]

        if pixel_values_videos is not None and video_grid_thw is not None:
            if video_grid_thw.dim() == 1:
                video_grid_thw = video_grid_thw.unsqueeze(0)
            out = hf_model.get_video_features(
                pixel_values_videos=pixel_values_videos.to(device, dtype=dtype),
                video_grid_thw=video_grid_thw.to(device),
            )
            po = out.pooler_output
            if isinstance(po, (list, tuple)):
                base = torch.cat(list(po), dim=0)
            else:
                base = po
            ds = out.deepstack_features
            video_concat = torch.cat([base] + list(ds), dim=1)
            video_n_tokens = base.shape[0]

    parts = []
    for m in modality_order:
        if m == "image" and image_concat is not None:
            parts.append(image_concat)
        elif m == "video" and video_concat is not None:
            parts.append(video_concat)
    if not parts:
        raise ValueError("No multimodal inputs provided")
    mm_embedding = torch.cat(parts, dim=0)
    return {
        "mm_embedding": mm_embedding,
        "image_mm_tokens": image_n_tokens,
        "video_mm_tokens": video_n_tokens,
    }


def smoke():
    import os, sys
    from pathlib import Path
    BASE = Path("/workspace/DriveLM_VLM_Project")
    sys.path.insert(0, str(BASE / "scripts"))
    os.chdir(BASE)
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from multimodal_planning_dataset import MultiModalPlanningDataset

    ckpt = str(BASE / "checkpoints_qwen25/nusc_planning_b5pp_1cam_qwen3vl_multimodal/final")
    print(f"[1.5] loading {ckpt}")
    model = AutoModelForImageTextToText.from_pretrained(
        ckpt, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda:0").eval()
    proc = AutoProcessor.from_pretrained(ckpt)
    ds = MultiModalPlanningDataset(
        infos_path=str(BASE / "data/uniad_infos/nuscenes_infos_temporal_val.pkl"),
        nusc_root=str(BASE / "data/nuscenes"),
        processor=proc, max_length=12288,
        num_past_frames=4, num_future_waypoints=6, video_fps=2.0,
        vla_loss_mode="answer_and_traj", max_samples=2, require_full_future=True,
        planning_cams=["CAM_FRONT"], require_all_cams=True,
        hdmap_dir=str(BASE / "data/preproc/hdmap_bev"),
        bbox_jsonl=str(BASE / "data/preproc/bbox_egostate_val.jsonl"),
        split="val", modality_dropout_p=0.0,
    )
    s = ds[0]
    # B.5'' is video first
    result = assemble_mm_embedding(
        hf_model=model,
        pixel_values=s["pixel_values"],
        image_grid_thw=s["image_grid_thw"],
        pixel_values_videos=s["pixel_values_videos"],
        video_grid_thw=s["video_grid_thw"],
        modality_order=("video", "image"),
    )
    e = result["mm_embedding"]
    print(f"[1.5] mm_embedding shape: {tuple(e.shape)} dtype: {e.dtype}")
    print(f"[1.5] video tokens: {result['video_mm_tokens']}, image tokens: {result['image_mm_tokens']}")
    expected_mm = result["video_mm_tokens"] + result["image_mm_tokens"]
    expected_dim = model.config.text_config.hidden_size * (
        1 + len(model.config.vision_config.deepstack_visual_indexes)
    )
    print(f"[1.5] expected: ({expected_mm}, {expected_dim})")
    assert tuple(e.shape) == (expected_mm, expected_dim), f"shape mismatch"
    print(f"[1.5] PASS")
    torch.save(result, BASE / "deploy/multimodal_trt/_phase1_5_mm_embedding.pt")
    print(f"[1.5] saved → deploy/multimodal_trt/_phase1_5_mm_embedding.pt")


if __name__ == "__main__":
    smoke()
