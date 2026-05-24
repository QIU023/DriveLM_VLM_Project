"""Phase 1 — HF vision tower pre-compute for B.5'' Qwen3-VL.

Goal: load real val sample (video + HD-map image), run HF Qwen3-VL vision tower
on both, split into base + 3-level deepstack, concat to (mm_total, hidden*4)
shape required by TRT-LLM's fuse_input_embeds cache-hit path.

Gate: embed shape == (484+144, 2560*4) = (628, 10240).

Run:
  /usr/bin/python3 deploy/multimodal_trt/phase1_vision_precompute.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_BASE = _HERE.parent.parent
sys.path.insert(0, str(_BASE / "scripts"))


def main():
    ckpt = str(_BASE / "checkpoints_qwen25/nusc_planning_b5pp_1cam_qwen3vl_multimodal/final")
    print(f"[phase1] ckpt={ckpt}")

    from transformers import AutoModelForImageTextToText, AutoProcessor
    from multimodal_planning_dataset import MultiModalPlanningDataset

    print(f"[phase1] loading HF model ...")
    t0 = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(
        ckpt, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda:0").eval()
    proc = AutoProcessor.from_pretrained(ckpt)
    print(f"[phase1] loaded in {time.perf_counter()-t0:.1f}s")

    # Inspect vision tower
    vt = model.visual if hasattr(model, "visual") else model.model.visual
    print(f"[phase1] vision tower class: {type(vt).__name__}")
    print(f"[phase1] vision tower has deepstack? "
          f"{hasattr(vt, 'deepstack_visual_indexes') or hasattr(model.config.vision_config, 'deepstack_visual_indexes')}")
    print(f"[phase1] deepstack_visual_indexes: "
          f"{getattr(model.config.vision_config, 'deepstack_visual_indexes', None)}")
    print(f"[phase1] vision hidden: {model.config.vision_config.hidden_size}, "
          f"out_hidden: {model.config.vision_config.out_hidden_size}, "
          f"LM hidden: {model.config.text_config.hidden_size}")

    print(f"[phase1] building val sample ...")
    ds = MultiModalPlanningDataset(
        infos_path=str(_BASE / "data/uniad_infos/nuscenes_infos_temporal_val.pkl"),
        nusc_root=str(_BASE / "data/nuscenes"),
        processor=proc, max_length=12288,
        num_past_frames=4, num_future_waypoints=6, video_fps=2.0,
        vla_loss_mode="answer_and_traj", max_samples=2, require_full_future=True,
        planning_cams=["CAM_FRONT"], require_all_cams=True,
        hdmap_dir=str(_BASE / "data/preproc/hdmap_bev"),
        bbox_jsonl=str(_BASE / "data/preproc/bbox_egostate_val.jsonl"),
        split="val", modality_dropout_p=0.0,
    )
    sample = ds[0]
    pixel_values = sample["pixel_values"].to("cuda:0", dtype=torch.bfloat16)
    pixel_values_videos = sample["pixel_values_videos"].to("cuda:0", dtype=torch.bfloat16)
    image_grid_thw = sample["image_grid_thw"].to("cuda:0")
    video_grid_thw = sample["video_grid_thw"].to("cuda:0")
    if image_grid_thw.dim() == 1:
        image_grid_thw = image_grid_thw.unsqueeze(0)
    if video_grid_thw.dim() == 1:
        video_grid_thw = video_grid_thw.unsqueeze(0)
    print(f"[phase1] sample shapes:")
    print(f"  pixel_values: {tuple(pixel_values.shape)} {pixel_values.dtype}")
    print(f"  pixel_values_videos: {tuple(pixel_values_videos.shape)} {pixel_values_videos.dtype}")
    print(f"  image_grid_thw: {tuple(image_grid_thw.shape)} {image_grid_thw.tolist()}")
    print(f"  video_grid_thw: {tuple(video_grid_thw.shape)} {video_grid_thw.tolist()}")

    # Run vision tower forward — HF Qwen3VL visual returns (hidden, deepstack_features_list)
    print(f"[phase1] running vision tower on image ...")
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        # HF Qwen3VL visual returns tuple (embeds, deepstack_features) where deepstack_features
        # is a list of len(deepstack_visual_indexes) tensors, each (seq_len, vision_hidden)
        image_out = vt(pixel_values, grid_thw=image_grid_thw)
        video_out = vt(pixel_values_videos, grid_thw=video_grid_thw)

    print(f"[phase1] image vision out type: {type(image_out)}")
    if isinstance(image_out, tuple):
        print(f"  image_embeds: {tuple(image_out[0].shape)}")
        if len(image_out) > 1 and image_out[1] is not None:
            ds_feat = image_out[1]
            if isinstance(ds_feat, list):
                print(f"  deepstack: list of {len(ds_feat)} tensors, each shape "
                      f"{tuple(ds_feat[0].shape)}")
            else:
                print(f"  deepstack tensor: {tuple(ds_feat.shape)}")
    else:
        print(f"  image_out shape: {tuple(image_out.shape) if hasattr(image_out, 'shape') else 'N/A'}")

    print(f"[phase1] video vision out type: {type(video_out)}")
    if isinstance(video_out, tuple):
        print(f"  video_embeds: {tuple(video_out[0].shape)}")
        if len(video_out) > 1 and video_out[1] is not None:
            ds_feat = video_out[1]
            if isinstance(ds_feat, list):
                print(f"  deepstack: list of {len(ds_feat)} tensors, each shape "
                      f"{tuple(ds_feat[0].shape)}")
            else:
                print(f"  deepstack tensor: {tuple(ds_feat.shape)}")

    # Save for Phase 2/3 consumption
    out_path = _BASE / "deploy/multimodal_trt/_phase1_embeds.pt"
    torch.save({
        "image_out": image_out,
        "video_out": video_out,
        "image_grid_thw": image_grid_thw,
        "video_grid_thw": video_grid_thw,
        "sample_meta": {
            "prompt_len": int(sample["_meta_prompt_len"]),
            "input_ids_shape": tuple(sample["input_ids"].shape),
            "mm_token_type_ids_shape": tuple(sample["mm_token_type_ids"].shape),
        },
    }, out_path)
    print(f"\n[phase1] saved embeds + meta → {out_path}")
    print(f"[phase1] DONE. Inspect output above; next: cat base+deepstack on dim=1 for TRT.")


if __name__ == "__main__":
    main()
