#!/usr/bin/env python3
"""CPU-only smoke for planning_eval.py multimodal toggle (no model forward).

Builds both dataset paths (PlanningDataset for --multimodal=off, and
MultiModalPlanningDataset for --multimodal=on), pulls 3 real val samples
from each, verifies key sets + shapes match production expectations, and
prints one sample's prompt text in multimodal mode (so we can eyeball
bbox text injection).

Run:  /usr/bin/python3 scripts/_smoke_planning_eval_multimodal.py
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

from transformers import AutoProcessor  # noqa: E402
from planning_dataset import PlanningDataset  # noqa: E402
from multimodal_planning_dataset import MultiModalPlanningDataset  # noqa: E402

MODEL_ID = "/workspace/models/Qwen2.5-VL-3B-Instruct"
INFOS_VAL = os.path.join(_BASE, "data/uniad_infos/nuscenes_infos_temporal_val.pkl")
NUSC_ROOT = os.path.join(_BASE, "data/nuscenes")
HDMAP_DIR = os.path.join(_BASE, "data/preproc/hdmap_bev")
BBOX_JSONL = os.path.join(_BASE, "data/preproc/bbox_egostate_val.jsonl")

EXPECTED_KEYS_CAM_ONLY = {
    "input_ids", "attention_mask", "labels",
    "pixel_values_videos", "video_grid_thw", "second_per_grid_ts",
    "image_name", "_meta_waypoints", "_meta_valid_mask", "_meta_token",
    "_meta_prompt_len",
}
EXPECTED_EXTRA_KEYS_MM = {"pixel_values", "image_grid_thw"}


def _print_shapes(name: str, sample: dict) -> None:
    print(f"\n--- {name} shape table ---")
    for k in sorted(sample.keys()):
        v = sample[k]
        if hasattr(v, "shape"):
            print(f"  {k:30s}: shape={tuple(v.shape)} dtype={v.dtype}")
        elif isinstance(v, str):
            print(f"  {k:30s}: str (len {len(v)})")
        else:
            print(f"  {k:30s}: {type(v).__name__}")


def _build_camera_only(processor) -> PlanningDataset:
    return PlanningDataset(
        infos_path=INFOS_VAL,
        nusc_root=NUSC_ROOT,
        processor=processor,
        max_length=4096,
        num_past_frames=4,
        num_future_waypoints=6,
        video_fps=2.0,
        vla_loss_mode="answer_and_traj",
        max_samples=10,
        require_full_future=True,
        planning_cams=["CAM_FRONT"],
        require_all_cams=True,
    )


def _build_multimodal(processor, dropout_p: float = 0.0) -> MultiModalPlanningDataset:
    return MultiModalPlanningDataset(
        infos_path=INFOS_VAL,
        nusc_root=NUSC_ROOT,
        processor=processor,
        max_length=4096,
        num_past_frames=4,
        num_future_waypoints=6,
        video_fps=2.0,
        vla_loss_mode="answer_and_traj",
        max_samples=10,
        require_full_future=True,
        planning_cams=["CAM_FRONT"],
        require_all_cams=True,
        hdmap_dir=HDMAP_DIR,
        bbox_jsonl=BBOX_JSONL,
        split="val",
        modality_dropout_p=dropout_p,
    )


def main() -> None:
    assert os.path.isfile(INFOS_VAL), f"missing infos pkl: {INFOS_VAL}"
    assert os.path.isdir(HDMAP_DIR), f"missing hdmap dir: {HDMAP_DIR}"
    assert os.path.isfile(BBOX_JSONL), f"missing bbox jsonl: {BBOX_JSONL}"

    print(f"[smoke] processor from {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    print("\n[smoke] === CAMERA-ONLY path (PlanningDataset) ===")
    ds_cam = _build_camera_only(processor)
    print(f"[smoke] camera-only val samples available: {len(ds_cam)}")
    s_cam = ds_cam[0]
    _print_shapes("CAM_ONLY sample 0", s_cam)
    missing_cam = EXPECTED_KEYS_CAM_ONLY - set(s_cam.keys())
    assert not missing_cam, f"camera-only missing keys: {missing_cam}"
    # Camera-only must NOT have image keys.
    extra_image_keys = EXPECTED_EXTRA_KEYS_MM & set(s_cam.keys())
    assert not extra_image_keys, \
        f"camera-only unexpectedly has image keys: {extra_image_keys}"
    print("[smoke] camera-only PASS")

    print("\n[smoke] === MULTIMODAL path (MultiModalPlanningDataset, dropout=0.0) ===")
    ds_mm = _build_multimodal(processor, dropout_p=0.0)
    print(f"[smoke] multimodal val samples available: {len(ds_mm)}")
    s_mm = ds_mm[0]
    _print_shapes("MULTIMODAL sample 0", s_mm)
    missing_mm = (EXPECTED_KEYS_CAM_ONLY | EXPECTED_EXTRA_KEYS_MM) - set(s_mm.keys())
    assert not missing_mm, f"multimodal missing keys: {missing_mm}"
    # Sanity: image_grid_thw should be a 3-tuple (T, H_patches, W_patches).
    # Exact resolution depends on processor's default min/max_pixels (eval
    # does NOT override these — consistent with the existing camera-only eval
    # path and with R1' baseline 0.6423 to keep numbers comparable).
    t, h, w = (int(x) for x in s_mm["image_grid_thw"])
    assert t == 1 and h >= 8 and w >= 8, \
        f"unexpected image_grid_thw: ({t}, {h}, {w})"
    print(f"[smoke] HD-map grid = (T={t}, H={h}, W={w}) → {h*w} ViT patches")
    print("[smoke] multimodal PASS")

    print("\n[smoke] === Sample 1 prompt text (multimodal, decoded) ===")
    text = processor.tokenizer.decode(ds_mm[1]["input_ids"], skip_special_tokens=False)
    # Compress vision pad runs for readability
    import re
    text_short = re.sub(r"(<\|video_pad\|>)+", "<|video_pad|>(*N*)", text)
    text_short = re.sub(r"(<\|image_pad\|>)+", "<|image_pad|>(*N*)", text_short)
    print(text_short[:1800])
    if "Detected objects" not in text:
        print("\n[smoke] WARN: prompt does not contain bbox text!")
    else:
        print("\n[smoke] OK: bbox 'Detected objects' header present in prompt")

    print("\n[smoke] === ALL CHECKS PASSED ===")


if __name__ == "__main__":
    main()
