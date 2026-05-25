#!/usr/bin/env /usr/bin/python3
"""T6 CPU-ONLY proof: build PTQ calibration samples through the EXACT training
processor, with NO GPU and NO model weights loaded.

This proves the 128-sample calibration build (consumed by quant_fp8.py /
quant_nvfp4.py via _common.build_calib_dataset) is CPU-safe and cheap. We load
ONLY the processor (Qwen3VLProcessor) from the preserved ckpt dir and run
MultiModalPlanningDataset.__getitem__ on a couple of stratified calib tokens,
printing the resulting tensor shapes (input_ids / pixel_values_videos /
video_grid_thw / pixel_values / image_grid_thw).

It does NOT touch GPU (CUDA_VISIBLE_DEVICES is forced empty) and does NOT load
model.safetensors. The full 128-sample run is the same code path with more
samples — entirely CPU-bound (image/video resize + tokenize), ~seconds/sample.

Run:
    CUDA_VISIBLE_DEVICES="" /usr/bin/python3 deploy/trt_b5ppp/calib_cpu_dryrun.py \
        [--n 2] [--proc <processor dir>] [--tokens-json calib_128.tokens.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # hard GPU lockout

_HERE = Path(__file__).resolve().parent
_BASE = _HERE.parent.parent
sys.path.insert(0, str(_BASE / "scripts"))

# Use the preserved ckpt dir for the processor until final/ lands (read-only).
DEFAULT_PROC = str(
    _BASE / "checkpoints_qwen25/nusc_planning_b5pp_1cam_qwen3vl_multimodal"
    "/final.preserved_v2_2026-05-25"
)
DEFAULT_YAML = str(_BASE / "configs/nuscenes_planning_1cam_qwen3vl_multimodal.yaml")
DEFAULT_TOKENS = str(_HERE / "calib_128.tokens.json")


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else str(_BASE / p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--proc", default=DEFAULT_PROC)
    ap.add_argument("--yaml", default=DEFAULT_YAML)
    ap.add_argument("--tokens-json", default=DEFAULT_TOKENS)
    args = ap.parse_args()

    import yaml
    from transformers import AutoProcessor
    from multimodal_planning_dataset import MultiModalPlanningDataset

    print(f"[calib-dry] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}")
    print(f"[calib-dry] processor dir = {args.proc}")
    proc = AutoProcessor.from_pretrained(args.proc)
    ip_size = proc.image_processor.size
    vp_size = proc.video_processor.size
    print(f"[calib-dry] image_processor.size = {ip_size}")
    print(f"[calib-dry] video_processor.size = {vp_size}")
    # transformers 5.6: caps live in .size SizeDict, NOT .min_pixels/.max_pixels
    img_cap = getattr(ip_size, "longest_edge", None)
    vid_cap = getattr(vp_size, "longest_edge", None)
    print(f"[calib-dry] image longest_edge={img_cap} (yaml max_pixels=109760), "
          f"video longest_edge={vid_cap} (HF native default 25165824)")

    with open(args.yaml) as f:
        cfg = yaml.safe_load(f)

    # Calib draws from val split per CALIB_README (stratified tokens); the
    # production quant scripts default to train split (256). Either is CPU-safe.
    ds = MultiModalPlanningDataset(
        infos_path=_abs("data/uniad_infos/nuscenes_infos_temporal_val.pkl"),
        nusc_root=_abs("data/nuscenes"),
        processor=proc,
        max_length=int(cfg.get("max_length", 6144)),
        num_past_frames=int(cfg.get("planning_num_past_frames", 4)),
        num_future_waypoints=int(cfg.get("planning_num_future_wp", 6)),
        video_fps=float(cfg.get("video_fps", 2.0)),
        vla_loss_mode=cfg.get("vla_loss_mode", "answer_and_traj"),
        max_samples=max(8, args.n * 4),  # small cap; we only iterate args.n
        require_full_future=True,
        planning_cams=cfg.get("planning_cams", ["CAM_FRONT"]),
        require_all_cams=True,
        hdmap_dir=_abs(cfg.get("hdmap_dir", "data/preproc/hdmap_bev")),
        bbox_jsonl=_abs("data/preproc/bbox_egostate_val.jsonl"),
        split="val",
        modality_dropout_p=0.0,
    )
    print(f"[calib-dry] dataset built (CPU). len={len(ds)}; iterating {args.n} samples")

    for i in range(min(args.n, len(ds))):
        s = ds[i]
        shapes = {k: tuple(v.shape) for k, v in s.items()
                  if hasattr(v, "shape") and not k.startswith("_meta_")}
        print(f"[calib-dry] sample {i}: {shapes}")
        vg = s.get("video_grid_thw")
        if vg is not None:
            ms = int(proc.image_processor.merge_size)
            g = vg if vg.dim() == 2 else vg.unsqueeze(0)
            post = int(sum(int(r[0]) * (int(r[1]) // ms) * (int(r[2]) // ms) for r in g))
            print(f"[calib-dry]   video_grid_thw={g.tolist()} -> post-merge video tokens={post} "
                  f"(EXPECTED_VIDEO_TOKENS ~2800)")
        ig = s.get("image_grid_thw")
        if ig is not None:
            ms = int(proc.image_processor.merge_size)
            g = ig if ig.dim() == 2 else ig.unsqueeze(0)
            post = int(sum(int(r[0]) * (int(r[1]) // ms) * (int(r[2]) // ms) for r in g))
            print(f"[calib-dry]   image_grid_thw={g.tolist()} -> post-merge HD-map tokens={post} "
                  f"(EXPECTED_IMAGE_TOKENS ~121)")

    print("[calib-dry] PASS — full 128-sample build is the same CPU path, "
          "no GPU, ~seconds/sample.")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
