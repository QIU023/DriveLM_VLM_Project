"""Phase 4 prep — capture HF full-multimodal prefill last-token logits for B.5''.

This produces the GROUND-TRUTH top-5 token predictions that the TRT engine
(with pre-computed embeds injected) must match for the parity gate.

Outputs JSON with:
  - top5_token_ids (list of 5 int)
  - top5_logits (list of 5 float)
  - top5_decoded (list of 5 str)
  - prefill_last_token_logits (list of vocab_size float, gzipped if too big)
  - sample meta (prompt_len, shapes)

Usage:
  /usr/bin/python3 deploy/multimodal_trt/phase4_hf_baseline_logits.py

Run AFTER Phase 1, Agent A (mrope), Agent B (inject hook) all land.
This script ITSELF does not depend on either agent — it's a pure HF baseline.
"""
from __future__ import annotations

import json
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
    out_path = _HERE / "_phase4_hf_baseline_logits.json"

    from transformers import AutoModelForImageTextToText, AutoProcessor
    from multimodal_planning_dataset import MultiModalPlanningDataset

    print(f"[phase4-hf] loading HF model {ckpt}")
    t0 = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(
        ckpt, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda:0").eval()
    proc = AutoProcessor.from_pretrained(ckpt)
    tok = proc.tokenizer
    print(f"[phase4-hf] loaded in {time.perf_counter()-t0:.1f}s")

    print(f"[phase4-hf] building val sample[0]")
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
    prompt_len = int(sample["_meta_prompt_len"])
    input_ids = sample["input_ids"].unsqueeze(0)[:, :prompt_len].to("cuda:0")
    attention_mask = sample["attention_mask"].unsqueeze(0)[:, :prompt_len].to("cuda:0")
    mm_token_type_ids = sample["mm_token_type_ids"].unsqueeze(0)[:, :prompt_len].to("cuda:0")
    pixel_values = sample["pixel_values"].to("cuda:0", dtype=torch.bfloat16)
    pixel_values_videos = sample["pixel_values_videos"].to("cuda:0", dtype=torch.bfloat16)
    image_grid_thw = sample["image_grid_thw"].to("cuda:0")
    video_grid_thw = sample["video_grid_thw"].to("cuda:0")
    if image_grid_thw.dim() == 1:
        image_grid_thw = image_grid_thw.unsqueeze(0)
    if video_grid_thw.dim() == 1:
        video_grid_thw = video_grid_thw.unsqueeze(0)
    print(f"[phase4-hf] prompt_len={prompt_len}")
    print(f"[phase4-hf] input_ids: {tuple(input_ids.shape)}")

    print(f"[phase4-hf] running HF full multimodal prefill forward")
    fwd_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        pixel_values_videos=pixel_values_videos,
        video_grid_thw=video_grid_thw,
        mm_token_type_ids=mm_token_type_ids,
        use_cache=False,
    )
    with torch.no_grad():
        out = model(**fwd_kwargs)

    last_logits = out.logits[0, -1, :].float().cpu()
    top5_v, top5_i = last_logits.topk(5)
    top5_decoded = [tok.decode([int(i)]) for i in top5_i.tolist()]

    print(f"[phase4-hf] last-token top-5:")
    for v, i, d in zip(top5_v.tolist(), top5_i.tolist(), top5_decoded):
        print(f"  id={i:6d}  logit={v:+.4f}  decode={d!r}")

    result = {
        "ckpt": ckpt,
        "sample_idx": 0,
        "prompt_len": prompt_len,
        "vocab_size": int(last_logits.shape[0]),
        "top5_token_ids": [int(i) for i in top5_i.tolist()],
        "top5_logits": [float(v) for v in top5_v.tolist()],
        "top5_decoded": top5_decoded,
        "last_token_logits_full": last_logits.tolist(),
    }
    with open(out_path, "w") as f:
        json.dump(result, f)
    print(f"[phase4-hf] saved → {out_path}")


if __name__ == "__main__":
    main()
