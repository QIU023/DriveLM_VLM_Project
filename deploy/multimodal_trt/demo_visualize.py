"""Demo: visualize all modalities + run TRT inference + overlay predicted trajectory.

For each chosen val sample:
  1) Build dataset sample (tensors for TRT inference + raw paths for visualization)
  2) Compute Phase 1.5 embed + Phase 2 mrope_config
  3) Run TRT.generate(max_tokens=14, greedy) — same pipeline as bench
  4) Decode trajectory tokens → (T=6, 2) waypoints in meters
  5) Plot: 4 cam frames | HD-map BEV | bbox/ego text panel | GT-vs-pred trajectory overlay
  6) Save → deploy/multimodal_trt/demos/sample_{idx}.png

Output: PNG per sample + JSON summary.
"""
from __future__ import annotations

import argparse
import importlib.metadata as _md
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
_orig_files = _md.files
def _files_shim(name):
    try:
        return _orig_files(name)
    except _md.PackageNotFoundError:
        if "tileiras" in name:
            return None
        raise
_md.files = _files_shim

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

_HERE = Path(__file__).resolve().parent
_BASE = _HERE.parent.parent
sys.path.insert(0, str(_BASE / "scripts"))
sys.path.insert(0, str(_HERE))


def build_unexpanded(input_ids, prompt_len, image_pad_id, video_pad_id, tok):
    new_ids = []
    i = 0
    ids = input_ids[:prompt_len].tolist()
    while i < len(ids):
        t = ids[i]
        if t == image_pad_id:
            new_ids.append(image_pad_id)
            while i < len(ids) and ids[i] == image_pad_id: i += 1
        elif t == video_pad_id:
            new_ids.append(image_pad_id)
            while i < len(ids) and ids[i] == video_pad_id: i += 1
        else:
            new_ids.append(t); i += 1
    return tok.decode(new_ids, skip_special_tokens=False), new_ids


def plot_sample(sample_idx, cam_pils, hdmap_pil, bbox_text, ego_text,
                pred_wp, gt_wp, pred_tokens, decoded_text, ttft_ms, full_ms, out_path):
    """8-panel figure: 4 cam frames + HD-map + bbox text + ego text + traj overlay."""
    fig = plt.figure(figsize=(20, 11))
    gs = fig.add_gridspec(3, 5, height_ratios=[1.2, 1.2, 0.8], hspace=0.35, wspace=0.25)

    # Row 1: 4 cam frames
    for i, pil in enumerate(cam_pils[:4]):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(np.asarray(pil))
        ax.set_title(f"CAM_FRONT t-{3-i} ({(i*0.5):.1f}s ago)", fontsize=10)
        ax.axis("off")
    # Row 1 last cell: parameters
    ax = fig.add_subplot(gs[0, 4])
    ax.axis("off")
    ax.text(0.05, 0.95, f"Sample idx: {sample_idx}", fontsize=11, fontweight="bold", va="top")
    ax.text(0.05, 0.85, f"Backend: TRT-LLM 1.3 (mm-disagg)", fontsize=9, va="top")
    ax.text(0.05, 0.78, f"Model: Qwen3-VL-4B B.5'' VLA", fontsize=9, va="top")
    ax.text(0.05, 0.71, f"TTFT: {ttft_ms:.1f} ms", fontsize=10, color="#0a6", va="top")
    ax.text(0.05, 0.64, f"Full 14-tok: {full_ms:.1f} ms", fontsize=10, color="#0a6", va="top")
    ax.text(0.05, 0.55, f"Pred tokens (first 6):", fontsize=9, va="top")
    ax.text(0.05, 0.48, f"  {pred_tokens[:6]}", fontsize=8, family="monospace", va="top")
    ax.text(0.05, 0.40, f"Pred text:", fontsize=9, va="top")
    txt = decoded_text[:200] + ("..." if len(decoded_text) > 200 else "")
    ax.text(0.05, 0.33, f"  {txt!r}", fontsize=7, family="monospace", va="top", wrap=True)

    # Row 2 col 0: HD-map BEV
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(np.asarray(hdmap_pil))
    ax.set_title("HD-map BEV (224x224)", fontsize=10)
    ax.axis("off")

    # Row 2 col 1-2: bbox text panel
    ax = fig.add_subplot(gs[1, 1:3])
    ax.axis("off")
    ax.set_title("Detected objects (text input to LM)", fontsize=10, loc="left")
    txt = (bbox_text[:1100] + "...") if len(bbox_text) > 1100 else bbox_text
    ax.text(0.02, 0.97, txt, fontsize=7, family="monospace", va="top", wrap=True)

    # Row 2 col 3: ego state + prompt
    ax = fig.add_subplot(gs[1, 3])
    ax.axis("off")
    ax.set_title("Ego state + prompt", fontsize=10, loc="left")
    ax.text(0.02, 0.97, ego_text[:600], fontsize=8, family="monospace", va="top", wrap=True)

    # Row 2 col 4: trajectory overlay on BEV-like axes (top-down, ego at origin)
    ax = fig.add_subplot(gs[1, 4])
    ax.set_aspect("equal")
    ax.set_xlabel("right (m)"); ax.set_ylabel("forward (m)")
    ax.set_title("Predicted trajectory (next 3s)", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
    ax.scatter([0], [0], c="black", s=60, marker="^", label="ego", zorder=5)
    if gt_wp is not None and len(gt_wp) > 0:
        ax.plot([0] + list(gt_wp[:, 1]), [0] + list(gt_wp[:, 0]), "o-", c="#2a8", label="GT", lw=2, ms=4)
    if pred_wp is not None and len(pred_wp) > 0:
        ax.plot([0] + list(pred_wp[:, 1]), [0] + list(pred_wp[:, 0]), "o--", c="#e63", label="Pred", lw=2, ms=4)
    rng = 30
    ax.set_xlim(-rng, rng); ax.set_ylim(-5, rng + 5)
    ax.legend(loc="upper left", fontsize=8)

    # Row 3: input tally
    ax = fig.add_subplot(gs[2, :])
    ax.axis("off")
    ax.text(0.01, 0.7,
            f"Input payload: 4 × CAM_FRONT frames @ 2Hz (1280x720)  +  1 × HD-map BEV (224x224)  "
            f"+  {len(bbox_text.split(chr(10)))} bbox text lines  +  ego speed/yaw  +  planning prompt",
            fontsize=10, va="top")
    ax.text(0.01, 0.45,
            f"Output: 14 trajectory tokens → 6 future waypoints (Δt = 0.5s each → 3s horizon)  |  "
            f"GT: green  |  Pred: orange (dashed)",
            fontsize=10, va="top")
    if gt_wp is not None and pred_wp is not None and len(gt_wp) == len(pred_wp):
        l2 = np.linalg.norm(gt_wp - pred_wp, axis=1).mean()
        ax.text(0.01, 0.20, f"L2 error vs GT: {l2:.3f} m  (mean across 6 waypoints)",
                fontsize=11, fontweight="bold", color="#0a6" if l2 < 1.0 else "#e63", va="top")

    fig.suptitle(f"B.5'' Qwen3-VL VLA — TRT-LLM 1.3 deploy demo — sample {sample_idx}",
                 fontsize=13, fontweight="bold")
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(_BASE / "checkpoints_qwen25/nusc_planning_b5pp_1cam_qwen3vl_multimodal/final"))
    ap.add_argument("--sample-indices", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out-dir", default=str(_HERE / "demos"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForImageTextToText, AutoProcessor, AutoConfig
    from multimodal_planning_dataset import MultiModalPlanningDataset
    from phase1_5_vision_embeds import assemble_mm_embedding
    from phase2_mrope_config import build_mrope_config
    from trajectory_tokenizer import TrajectoryTokenizer, TrajectoryTokenizerConfig
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.disaggregated_params import DisaggregatedParams
    from tensorrt_llm._torch.shared_tensor import SharedTensorContainer
    from tensorrt_llm.llmapi import KvCacheConfig

    print(f"[demo] loading HF model for vision pre-compute ...")
    hf_model = AutoModelForImageTextToText.from_pretrained(
        args.ckpt, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda:0").eval()
    proc = AutoProcessor.from_pretrained(args.ckpt)
    tok = proc.tokenizer
    image_pad_id = tok.convert_tokens_to_ids("<|image_pad|>")
    video_pad_id = tok.convert_tokens_to_ids("<|video_pad|>")
    traj_tok = TrajectoryTokenizer(TrajectoryTokenizerConfig())
    traj_tok.register_with_tokenizer(tok) if hasattr(traj_tok, "register_with_tokenizer") else None

    print(f"[demo] building dataset")
    ds = MultiModalPlanningDataset(
        infos_path=str(_BASE / "data/uniad_infos/nuscenes_infos_temporal_val.pkl"),
        nusc_root=str(_BASE / "data/nuscenes"),
        processor=proc, max_length=12288,
        num_past_frames=4, num_future_waypoints=6, video_fps=2.0,
        vla_loss_mode="answer_and_traj", max_samples=max(args.sample_indices) + 1,
        require_full_future=True,
        planning_cams=["CAM_FRONT"], require_all_cams=True,
        hdmap_dir=str(_BASE / "data/preproc/hdmap_bev"),
        bbox_jsonl=str(_BASE / "data/preproc/bbox_egostate_val.jsonl"),
        split="val", modality_dropout_p=0.0,
    )

    # Pre-compute embeds + mrope + raw PILs for each sample (BEFORE TRT load)
    sample_packets = []
    for idx in args.sample_indices:
        print(f"[demo] === sample {idx}: pre-compute ===")
        s = ds[idx]
        prompt_len = int(s["_meta_prompt_len"])
        eres = assemble_mm_embedding(
            hf_model=hf_model,
            pixel_values=s["pixel_values"],
            image_grid_thw=s["image_grid_thw"],
            pixel_values_videos=s["pixel_values_videos"],
            video_grid_thw=s["video_grid_thw"],
            modality_order=("video", "image"),
        )
        n_video = eres["video_mm_tokens"]
        n_image = eres["image_mm_tokens"]
        mm_full = eres["mm_embedding"].to("cuda:0", dtype=torch.bfloat16)
        # Count blocks per orig prompt
        orig_ids = s["input_ids"][:prompt_len].tolist()
        n_vid_blk = 0; n_img_blk = 0; j = 0
        while j < len(orig_ids):
            t = orig_ids[j]
            if t == video_pad_id:
                n_vid_blk += 1
                while j < len(orig_ids) and orig_ids[j] == video_pad_id: j += 1
            elif t == image_pad_id:
                n_img_blk += 1
                while j < len(orig_ids) and orig_ids[j] == image_pad_id: j += 1
            else:
                j += 1
        vchunk = n_video // n_vid_blk
        ichunk = n_image // n_img_blk
        video_embed = mm_full[:n_video].contiguous()
        image_embed = mm_full[n_video:].contiguous()
        video_chunks = [video_embed[i*vchunk:(i+1)*vchunk].contiguous() for i in range(n_vid_blk)]
        image_chunks = [image_embed[i*ichunk:(i+1)*ichunk].contiguous() for i in range(n_img_blk)]
        # Build handles in orig text order
        mm_tensors = []
        vi = 0; ii = 0; k = 0
        while k < len(orig_ids):
            t = orig_ids[k]
            if t == video_pad_id:
                mm_tensors.append(video_chunks[vi]); vi += 1
                while k < len(orig_ids) and orig_ids[k] == video_pad_id: k += 1
            elif t == image_pad_id:
                mm_tensors.append(image_chunks[ii]); ii += 1
                while k < len(orig_ids) and orig_ids[k] == image_pad_id: k += 1
            else:
                k += 1
        # Mrope
        mrope = build_mrope_config(
            model_config=hf_model.config,
            input_ids=s["input_ids"], mm_token_type_ids=s["mm_token_type_ids"],
            image_grid_thw=s["image_grid_thw"].clone(),
            video_grid_thw=s["video_grid_thw"].clone(),
            attention_mask=s["attention_mask"],
        )
        mrope_pos_ids = mrope["mrope_position_ids"][:, :, :prompt_len].to("cuda:0", dtype=torch.int32).contiguous()
        mrope_deltas = mrope["mrope_position_deltas"].view(-1).to("cuda:0", dtype=torch.int32).contiguous()
        text_prompt, _ = build_unexpanded(s["input_ids"], prompt_len, image_pad_id, video_pad_id, tok)
        # Raw PILs for viz: re-load frames + HD map
        from multimodal_planning_dataset import _build_user_content_multimodal
        base_idx = ds._keep[idx]
        info = ds.infos[base_idx]
        sample_token = info["token"]
        hist = ds._walk_history(base_idx)
        cam_clip = ds._load_frames(hist, "CAM_FRONT")
        hdmap_pil = ds._load_hdmap(sample_token)
        bbox_text = ds._lookup_bbox(sample_token) or "(no detections)"
        from multimodal_planning_dataset import _format_ego_speed_preamble, _multicam_prompt_suffix
        ego_preamble = _format_ego_speed_preamble(info)
        prompt_suffix = _multicam_prompt_suffix(["CAM_FRONT"])
        ego_text = ego_preamble + "\n\n" + prompt_suffix.strip()
        # GT waypoints from dataset
        wp_gt, _ = ds._compute_waypoints(base_idx)
        sample_packets.append(dict(
            idx=idx, sample_token=sample_token,
            mm_tensors=mm_tensors, mrope_pos_ids=mrope_pos_ids, mrope_deltas=mrope_deltas,
            text_prompt=text_prompt, cam_pils=cam_clip, hdmap_pil=hdmap_pil,
            bbox_text=bbox_text, ego_text=ego_text, gt_wp=wp_gt,
            n_video=n_video, n_image=n_image,
        ))

    # Free HF, load TRT
    print(f"[demo] === free HF, load TRT ===")
    del hf_model
    torch.cuda.empty_cache()
    import gc; gc.collect()

    llm = LLM(
        model=args.ckpt, max_batch_size=1, max_seq_len=2048, max_num_tokens=2048,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.5),
        trust_remote_code=True,
    )
    sp = SamplingParams(max_tokens=14, temperature=0.0)
    print(f"[demo] TRT loaded; running {len(sample_packets)} samples")

    summary = []
    for pkt in sample_packets:
        print(f"[demo] --- run sample {pkt['idx']} ---")
        mm_handles = [SharedTensorContainer.from_tensor(t).dump_to_dict() for t in pkt["mm_tensors"]]
        mrope_pos_h = SharedTensorContainer.from_tensor(pkt["mrope_pos_ids"]).dump_to_dict()
        mrope_delta_h = SharedTensorContainer.from_tensor(pkt["mrope_deltas"]).dump_to_dict()
        disagg = DisaggregatedParams(
            request_type="context_and_generation",
            multimodal_embedding_handles=mm_handles,
            mrope_position_ids_handle=mrope_pos_h,
            mrope_position_deltas_handle=mrope_delta_h,
        )
        # Warmup once for this sample
        _ = llm.generate([{"prompt": pkt["text_prompt"]}], sampling_params=sp, disaggregated_params=disagg)
        # Time TTFT (1 tok) + full (14 tok)
        sp_one = SamplingParams(max_tokens=1, temperature=0.0)
        t0 = time.perf_counter(); _ = llm.generate([{"prompt": pkt["text_prompt"]}], sampling_params=sp_one, disaggregated_params=disagg)
        ttft_ms = 1000 * (time.perf_counter() - t0)
        t0 = time.perf_counter()
        out = llm.generate([{"prompt": pkt["text_prompt"]}], sampling_params=sp, disaggregated_params=disagg)
        full_ms = 1000 * (time.perf_counter() - t0)
        pred_token_ids = list(out[0].outputs[0].token_ids)
        pred_text = out[0].outputs[0].text
        pred_wp = traj_tok.decode(pred_token_ids)
        gt_wp = np.asarray(pkt["gt_wp"], dtype=np.float32)
        # Trim/pad to compare
        n_compare = min(len(gt_wp), len(pred_wp))
        l2 = float(np.linalg.norm(gt_wp[:n_compare] - pred_wp[:n_compare], axis=1).mean()) if n_compare > 0 else float("nan")
        print(f"[demo]   pred tokens: {pred_token_ids[:8]}...")
        print(f"[demo]   pred waypoints (m): {pred_wp.tolist()}")
        print(f"[demo]   gt waypoints (m): {gt_wp.tolist()}")
        print(f"[demo]   L2 error: {l2:.3f} m   TTFT: {ttft_ms:.1f} ms   full: {full_ms:.1f} ms")
        # Plot
        out_path = out_dir / f"sample_{pkt['idx']:02d}.png"
        plot_sample(
            pkt["idx"], pkt["cam_pils"], pkt["hdmap_pil"], pkt["bbox_text"], pkt["ego_text"],
            pred_wp, gt_wp, pred_token_ids, pred_text, ttft_ms, full_ms, str(out_path),
        )
        print(f"[demo]   saved → {out_path}")
        summary.append(dict(
            idx=pkt["idx"], sample_token=pkt["sample_token"],
            pred_token_ids=pred_token_ids, pred_waypoints_m=pred_wp.tolist(),
            gt_waypoints_m=gt_wp.tolist(), l2_mean_m=l2,
            ttft_ms=ttft_ms, full_ms=full_ms,
            png=str(out_path),
        ))

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({"backend": "TRT-LLM 1.3 mm-disagg", "ckpt": args.ckpt, "samples": summary}, f, indent=2)
    print(f"\n[demo] summary → {summary_path}")
    print(f"[demo] all PNGs in: {out_dir}/")


if __name__ == "__main__":
    main()
