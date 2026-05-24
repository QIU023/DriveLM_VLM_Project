"""Phase 5 — TRT-LLM 1.3 multimodal bench on B.5'' Qwen3-VL-4B.

End-to-end:
  1. Load HF model (vision tower + tokenizer)
  2. Load TRT-LLM engine
  3. Build real val sample[0]
  4. Compute mm_embedding via Phase 1.5 (HF vision + deepstack concat, video-first)
  5. Compute mrope_config via Phase 2 (HF get_rope_index)
  6. Construct UNEXPANDED text prompt (single <|video_pad|> + single <|image_pad|>)
  7. Generate via LLM(..., disaggregated_params=...)
  8. Phase 4 gate: top-1 token must match HF baseline (151934)
  9. If gate PASS, run TTFT/decode/throughput bench
 10. Save → deploy/trt_bench/B5pp_trt_qwen3vl_multimodal_bf16.json

ABORT criteria:
  - Gate top-1 mismatch → embed/mrope ordering wrong, surface honestly
  - Engine raises → surface
"""
from __future__ import annotations

import argparse
import importlib.metadata as _md
import json
import os
import sys
import time
from pathlib import Path

# Workaround: nvidia-cuda-tileiras metadata missing on RTX 5090 (compute 12.0)
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

import torch  # noqa: E402

_HERE = Path(__file__).resolve().parent
_BASE = _HERE.parent.parent
sys.path.insert(0, str(_BASE / "scripts"))
sys.path.insert(0, str(_HERE))


def percentile(values, p):
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    f = int(k); c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f) if f != c else s[f]


def build_unexpanded_prompt(input_ids, prompt_len, tokenizer, image_pad_id, video_pad_id, vision_start_id, vision_end_id):
    """Collapse expanded image/video pad runs back to single pads with surrounding vision_start/end.

    The dataset gives expanded input_ids (121 image pads, 36 video pads). TRT
    needs unexpanded form (one pad per mm block) — engine will re-expand based
    on embed.shape[0]. We rebuild by:
      - Walk input_ids[:prompt_len]
      - When we hit a run of image_pad: emit ONE <|image_pad|> (assume surrounded by vision_start/end already)
      - When we hit a run of video_pad: emit ONE <|video_pad|>
      - Otherwise: emit the original token
    Then decode.
    """
    # WORKAROUND: TRT-LLM 1.3.0rc15's Qwen3VLInputProcessorBase.get_prompt_token_ids
    # only counts `image_token_id` placeholders (line 443 modeling_qwen3vl.py:
    # "TODO: what about video_token_id?"). Video pads aren't supported in disagg.
    # We collapse runs AND convert every video_pad to image_pad. The LM doesn't
    # care which placeholder id is used (fuse_input_embeds replaces the embed
    # entirely with our pre-computed mm_embedding), and M-RoPE position_ids are
    # also pre-computed (Phase 2) using the ORIGINAL mm_token_type_ids — so the
    # spatial-temporal semantics survive even though TRT sees only image_pads.
    new_ids = []
    i = 0
    ids = input_ids[:prompt_len].tolist()
    while i < len(ids):
        t = ids[i]
        if t == image_pad_id:
            new_ids.append(image_pad_id)
            j = i
            while j < len(ids) and ids[j] == image_pad_id:
                j += 1
            i = j
        elif t == video_pad_id:
            new_ids.append(image_pad_id)  # masquerade as image
            j = i
            while j < len(ids) and ids[j] == video_pad_id:
                j += 1
            i = j
        else:
            new_ids.append(t)
            i += 1
    text = tokenizer.decode(new_ids, skip_special_tokens=False)
    return text, new_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(_BASE / "checkpoints_qwen25/nusc_planning_b5pp_1cam_qwen3vl_multimodal/final"))
    ap.add_argument("--n-warmup", type=int, default=3)
    ap.add_argument("--n-runs", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=14)
    ap.add_argument("--out", default=str(_BASE / "deploy/trt_bench/B5pp_trt_qwen3vl_multimodal_bf16.json"))
    ap.add_argument("--gate-only", action="store_true", help="Run parity gate then exit (no bench)")
    args = ap.parse_args()

    # === Imports after shim ===
    from transformers import AutoModelForImageTextToText, AutoProcessor, AutoConfig
    from multimodal_planning_dataset import MultiModalPlanningDataset
    from phase1_5_vision_embeds import assemble_mm_embedding
    from phase2_mrope_config import build_mrope_config
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.disaggregated_params import DisaggregatedParams
    from tensorrt_llm._torch.shared_tensor import SharedTensorContainer
    from tensorrt_llm.llmapi import KvCacheConfig

    print(f"[bench] === STEP 1: load HF model for vision pre-compute ===")
    t0 = time.perf_counter()
    hf_model = AutoModelForImageTextToText.from_pretrained(
        args.ckpt, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda:0").eval()
    proc = AutoProcessor.from_pretrained(args.ckpt)
    tok = proc.tokenizer
    print(f"[bench] HF loaded in {time.perf_counter()-t0:.1f}s")

    image_pad_id = tok.convert_tokens_to_ids("<|image_pad|>")
    video_pad_id = tok.convert_tokens_to_ids("<|video_pad|>")
    vision_start_id = tok.convert_tokens_to_ids("<|vision_start|>")
    vision_end_id = tok.convert_tokens_to_ids("<|vision_end|>")
    print(f"[bench] image_pad={image_pad_id} video_pad={video_pad_id} vision_start={vision_start_id} vision_end={vision_end_id}")

    print(f"[bench] === STEP 2: build val sample[0] ===")
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
    print(f"[bench] prompt_len={prompt_len}, full input_ids len={sample['input_ids'].shape[0]}")

    print(f"[bench] === STEP 3: pre-compute vision embeds (Phase 1.5) ===")
    # B.5'' order: video first, image second
    embed_result = assemble_mm_embedding(
        hf_model=hf_model,
        pixel_values=sample["pixel_values"],
        image_grid_thw=sample["image_grid_thw"],
        pixel_values_videos=sample["pixel_values_videos"],
        video_grid_thw=sample["video_grid_thw"],
        modality_order=("video", "image"),
    )
    mm_embedding_full = embed_result["mm_embedding"].to("cuda:0", dtype=torch.bfloat16)
    n_video = embed_result["video_mm_tokens"]
    n_image = embed_result["image_mm_tokens"]
    print(f"[bench] mm_embedding shape: {tuple(mm_embedding_full.shape)} (video={n_video} + image={n_image})")
    # Split into per-pad-tag handles (video first then image — matches text order)
    video_embed = mm_embedding_full[:n_video].contiguous()
    image_embed = mm_embedding_full[n_video:].contiguous()
    print(f"[bench] video_embed: {tuple(video_embed.shape)}, image_embed: {tuple(image_embed.shape)}")

    print(f"[bench] === STEP 4: compute mrope_config (Phase 2, prompt-only slice) ===")
    mrope_full = build_mrope_config(
        model_config=hf_model.config,
        input_ids=sample["input_ids"],
        mm_token_type_ids=sample["mm_token_type_ids"],
        image_grid_thw=sample["image_grid_thw"].clone(),
        video_grid_thw=sample["video_grid_thw"].clone(),
        attention_mask=sample["attention_mask"],
    )
    # Slice to prompt-only
    mrope_pos_ids = mrope_full["mrope_position_ids"][:, :, :prompt_len].to("cuda:0", dtype=torch.int32).contiguous()
    # Phase 2 deltas: (1,1); Agent B working pattern: (1,) — squeeze last dim
    mrope_deltas = mrope_full["mrope_position_deltas"].view(-1).to("cuda:0", dtype=torch.int32).contiguous()
    print(f"[bench] mrope_position_ids: {tuple(mrope_pos_ids.shape)} {mrope_pos_ids.dtype}")
    print(f"[bench] mrope_position_deltas: {tuple(mrope_deltas.shape)} {mrope_deltas.dtype}")

    print(f"[bench] === STEP 5: build unexpanded prompt text ===")
    text_prompt, unexpanded_ids = build_unexpanded_prompt(
        sample["input_ids"], prompt_len, tok, image_pad_id, video_pad_id, vision_start_id, vision_end_id
    )
    # NOTE: after build_unexpanded_prompt masquerade, ALL pads are image_pad. We
    # still need to know the original video/image splits, so re-count from the
    # original input_ids[:prompt_len] before collapse.
    orig_ids = sample["input_ids"][:prompt_len].tolist()
    n_img_blocks_orig = 0; n_vid_blocks_orig = 0
    j = 0
    while j < len(orig_ids):
        t = orig_ids[j]
        if t == image_pad_id:
            n_img_blocks_orig += 1
            while j < len(orig_ids) and orig_ids[j] == image_pad_id: j += 1
        elif t == video_pad_id:
            n_vid_blocks_orig += 1
            while j < len(orig_ids) and orig_ids[j] == video_pad_id: j += 1
        else:
            j += 1
    n_img_blocks = n_img_blocks_orig
    n_vid_blocks = n_vid_blocks_orig
    n_total_pads_after = sum(1 for t in unexpanded_ids if t == image_pad_id)
    print(f"[bench] unexpanded_ids len={len(unexpanded_ids)}, orig blocks: img={n_img_blocks}, vid={n_vid_blocks}, "
          f"after-masquerade total image_pad blocks={n_total_pads_after}")
    assert n_total_pads_after == n_img_blocks + n_vid_blocks
    print(f"[bench] text preview (300 chars): {text_prompt[:300]!r}")

    # Split video embed into N equal temporal chunks (Qwen3-VL inserts one
    # <|video_pad|> per temporal patch when T > 1).
    assert n_video % n_vid_blocks == 0, f"video tokens {n_video} not divisible by {n_vid_blocks} blocks"
    assert n_image % n_img_blocks == 0, f"image tokens {n_image} not divisible by {n_img_blocks} blocks"
    video_chunk_size = n_video // n_vid_blocks
    image_chunk_size = n_image // n_img_blocks
    video_chunks = [video_embed[i * video_chunk_size:(i + 1) * video_chunk_size].contiguous()
                    for i in range(n_vid_blocks)]
    image_chunks = [image_embed[i * image_chunk_size:(i + 1) * image_chunk_size].contiguous()
                    for i in range(n_img_blocks)]
    print(f"[bench] video chunks: {n_vid_blocks} × ({video_chunk_size}, {video_embed.shape[1]})")
    print(f"[bench] image chunks: {n_img_blocks} × ({image_chunk_size}, {image_embed.shape[1]})")

    # Free HF model memory before loading TRT (both 4B models won't fit)
    print(f"[bench] === STEP 6: free HF, load TRT-LLM ===")
    del hf_model
    torch.cuda.empty_cache()
    import gc; gc.collect()

    cfg = AutoConfig.from_pretrained(args.ckpt, trust_remote_code=True)
    llm = LLM(
        model=args.ckpt,
        max_batch_size=1,
        max_seq_len=2048,
        max_num_tokens=2048,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.5),
        trust_remote_code=True,
    )
    print(f"[bench] TRT engine loaded")

    # Build disagg params: one handle per pad in ORIGINAL text-order.
    # After masquerade, unexpanded_ids only has image_pad — so we must walk
    # the ORIGINAL input_ids to know real video/image alternation.
    print(f"[bench] === STEP 7: construct DisaggregatedParams ===")
    mm_handles_tensors = []
    vi = 0; ii = 0
    k = 0
    orig_ids2 = sample["input_ids"][:prompt_len].tolist()
    while k < len(orig_ids2):
        t = orig_ids2[k]
        if t == video_pad_id:
            mm_handles_tensors.append(video_chunks[vi]); vi += 1
            while k < len(orig_ids2) and orig_ids2[k] == video_pad_id: k += 1
        elif t == image_pad_id:
            mm_handles_tensors.append(image_chunks[ii]); ii += 1
            while k < len(orig_ids2) and orig_ids2[k] == image_pad_id: k += 1
        else:
            k += 1
    mm_handles = [SharedTensorContainer.from_tensor(t).dump_to_dict() for t in mm_handles_tensors]
    print(f"[bench] mm_handles: {len(mm_handles)} (expected {n_vid_blocks + n_img_blocks})")
    mrope_pos_handle = SharedTensorContainer.from_tensor(mrope_pos_ids).dump_to_dict()
    mrope_delta_handle = SharedTensorContainer.from_tensor(mrope_deltas).dump_to_dict()
    disagg = DisaggregatedParams(
        request_type="context_and_generation",
        multimodal_embedding_handles=mm_handles,
        mrope_position_ids_handle=mrope_pos_handle,
        mrope_position_deltas_handle=mrope_delta_handle,
    )

    # === STEP 8: Phase 4 logit-parity gate ===
    print(f"[bench] === STEP 8: parity gate (max_tokens=1, compare top-1 to HF baseline) ===")
    hf_baseline = json.load(open(_HERE / "_phase4_hf_baseline_logits.json"))
    expected_top1 = hf_baseline["top5_token_ids"][0]
    print(f"[bench] HF baseline top-1 token: {expected_top1}")

    sp_gate = SamplingParams(max_tokens=1, temperature=0.0)
    out = llm.generate([{"prompt": text_prompt}], sampling_params=sp_gate, disaggregated_params=disagg)
    actual_top1 = list(out[0].outputs[0].token_ids)[0] if out[0].outputs[0].token_ids else None
    print(f"[bench] TRT top-1 token: {actual_top1}")
    print(f"[bench] prompt_token_ids len: {len(out[0].prompt_token_ids)}  (expected ~{prompt_len})")
    if actual_top1 != expected_top1:
        print(f"[bench] !!! GATE FAIL: TRT top-1 {actual_top1} != HF baseline {expected_top1}")
        print(f"[bench]     Embeds/mrope likely scattered wrong. Aborting bench.")
        return 2
    print(f"[bench] === GATE PASS ===")

    if args.gate_only:
        return 0

    # === STEP 9: bench ===
    print(f"[bench] === STEP 9: bench warmup ({args.n_warmup}) ===")
    sp_full = SamplingParams(max_tokens=args.max_new_tokens, temperature=0.0)
    for _ in range(args.n_warmup):
        _ = llm.generate([{"prompt": text_prompt}], sampling_params=sp_full, disaggregated_params=disagg)

    print(f"[bench] === STEP 10: bench runs ({args.n_runs}) ===")
    sp_one = SamplingParams(max_tokens=1, temperature=0.0)
    ttft, full = [], []
    for _ in range(args.n_runs):
        t = time.perf_counter()
        _ = llm.generate([{"prompt": text_prompt}], sampling_params=sp_one, disaggregated_params=disagg)
        ttft.append(time.perf_counter() - t)
    for _ in range(args.n_runs):
        t = time.perf_counter()
        _ = llm.generate([{"prompt": text_prompt}], sampling_params=sp_full, disaggregated_params=disagg)
        full.append(time.perf_counter() - t)

    decode = [(f - t) / (args.max_new_tokens - 1) for f, t in zip(full, ttft)]
    throughput = [args.max_new_tokens / f for f in full]

    results = {
        "ckpt": args.ckpt,
        "backend": "TRT-LLM 1.3.0rc15 PyTorch backend + mm-disagg (HF vision pre-compute + injected handles)",
        "model": "Qwen3-VL-4B B.5'' VLA",
        "modality_real": "video(1cam x 4f) + image(HD-map BEV) + text(bbox/ego/prompt)",
        "dtype": "bfloat16",
        "video_mm_tokens": n_video,
        "image_mm_tokens": n_image,
        "embed_dim": int(mm_embedding_full.shape[1]),
        "prompt_len_expanded": int(len(out[0].prompt_token_ids)),
        "max_new_tokens": args.max_new_tokens,
        "n_warmup": args.n_warmup,
        "n_runs": args.n_runs,
        "parity_gate": {
            "hf_baseline_top1": expected_top1,
            "trt_top1": actual_top1,
            "passed": True,
        },
        "TTFT_ms": {
            "mean": 1000 * sum(ttft) / len(ttft),
            "p50": 1000 * percentile(ttft, 50),
            "p99": 1000 * percentile(ttft, 99),
        },
        "per_token_decode_ms": {
            "mean": 1000 * sum(decode) / len(decode),
            "p50": 1000 * percentile(decode, 50),
            "p99": 1000 * percentile(decode, 99),
        },
        "full_traj_ms": {
            "mean": 1000 * sum(full) / len(full),
            "p50": 1000 * percentile(full, 50),
            "p99": 1000 * percentile(full, 99),
        },
        "throughput_toks_per_s": {
            "mean": sum(throughput) / len(throughput),
            "p50": percentile(throughput, 50),
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[bench] saved → {args.out}")
    print(f"  TTFT mean: {results['TTFT_ms']['mean']:.1f} ms")
    print(f"  decode mean: {results['per_token_decode_ms']['mean']:.2f} ms/tok")
    print(f"  full mean: {results['full_traj_ms']['mean']:.1f} ms / {args.max_new_tokens} tok")
    print(f"  throughput: {results['throughput_toks_per_s']['mean']:.1f} tok/s")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
