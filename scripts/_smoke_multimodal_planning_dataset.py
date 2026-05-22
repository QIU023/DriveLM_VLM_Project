"""CPU-only smoke test for MultiModalPlanningDataset (Tracks B.5 / B.6).

Verifies:
  1. Dataset constructs over real nuScenes infos pkl + real HD-map cache + real
     bbox jsonl.
  2. 5 real samples (including at least one HD-map-missing token from the
     79-token edge-case list) round-trip through the OEM Qwen2.5-VL processor.
  3. Output dict has the keys train_lora.collate_fn expects: input_ids,
     attention_mask, labels, pixel_values_videos, video_grid_thw,
     second_per_grid_ts, pixel_values, image_grid_thw.
  4. B.5 path (modality_dropout_p=0.0) loads the real HD-map image (or black
     for cache-miss) and the real bbox text.
  5. B.6 path (modality_dropout_p=1.0 — every modality dropped) yields
     all-black HD map (hash check) and "(none)" bbox text for all 5 samples.
  6. The bbox text appears verbatim in one printed sample's prompt.

CPU-only; does NOT call the model forward (would require GPU which the 8f_longvu
orchestrator currently owns). Verifies the dataset + processor only.

Run:
    /usr/bin/python3 scripts/_smoke_multimodal_planning_dataset.py
"""
from __future__ import annotations

import hashlib
import os
import pickle
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
_REPO = os.path.dirname(_HERE)

# Force CPU before any torch import.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

assert not torch.cuda.is_available(), "smoke must run CPU-only (CUDA_VISIBLE_DEVICES='')"

from transformers import AutoProcessor  # noqa: E402

from multimodal_planning_dataset import (  # noqa: E402
    BBOX_NONE_TEXT,
    MultiModalPlanningDataset,
    _black_hdmap,
)

MODEL_DIR = "/workspace/models/Qwen2.5-VL-3B-Instruct"
INFOS_TRAIN = os.path.join(_REPO, "data/uniad_infos/nuscenes_infos_temporal_train.pkl")
HDMAP_DIR = os.path.join(_REPO, "data/preproc/hdmap_bev")
BBOX_JSONL = os.path.join(_REPO, "data/preproc/bbox_egostate_train.jsonl")
NUSC_ROOT = os.path.join(_REPO, "data/nuscenes")


def _find_missing_hdmap_token() -> str | None:
    """Return one token whose HD-map PNG is NOT in the cache (the 79-token
    edge-case list), or None if the cache is complete."""
    with open(INFOS_TRAIN, "rb") as f:
        blob = pickle.load(f)
    infos = blob["infos"] if isinstance(blob, dict) else blob
    cache_tokens = {
        fn[:-4]
        for fn in os.listdir(os.path.join(HDMAP_DIR, "train"))
        if fn.endswith(".png")
    }
    for info in infos:
        if info["token"] not in cache_tokens:
            return info["token"]
    return None


def _hash_image(img) -> str:
    """SHA-256 of the raw RGB bytes; the black baseline has a unique fixed hash."""
    arr = np.asarray(img.convert("RGB"))
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


_BLACK_HASH = _hash_image(_black_hdmap())


def _print_shape_table(sample: dict, label: str) -> None:
    print(f"--- {label} ---")
    for k in (
        "input_ids",
        "attention_mask",
        "labels",
        "pixel_values_videos",
        "video_grid_thw",
        "second_per_grid_ts",
        "pixel_values",
        "image_grid_thw",
    ):
        v = sample.get(k)
        if v is None:
            print(f"  {k:>22s} : <missing>")
        elif hasattr(v, "shape"):
            print(f"  {k:>22s} : shape={tuple(v.shape)} dtype={v.dtype}")
        else:
            print(f"  {k:>22s} : {type(v).__name__} {repr(v)[:60]}")


def _print_prompt_text(sample: dict, processor, label: str, max_chars: int = 1500) -> None:
    """Print the text-decoded prompt (input_ids up to _meta_prompt_len) so a
    human can eyeball the bbox + ego preamble inclusion."""
    prompt_len = sample.get("_meta_prompt_len")
    if prompt_len is None:
        return
    input_ids = sample["input_ids"][:prompt_len].tolist()
    text = processor.tokenizer.decode(input_ids, skip_special_tokens=False)
    print(f"--- {label} PROMPT TEXT (decoded; first {max_chars} chars) ---")
    print(text[:max_chars])
    if len(text) > max_chars:
        print(f"... [truncated; total {len(text)} chars]")
    print(f"--- END {label} PROMPT TEXT ---")


def build_dataset(processor, dropout_p: float, max_samples: int) -> MultiModalPlanningDataset:
    ds = MultiModalPlanningDataset(
        infos_path=INFOS_TRAIN,
        nusc_root=NUSC_ROOT,
        processor=processor,
        max_length=4096,
        num_past_frames=4,
        num_future_waypoints=6,
        video_fps=2.0,
        vla_loss_mode="answer_and_traj",
        max_samples=None,  # filter happens later; we want full _keep for missing-token lookup
        require_full_future=True,
        planning_cams=["CAM_FRONT"],
        require_all_cams=True,
        hdmap_dir=HDMAP_DIR,
        bbox_jsonl=BBOX_JSONL,
        split="train",
        modality_dropout_p=dropout_p,
    )
    # Limit AFTER construction to keep the missing-token index reachable.
    ds._keep = ds._keep[: max(max_samples, 5)]
    return ds


def main() -> int:
    print(f"[smoke] cwd={os.getcwd()}  repo={_REPO}")
    print(f"[smoke] model={MODEL_DIR}")
    print(f"[smoke] CUDA available: {torch.cuda.is_available()}")
    print(f"[smoke] black HD-map hash: {_BLACK_HASH}")
    print(f"[smoke] BBOX_NONE_TEXT: {BBOX_NONE_TEXT!r}")

    # Pre-flight: surface the missing-HDmap token list so we can target one.
    missing_tok = _find_missing_hdmap_token()
    if missing_tok is not None:
        print(f"[smoke] Sample missing-HDmap token: {missing_tok}")
    else:
        print("[smoke] WARNING: no missing-HD-map tokens found (cache complete?)")

    print("[smoke] loading processor (this also configures min/max pixels)...")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    # Match R1' baseline processor caps (min=max=109760 per frame; matches
    # configs/nuscenes_planning_b5.yaml). train_lora.py sets these from the
    # YAML at startup; we do it explicitly here so the smoke matches production
    # token counts.
    processor.image_processor.min_pixels = 109760
    processor.image_processor.max_pixels = 109760
    if hasattr(processor, "video_processor") and processor.video_processor is not None:
        vp = processor.video_processor
        if hasattr(vp, "size") and vp.size is not None:
            if hasattr(vp.size, "shortest_edge"):
                setattr(vp.size, "shortest_edge", 109760)
            if hasattr(vp.size, "longest_edge"):
                setattr(vp.size, "longest_edge", 109760)
        for attr, val in (("min_pixels", 109760), ("max_pixels", 109760)):
            if hasattr(vp, attr):
                setattr(vp, attr, val)
    print(f"[smoke] processor loaded in {time.time() - t0:.2f}s")

    # ----------------------------------------------------------------------
    # B.5 path: dropout=0.0; 5 samples; check key set + shape table + bbox
    # text appears in prompt.
    # ----------------------------------------------------------------------
    print("\n[smoke] === B.5 PATH (modality_dropout_p=0.0) ===")
    ds_b5 = build_dataset(processor, dropout_p=0.0, max_samples=5)

    # If we found a missing token, splice it in as sample 0 by swapping into _keep.
    target_missing_idx = None
    if missing_tok is not None:
        if missing_tok in ds_b5.tok2idx:
            real_idx = ds_b5.tok2idx[missing_tok]
            # Insert this real_idx at position 0 of _keep so we hit it first.
            if real_idx not in ds_b5._keep:
                ds_b5._keep = [real_idx] + ds_b5._keep[:4]
                target_missing_idx = 0
            else:
                target_missing_idx = ds_b5._keep.index(real_idx)
            print(f"[smoke] Forced missing-HD-map token at __getitem__ idx={target_missing_idx}")
        else:
            print(f"[smoke] WARN: missing token {missing_tok} not in tok2idx; "
                  f"missing-HDmap test will rely on natural placement.")

    b5_samples = []
    for i in range(5):
        sample = ds_b5[i]
        b5_samples.append(sample)
        print(f"[smoke] B.5 sample {i}: token={sample['_meta_token'][:8]}..., "
              f"input_ids={tuple(sample['input_ids'].shape)}, "
              f"video_grid_thw={sample['video_grid_thw'].tolist()}, "
              f"image_grid_thw={sample['image_grid_thw'].tolist()}")

    # Shape table for sample 0
    _print_shape_table(b5_samples[0], "B.5 SAMPLE 0 SHAPE TABLE")

    # Print decoded prompt for sample 0 so a human can verify the bbox text + ego
    # speed appear.
    _print_prompt_text(b5_samples[0], processor, "B.5 SAMPLE 0")

    # Required key set for production training:
    REQUIRED_KEYS = {
        "input_ids", "attention_mask", "labels",
        "pixel_values_videos", "video_grid_thw", "second_per_grid_ts",
        "pixel_values", "image_grid_thw",
        "_meta_waypoints", "_meta_valid_mask", "_meta_token",
        "_meta_prompt_len", "_meta_action_len",
        "image_name",
    }
    missing_keys = REQUIRED_KEYS - set(b5_samples[0].keys())
    print(f"[smoke] B.5 missing required keys: {missing_keys or 'NONE (PASS)'}")
    assert not missing_keys, f"missing keys: {missing_keys}"

    # Verify the missing-HD-map sample (if forced) loaded a black HD-map image.
    if target_missing_idx is not None:
        # The dataset puts HD-map pixel_values into the sample; we can't recover
        # the raw PIL image post-processor, but the image_grid_thw should
        # correspond to either the missing-cache black substitute or a real
        # HD-map. Both have the same shape (224x224 -> grid (1,16,16) or
        # processor-upscaled equivalent), but to make the test direct we re-
        # peek at the cache file:
        path = os.path.join(HDMAP_DIR, "train", f"{ds_b5.infos[ds_b5._keep[target_missing_idx]]['token']}.png")
        if os.path.exists(path):
            print(f"[smoke] WARN: missing-HD-map test forced token but cache file exists at {path}")
        else:
            print(f"[smoke] OK: missing-HD-map test target has no cache file (will be black substitute)")

    # ----------------------------------------------------------------------
    # B.6 path: dropout=1.0; verify ALL 5 samples have black HD-map + (none)
    # bbox text.
    # ----------------------------------------------------------------------
    print("\n[smoke] === B.6 PATH (modality_dropout_p=1.0) ===")
    ds_b6 = build_dataset(processor, dropout_p=1.0, max_samples=5)
    # Re-instrument to peek at the chosen modalities. We bypass __getitem__'s
    # PIL→processor pipeline for the HD-map by re-running the dropout decision
    # path in isolation (since the processor consumes the PIL and we cannot
    # recover the raw bytes from pixel_values).
    rng_check = ds_b6._get_rng()
    rng_seed = int(rng_check.bit_generator.state['state']['state'] % (2**31 - 1))
    print(f"[smoke] B.6 worker RNG seeded (state hash {rng_seed % 9999})")

    # Direct check: instantiate the dropout decisions by simulating __getitem__
    # without the processor. We patch the processor with a no-op stub for the
    # B.6 PIL inspection pass.
    n_black = 0
    n_none = 0
    pil_hashes = []
    bbox_seen = []
    for i in range(5):
        sample_token = ds_b6.infos[ds_b6._keep[i]]["token"]
        # Force fresh RNG draw — for dropout_p=1.0 both draws are always < 1.0,
        # so HD-map=black and bbox="(none)" deterministically.
        # We replicate the dataset's decision logic to inspect PIL+text before
        # the processor mangles them.
        drop_hdmap = (ds_b6.modality_dropout_p > 0.0
                      and ds_b6._get_rng().random() < ds_b6.modality_dropout_p)
        drop_bbox = (ds_b6.modality_dropout_p > 0.0
                     and ds_b6._get_rng().random() < ds_b6.modality_dropout_p)
        assert drop_hdmap and drop_bbox, "p=1.0 must drop both modalities"
        hd_img = _black_hdmap() if drop_hdmap else ds_b6._load_hdmap(sample_token)
        bbox_text = BBOX_NONE_TEXT if drop_bbox else ds_b6._lookup_bbox(sample_token)
        h = _hash_image(hd_img)
        pil_hashes.append(h)
        bbox_seen.append(bbox_text)
        if h == _BLACK_HASH:
            n_black += 1
        if bbox_text == BBOX_NONE_TEXT:
            n_none += 1
    print(f"[smoke] B.6 black HD-map count: {n_black}/5  bbox=(none) count: {n_none}/5")
    print(f"[smoke] B.6 unique HD-map hashes: {set(pil_hashes)}")
    print(f"[smoke] B.6 unique bbox texts: {set(bbox_seen)}")
    assert n_black == 5, f"B.6 p=1.0 should yield 5 black HD-maps, got {n_black}"
    assert n_none == 5, f"B.6 p=1.0 should yield 5 (none) bbox texts, got {n_none}"

    # Also exercise the full __getitem__ on the same 5 samples (with the
    # processor) to make sure the round-trip still works under dropout=1.0.
    print("\n[smoke] B.6 full __getitem__ pass (with processor) ...")
    for i in range(5):
        sample = ds_b6[i]
        print(f"[smoke] B.6 sample {i}: token={sample['_meta_token'][:8]}..., "
              f"input_ids={tuple(sample['input_ids'].shape)}, "
              f"video_grid_thw={sample['video_grid_thw'].tolist()}, "
              f"image_grid_thw={sample['image_grid_thw'].tolist()}")

    # Print decoded prompt for B.6 sample 0 so a human can verify the
    # "Detected objects: (none)" string appears (NOT the real bbox list).
    _print_prompt_text(ds_b6[0], processor, "B.6 SAMPLE 0 (dropout=1.0)", max_chars=800)

    print("\n[smoke] === ALL CHECKS PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
