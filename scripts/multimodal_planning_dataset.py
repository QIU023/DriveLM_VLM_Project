"""Image-as-modality multi-modal nuScenes planning dataset (Tracks B.5 / B.6).

Extends ``PlanningDataset`` with two cached pre-rendered modalities consumed
through the OEM Qwen2.5-VL processor as ordinary multi-modal tokens (NO new
learnable projector — the native vision tower handles both the camera video
clip and the HD-map BEV image):

  1. HD map BEV PNG @ 224x224 RGB   ->  ``data/preproc/hdmap_bev/{split}/{token}.png``
  2. 3D bbox text (top-K objects)   ->  ``data/preproc/bbox_egostate_{split}.jsonl``

Per-sample chat-template user content (interleaved, in this exact order so the
processor's ``videos=[...]`` and ``images=[...]`` arguments line up positionally
with the corresponding ``<|vision_start|>...<|vision_end|>`` blocks emitted by
``apply_chat_template``):

    {"type": "video", camera clip}            # 4 past CAM_FRONT frames @ 2 Hz
    {"type": "image", HD-map BEV}             # 224x224 RGB, ego-up
    {"type": "text",  "Detected objects in ego frame:\\n..." + "\\n\\n" + planning prompt}

Ego state (speed scalar) is still injected via ``_format_ego_speed_preamble``,
which is part of the trailing text block (NO change to that mechanism).

NO LiDAR (cache not yet built; deferred to a later track).

Track B.5 :  ``modality_dropout_p = 0.0`` (always feed HD map + bbox)
Track B.6 :  ``modality_dropout_p = 0.1`` (XPeng-style per-modality robustness:
            independently zero out HD map -> all-black, bbox -> "(none)";
            CAMERA + ego state are NEVER dropped — robustness pattern from
            the production-deployment story, not the training story.)

Missing-HD-map policy (79 train + 7 val tokens that prep_hdmap_bev.py failed on
due to MultiLineString edge cases in the nuScenes map JSON): SUBSTITUTE a
black 224x224 RGB image. We do NOT skip the sample (a) because Dataset.__getitem__
returning IndexError mid-epoch would force a custom DataLoader iter and (b) all
79 / 7 are well-formed planning samples on every other axis (full future, valid
camera, valid bbox) — losing them silently is wasteful when a 1-line black-image
substitution preserves the loss signal everywhere except the (unused) HD-map
pixels for that ~0.3 % of samples. This matches the dropout-substitution shape
exactly (the model already sees black HD maps under B.6 dropout) so it does NOT
create a distributional surprise.

CPU-safe; no GPU side effects in __init__ or __getitem__. The image processor
is a numpy / PIL operation throughout — only the model forward needs GPU.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image

try:
    from planning_dataset import (  # type: ignore
        PlanningDataset,
        _format_ego_speed_preamble,
        _multicam_prompt_suffix,
        CAM_LABELS_3,
        NUM_PAST_FRAMES,
        NUM_FUTURE_WP,
        DEFAULT_PLANNING_CAMS,
    )
except ImportError:  # pragma: no cover
    from .planning_dataset import (  # type: ignore
        PlanningDataset,
        _format_ego_speed_preamble,
        _multicam_prompt_suffix,
        CAM_LABELS_3,
        NUM_PAST_FRAMES,
        NUM_FUTURE_WP,
        DEFAULT_PLANNING_CAMS,
    )

# Constants —————————————————————————————————————————————————————————
HDMAP_SIDE_PX = 224  # matches prep_hdmap_bev.py default; processor will accept any size,
                     # but the cache is fixed at 224 so a black substitute must match.
BBOX_NONE_TEXT = "Detected objects: (none)\n"


def _black_hdmap() -> Image.Image:
    """Return a fresh black 224x224 RGB PIL Image (used for both the missing-cache
    fallback and the B.6 modality-dropout HD-map drop). Token count after the
    Qwen2.5-VL image processor is identical to a real HD-map BEV (both are 224x224)
    so prompt length is stable across dropout draws."""
    return Image.new("RGB", (HDMAP_SIDE_PX, HDMAP_SIDE_PX), color=(0, 0, 0))


def _build_user_content_multimodal(
    info: dict,
    cams: List[str],
    bbox_text: str,
) -> list:
    """Return the apply_chat_template content list for a multi-modal sample.

    Layout (single CAM_FRONT only — multi-cam combinations are NOT supported in
    B.5/B.6 since the orchestrator's A.0 R1' baseline is 1-cam × 4f; if a future
    track needs multi-cam + HD-map we extend this):

        Single-cam:
            [{"type": "video"},                # camera clip
             {"type": "image"},                # HD-map BEV
             {"type": "text", <bbox + ego + prompt>}]

        Multi-cam (kept for forward-compat with the existing cam-loop pattern in
        planning_dataset._build_user_content_multicam):
            [<per-cam label + video> ...,
             {"type": "image"},                # HD-map BEV (after all cams)
             {"type": "text", <bbox + ego + prompt>}]

    The HD-map image block is intentionally placed AFTER the camera video so that
    a single-cam config matches the channel order
    ``videos=[camera_clip], images=[hdmap]`` passed to the processor. The
    processor consumes ``<|vision_start|><|video_pad|><|vision_end|>`` blocks
    from ``videos[]`` in left-to-right order, and ``<|image_pad|>`` blocks from
    ``images[]`` in left-to-right order — they are independent queues. As long
    as we pass exactly 1 image and 1 video in the order matching the content
    list above the processor produces the expected pixel_values / video_grid_thw
    / image_grid_thw triples.
    """
    # Build the trailing text block: ego speed (already baked in via
    # _format_ego_speed_preamble) + bbox text + planning instruction.
    ego_preamble = _format_ego_speed_preamble(info)
    prompt_suffix = _multicam_prompt_suffix(cams)
    # bbox_text already starts with its own header
    # ("Detected objects in ego frame:\n...") so we do NOT prepend another
    # "Detected objects:" — that would be a redundant double-header. The dropout
    # path injects BBOX_NONE_TEXT which DOES have its own "Detected objects:"
    # header for parity. Either way, the LM sees exactly one bbox header.
    trailing_text = (
        bbox_text.rstrip("\n")
        + "\n\n"
        + ego_preamble
        + prompt_suffix
    )

    if len(cams) == 1 and cams[0] == "CAM_FRONT":
        return [
            {"type": "video"},
            {"type": "image"},
            {"type": "text", "text": trailing_text},
        ]

    # Multi-cam fallback: interleave per-cam labels + video clips, then the HD
    # map, then the trailing text.
    content: list = []
    for cam in cams:
        label = CAM_LABELS_3.get(cam, cam)
        content.append({"type": "text", "text": f"{label}: "})
        content.append({"type": "video"})
        content.append({"type": "text", "text": " "})
    content.append({"type": "image"})
    content.append({"type": "text", "text": trailing_text})
    return content


class MultiModalPlanningDataset(PlanningDataset):
    """``PlanningDataset`` + cached HD-map BEV PNG + cached 3D bbox text, fed
    through the OEM Qwen2.5-VL processor as ordinary image+video tokens.

    No new learnable module — the native vision tower handles both modalities.

    Constructor delta vs PlanningDataset:
        hdmap_dir          : directory holding ``{train,val}/{sample_token}.png``
                             (224x224 RGB BEV rendered by prep_hdmap_bev.py).
        bbox_jsonl         : path to ``bbox_egostate_{split}.jsonl`` for this
                             split (one row per sample_token with bbox_text).
        split              : "train" | "val"; used to pick the hdmap_dir subdir.
        modality_dropout_p : float in [0,1]. Per-sample, INDEPENDENTLY drop
                             HD map -> all-black, bbox -> "(none)". Camera and
                             ego speed are NEVER dropped.

    Returns the same key set as PlanningDataset plus ``pixel_values`` and
    ``image_grid_thw`` (the Qwen2.5-VL processor emits these for the HD-map
    image input). train_lora.collate_fn already supports these keys (see
    train_lora.py:282 collate_fn).
    """

    def __init__(
        self,
        *args,
        hdmap_dir: str,
        bbox_jsonl: str,
        split: str = "train",
        modality_dropout_p: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        self.split = split
        # Resolve hdmap_dir/{split}
        if not os.path.isabs(hdmap_dir):
            raise ValueError(
                f"hdmap_dir must be absolute by the time it reaches the dataset; got {hdmap_dir!r}"
            )
        self.hdmap_split_dir = os.path.join(hdmap_dir, split)
        if not os.path.isdir(self.hdmap_split_dir):
            raise FileNotFoundError(
                f"HD-map split dir not found: {self.hdmap_split_dir}"
            )
        if not os.path.isfile(bbox_jsonl):
            raise FileNotFoundError(f"bbox jsonl not found: {bbox_jsonl}")

        # Load bbox jsonl into a {token: bbox_text} dict (lazy, but on first
        # __init__ since the file is small — ~28 K rows, ~50 MB).
        self._bbox: Dict[str, str] = {}
        with open(bbox_jsonl, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                tok = row.get("sample_token")
                if tok is None:
                    continue
                # bbox_text starts with "Detected objects in ego frame:\n..."
                # (or is the empty / placeholder string for no-detection frames).
                self._bbox[tok] = row.get("bbox_text", "")

        self.modality_dropout_p = float(modality_dropout_p)
        if not (0.0 <= self.modality_dropout_p <= 1.0):
            raise ValueError(
                f"modality_dropout_p must be in [0, 1]; got {self.modality_dropout_p}"
            )

        # Per-worker RNG (each DataLoader worker calls __getitem__ in its own
        # process; numpy / random state is per-process by default — explicit
        # default_rng is reproducible across worker counts at the same seed).
        # We use numpy's per-thread default_rng so that:
        #   * each worker process sees its own RNG sequence
        #   * dropout draws are not correlated across __getitem__ calls
        # The RNG is constructed lazily in __getitem__ to handle PyTorch DataLoader
        # worker forks correctly.
        self._rng: Optional[np.random.Generator] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_rng(self) -> np.random.Generator:
        if self._rng is None:
            # Seed from torch's per-worker seed if available (DataLoader sets
            # this), else from /dev/urandom.
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                seed = int(worker_info.seed % (2**31 - 1))
            else:
                seed = int(np.random.SeedSequence().entropy % (2**31 - 1))
            self._rng = np.random.default_rng(seed)
        return self._rng

    def _load_hdmap(self, sample_token: str) -> Image.Image:
        """Load HD-map BEV PNG for sample_token; if cache miss (79 train / 7
        val) substitute a black 224x224 image. See module docstring for
        rationale."""
        path = os.path.join(self.hdmap_split_dir, f"{sample_token}.png")
        if not os.path.isfile(path):
            return _black_hdmap()
        return Image.open(path).convert("RGB")

    def _lookup_bbox(self, sample_token: str) -> str:
        """Return the bbox text for ``sample_token``. Trusts that the bbox
        jsonl covers every infos entry (verified: bbox train rows == infos
        train tokens == 28130; bbox val rows == infos val tokens == 6019).
        KeyError here is a real bug (missing preprocessing), NOT a runtime
        recovery scenario, so we let it propagate."""
        return self._bbox[sample_token]

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        # Mirror PlanningDataset.__getitem__ down to the processor call, then
        # rebuild the processor input with HD-map + bbox injected.
        base_idx = self._keep[i]
        info = self.infos[base_idx]
        sample_token = info["token"]

        # 1. Past frames + clips (same as PlanningDataset).
        hist = self._walk_history(base_idx)
        if len(self.planning_cams) == 1:
            clips: List[List[Image.Image]] = [self._load_frames(hist, self.planning_cams[0])]
        else:
            clips = self._load_frames_multicam(hist)

        # 2. Future waypoints + tokenized trajectory (same as PlanningDataset).
        wp, valid_mask = self._compute_waypoints(base_idx)
        action_tokens = self.tok.encode(wp, with_boundaries=True)

        # 3. HD map + bbox. Apply per-modality dropout (camera + ego state never
        # dropped — XPeng deployment robustness pattern: the camera is the
        # always-present runtime modality, so the model must never learn to
        # ignore it; HD map + bbox come from on-board fusion modules that can
        # fail in production, so the model should gracefully degrade).
        rng = self._get_rng()
        drop_hdmap = (self.modality_dropout_p > 0.0
                      and rng.random() < self.modality_dropout_p)
        drop_bbox = (self.modality_dropout_p > 0.0
                     and rng.random() < self.modality_dropout_p)

        if drop_hdmap:
            hdmap_img = _black_hdmap()
        else:
            hdmap_img = self._load_hdmap(sample_token)

        if drop_bbox:
            bbox_text = BBOX_NONE_TEXT
        else:
            bbox_text = self._lookup_bbox(sample_token)
            # Some early-keyframe samples have an empty detection list; the
            # preprocessing emits a short placeholder. Treat empty-string as
            # "no detections" with the same canonical wording the dropout path
            # uses (keeps the prompt vocabulary consistent).
            if not bbox_text.strip():
                bbox_text = BBOX_NONE_TEXT

        # 4. Build the chat-template user content with the HD-map slot.
        user_content = _build_user_content_multimodal(info, self.planning_cams, bbox_text)
        proc_messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": "Predicted trajectory:"},
        ]
        text = self.processor.apply_chat_template(
            proc_messages, tokenize=False, add_generation_prompt=False
        )

        # 5. Run processor with both videos= and images=. Order:
        #     videos = [camera_clip(s) in self.planning_cams order]
        #     images = [hdmap_img]   # single HD map
        from transformers.video_utils import VideoMetadata  # local import
        metadata = [
            VideoMetadata(
                total_num_frames=len(clip),
                fps=self.video_fps,
                frames_indices=list(range(len(clip))),
                height=clip[0].height,
                width=clip[0].width,
            )
            for clip in clips
        ]
        inputs = self.processor(
            text=[text],
            videos=clips,
            video_metadata=metadata,
            images=[hdmap_img],
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        pixel_values_videos = inputs.get("pixel_values_videos")
        video_grid_thw = inputs.get("video_grid_thw")
        second_per_grid_ts = inputs.get("second_per_grid_ts")
        pixel_values = inputs.get("pixel_values")          # HD-map image
        image_grid_thw = inputs.get("image_grid_thw")      # HD-map image grid
        # Qwen3-VL M-RoPE requires `mm_token_type_ids` (0=text, 1=image, 2=video)
        # per-token. Processor returns it for Qwen3-family; Qwen2.5-VL returns None.
        mm_token_type_ids = inputs.get("mm_token_type_ids")
        if mm_token_type_ids is not None:
            mm_token_type_ids = mm_token_type_ids.squeeze(0)

        # 6. Append action tokens INSIDE the assistant turn (verbatim copy of
        # the PlanningDataset logic; we cannot call super().__getitem__ because
        # the processor call has to be re-done with the image arg).
        im_end_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        ids_list = input_ids.tolist()
        insert_at = None
        for k in range(len(ids_list) - 1, -1, -1):
            if ids_list[k] == im_end_id:
                insert_at = k
                break
        if insert_at is None:
            insert_at = len(ids_list)
        action_insert_start = insert_at
        new_ids = ids_list[:insert_at] + list(action_tokens) + ids_list[insert_at:]
        input_ids = torch.tensor(new_ids, dtype=input_ids.dtype)
        attention_mask = torch.ones_like(input_ids)
        # Keep mm_token_type_ids aligned with input_ids: action tokens are text
        # (type=0), inserted at the same offset.
        if mm_token_type_ids is not None:
            mm_dtype = mm_token_type_ids.dtype
            mm_token_type_ids = torch.cat([
                mm_token_type_ids[:insert_at],
                torch.zeros(len(action_tokens), dtype=mm_dtype),
                mm_token_type_ids[insert_at:],
            ], dim=0)

        # 7. Truncate if too long (parent's logic).
        if input_ids.shape[0] > self.max_length:
            overflow = input_ids.shape[0] - self.max_length
            trim_from = max(1, action_insert_start - len(action_tokens) - overflow)
            trim_to = trim_from + overflow
            keep = torch.cat([input_ids[:trim_from], input_ids[trim_to:]], dim=0)
            input_ids = keep
            attention_mask = torch.ones_like(input_ids)
            action_insert_start -= overflow
            if mm_token_type_ids is not None:
                mm_token_type_ids = torch.cat([
                    mm_token_type_ids[:trim_from],
                    mm_token_type_ids[trim_to:],
                ], dim=0)

        # 8. Label masking (parent's logic).
        labels = input_ids.clone()
        assistant_token_str = "<|im_start|>assistant\n"
        assistant_tokens = self.processor.tokenizer.encode(
            assistant_token_str, add_special_tokens=False
        )
        input_list = input_ids.tolist()
        assistant_start = -1
        for k in range(len(input_list) - len(assistant_tokens) + 1):
            if input_list[k : k + len(assistant_tokens)] == assistant_tokens:
                assistant_start = k + len(assistant_tokens)
                break
        if assistant_start > 0:
            labels[:assistant_start] = -100
        labels[attention_mask == 0] = -100

        if self.vla_loss_mode == "traj_only":
            labels[:action_insert_start] = -100

        # 9. Mask invalid future waypoints (parent's logic).
        if (valid_mask < 1.0).any() and action_insert_start is not None:
            for t in range(self.num_future):
                if valid_mask[t] == 0.0:
                    bin_x_pos = action_insert_start + 1 + 2 * t
                    bin_y_pos = bin_x_pos + 1
                    if bin_y_pos < labels.shape[0]:
                        labels[bin_x_pos] = -100
                        labels[bin_y_pos] = -100

        # 10. Pack result dict (superset of PlanningDataset's keys).
        result: Dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        if pixel_values_videos is not None:
            result["pixel_values_videos"] = (
                pixel_values_videos.squeeze(0)
                if pixel_values_videos.dim() > 2
                else pixel_values_videos
            )
        if video_grid_thw is not None:
            result["video_grid_thw"] = (
                video_grid_thw.squeeze(0) if video_grid_thw.dim() > 1 else video_grid_thw
            )
        if second_per_grid_ts is not None:
            if not isinstance(second_per_grid_ts, torch.Tensor):
                second_per_grid_ts = torch.tensor(second_per_grid_ts, dtype=torch.float32)
            result["second_per_grid_ts"] = second_per_grid_ts

        # HD-map image keys — train_lora.collate_fn already concatenates these
        # (see train_lora.py:310-319). pixel_values is (N_patches, hidden_dim);
        # image_grid_thw is (1, 3). Keep as-is.
        if pixel_values is not None:
            result["pixel_values"] = (
                pixel_values.squeeze(0) if pixel_values.dim() > 3 else pixel_values
            )
        if image_grid_thw is not None:
            result["image_grid_thw"] = (
                image_grid_thw.squeeze(0) if image_grid_thw.dim() > 1 else image_grid_thw
            )
        if mm_token_type_ids is not None:
            result["mm_token_type_ids"] = mm_token_type_ids

        result["image_name"] = os.path.basename(self._image_path(info))
        result["_meta_waypoints"] = torch.tensor(wp, dtype=torch.float32)
        result["_meta_valid_mask"] = torch.tensor(valid_mask, dtype=torch.float32)
        result["_meta_token"] = sample_token
        result["_meta_prompt_len"] = int(action_insert_start)
        result["_meta_action_len"] = int(len(action_tokens))
        return result


# ============================================================================
# Factory used by train_lora.py
# ============================================================================

def build_multimodal_planning_dataset(cfg: dict, processor, split: str = "train"):
    """Convenience factory called from train_lora.main when
    cfg.dataset_kind == 'nuscenes_planning_multimodal'.

    Inherits all PlanningDataset config keys; adds:
        hdmap_dir          : absolute or repo-relative path to data/preproc/hdmap_bev
        bbox_jsonl         : template with "{split}" placeholder, or an explicit
                             split-specific path. The factory swaps {split} for
                             the current split.
        modality_dropout_p : 0.0 (B.5) or 0.1 (B.6)
    """
    infos_key = "infos_train" if split == "train" else "infos_val"
    infos_path = cfg.get(infos_key)
    if not infos_path:
        raise KeyError(f"Missing config key {infos_key!r}")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isabs(infos_path):
        infos_path = os.path.join(base_dir, infos_path)
    nusc_root = cfg.get("nusc_root", os.path.join(base_dir, "data", "nuscenes"))
    if not os.path.isabs(nusc_root):
        nusc_root = os.path.join(base_dir, nusc_root)

    hdmap_dir = cfg.get("hdmap_dir", "data/preproc/hdmap_bev")
    if not os.path.isabs(hdmap_dir):
        hdmap_dir = os.path.join(base_dir, hdmap_dir)

    bbox_template = cfg.get("bbox_jsonl", "data/preproc/bbox_egostate_{split}.jsonl")
    bbox_path = bbox_template.replace("{split}", split)
    if not os.path.isabs(bbox_path):
        bbox_path = os.path.join(base_dir, bbox_path)

    max_samples = cfg.get(f"{split}_max_samples", None)
    planning_cams_cfg = cfg.get("planning_cams", None)
    if isinstance(planning_cams_cfg, str):
        planning_cams_cfg = [planning_cams_cfg]

    return MultiModalPlanningDataset(
        infos_path=infos_path,
        nusc_root=nusc_root,
        processor=processor,
        max_length=int(cfg.get("max_length", 2560)),
        num_past_frames=int(cfg.get("planning_num_past_frames", NUM_PAST_FRAMES)),
        num_future_waypoints=int(cfg.get("planning_num_future_wp", NUM_FUTURE_WP)),
        video_fps=float(cfg.get("video_fps", 2.0)),
        vla_loss_mode=str(cfg.get("vla_loss_mode", "answer_and_traj")),
        max_samples=max_samples,
        require_full_future=bool(cfg.get("planning_require_full_future", True)),
        planning_cams=planning_cams_cfg,
        require_all_cams=bool(cfg.get("planning_require_all_cams", True)),
        # MultiModal-only kwargs
        hdmap_dir=hdmap_dir,
        bbox_jsonl=bbox_path,
        split=split,
        modality_dropout_p=float(cfg.get("modality_dropout_p", 0.0)),
    )
