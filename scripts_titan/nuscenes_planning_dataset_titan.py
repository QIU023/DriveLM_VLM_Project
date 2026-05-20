# Agent A — torchtitan-friendly wrapper around the HF PlanningDataset.
#
# torchtitan's MMDataLoader consumes an IterableDataset that yields per-
# sample dicts with the keys:
#
#   {
#       "input_ids":          LongTensor(seq_len,)
#       "labels":             LongTensor(seq_len,)
#       "positions":          LongTensor(seq_len,)   -- per-token positions
#       "pixel_values":       list[Tensor(T,H,W,C)] (optional, omit if none)
#       "pixel_values_videos": list[Tensor(T,H,W,C)] (optional, omit if none)
#   }
#
# The MultiModalCollator (mm_collator.py) then:
#   * runs `vision_to_patches()` on each (T,H,W,C) tensor to produce
#     padded patches and per-item `grid_thw`,
#   * builds the dict the model's forward() consumes, namely
#     {"input", "positions", "pixel_values", "grid_thw",
#      "pixel_values_videos", "grid_thw_videos", "special_tokens"}.
#
# The existing HF PlanningDataset (`scripts/planning_dataset.py`) runs the
# Qwen3-VL HF processor to produce ALREADY-PATCHIFIED video tensors
# alongside `video_grid_thw`.  torchtitan wants the pre-patchify
# (T, H_pixels, W_pixels, C) form so its own collator can patch.  We
# therefore re-load the raw frames here and skip the processor's vision
# pipeline, while reusing the geometry / waypoint / token logic from the
# HF dataset.
#
# Pivot history (2026-05-20):
#   This dataset previously consumed a Qwen2.5-VL HF processor and used
#   OpenAI-CLIP image normalization stats.  We pivoted to Qwen3-VL-8B
#   native (torchtitan upstream); the dataset is now driven by the
#   Qwen/Qwen3-VL-8B-Instruct AutoProcessor and uses (0.5, 0.5, 0.5) /
#   (0.5, 0.5, 0.5) image stats to match Qwen3-VL's vision tower.
#   Special tokens (<|vision_start|>, <|video_pad|>, <|vision_end|>,
#   <|im_start|>, <|im_end|>) carry over unchanged because Qwen3-VL
#   inherits the Qwen3 tokenizer surface.

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import IterableDataset

# Reuse the existing HF PlanningDataset for waypoint computation and
# multi-cam prompt construction.  Import sits behind a path mutation so
# the file works whether the caller adds `scripts/` to sys.path or not.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS_DIR = os.path.join(_REPO_ROOT, "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

# pyrefly: ignore [import-error]  -- resolved at runtime via sys.path.
from planning_dataset import (  # type: ignore  # noqa: E402
    DEFAULT_PLANNING_CAMS,
    NUM_FUTURE_WP,
    NUM_PAST_FRAMES,
    PlanningDataset,
    _build_user_content_multicam,
)
from trajectory_tokenizer import (  # type: ignore  # noqa: E402
    TrajectoryTokenizer,
    TrajectoryTokenizerConfig,
)

# torchtitan's shared video preprocessing utility — uses smart_resize to
# round (H, W) to multiples of patch_size * merge_size (32 for Qwen3-VL)
# while keeping the pixel area within [min_pixels, max_pixels].  This is
# what the stock cc12m/obelics LLaVA-style datasets call, so we share it
# here instead of rolling our own.  Without this resize, nuScenes images
# (900x1600) fail `vision_to_patches` because 900 % 32 != 0.
from torchtitan.hf_datasets.multimodal.utils.video import (  # noqa: E402
    process_video,
)
from torchtitan.hf_datasets.multimodal.utils.image import (  # noqa: E402
    smart_resize as _titan_smart_resize,
)


# Default image normalisation for the torchtitan vision pipeline.
# Qwen3-VL switched from OpenAI-CLIP mean/std (Qwen2.5-VL) to (0.5,0.5,0.5)
# / (0.5,0.5,0.5).  This matches torchtitan's stock
# qwen3_vl.config_registry._qwen3_vl_dataloader and the upstream Qwen3-VL
# HF processor.
QWEN3_VL_IMAGE_MEAN = (0.5, 0.5, 0.5)
QWEN3_VL_IMAGE_STD = (0.5, 0.5, 0.5)

# Vision-token geometry.  Must agree with the dataloader/collator config
# (see scripts_titan/train_titan_qwen3_vl.py:_qwen3_vl_dataloader and
# torchtitan/models/qwen3_vl/config_registry._qwen3_vl_dataloader).
QWEN3_VL_PATCH_SIZE = 16
QWEN3_VL_SPATIAL_MERGE_SIZE = 2
QWEN3_VL_TEMPORAL_PATCH_SIZE = 2

# Per-frame pixel budget for ``process_video`` (image-mode smart_resize).
# We pick 294,912 px (≈384x704 at 9:16 aspect) so that 3-cam * 4-frame
# samples emit ~528 visual tokens per cam after the 2x2 spatial merger
# (2 temporal patches * 12 * 22), giving ~1584 visual tokens total per
# sample -- comfortably within the 8192 seq_len cap of the 3-cam config.
#
# The HF Qwen3VLVideoProcessor uses a *volumetric* budget (size.longest_
# _edge), checking ``t_bar * h_bar * w_bar <= max_pixels`` where t_bar =
# ceil(num_frames / temporal_factor) * temporal_factor = num_frames here
# (4 past frames, temporal_factor=2 -> t_bar=4).  We therefore set the HF
# processor's budget to ``QWEN3_VL_IMAGE_MAX_PIXELS * num_frames`` so the
# resize geometry agrees with ours frame-for-frame.  Without this, HF's
# smart_resize lands at 896x1600 (default budget 25M) while ours lands
# at a smaller resolution -- causing the <|video_pad|> placeholder count
# in input_ids to disagree with the patch count emitted by
# MultiModalCollator (observed in smoke v5 as "Number of vision placeholder
# tokens (8101) does not match number of vision tokens (8400)").
QWEN3_VL_IMAGE_MIN_PIXELS = 16 * 16  # 256 px (one merged token)
QWEN3_VL_IMAGE_MAX_PIXELS = 294_912  # 4 * (32 * 32) * 72 = 384x704-ish
# HF video-processor shortest/longest edge (volumetric in (t, h, w))
QWEN3_VL_VIDEO_MIN_PIXELS = 4096        # HF lower bound; never binding here
# QWEN3_VL_VIDEO_MAX_PIXELS is computed per-sample as
# ``QWEN3_VL_IMAGE_MAX_PIXELS * num_frames`` (see _build_sample).


def _pil_frames_to_uint8_thwc(frames: list[Image.Image]) -> torch.Tensor:
    """Convert a list of PIL frames to a (T, H, W, C) uint8 tensor.

    Frames are aligned to the first frame's (H, W) via PIL bilinear
    resize if any mismatch occurs.  The output is uint8 (not normalized
    or rescaled) because `process_video` expects raw uint8 input and
    will do smart_resize + dtype scaling + normalize itself.
    """
    if not frames:
        raise ValueError("empty frames list")
    h0, w0 = frames[0].height, frames[0].width
    arrs = []
    for f in frames:
        if f.mode != "RGB":
            f = f.convert("RGB")
        if (f.height, f.width) != (h0, w0):
            f = f.resize((w0, h0), Image.BILINEAR)
        arrs.append(np.asarray(f, dtype=np.uint8))  # (H, W, C)
    stacked = np.stack(arrs, axis=0)  # (T, H, W, C)
    return torch.from_numpy(stacked)  # uint8


class NuScenesPlanningDatasetTitan(IterableDataset):
    """torchtitan IterableDataset wrapping the HF PlanningDataset.

    Reuses ``PlanningDataset._walk_history``, ``_load_frames``,
    ``_compute_waypoints``, and ``_build_user_content_multicam`` so the
    per-sample text + waypoint + token logic matches the HF training path
    EXACTLY.  The only difference is that we emit raw (T, H, W, C)
    normalised video tensors plus token IDs, instead of running the
    Qwen3-VL HF processor's vision pipeline (torchtitan's
    MultiModalCollator does the patching).

    The HF PlanningDataset takes a HF `processor` and uses it to (a) build
    the chat template and (b) tokenize the prompt.  We KEEP step (b) for
    consistency with the existing trained model; we IGNORE step (a)'s
    image-processing side-effects (we re-load raw frames ourselves).

    Streaming semantics: this dataset is a thin wrapper around an underlying
    map-style PlanningDataset, so we implement standard sharded round-robin
    iteration for DP — each rank visits its assigned indices in order.  No
    sample-packing (torchtitan packing assumes pure-text or small images;
    nuScenes 3-cam videos are far too large to pack into 2K seqs).
    """

    def __init__(
        self,
        infos_path: str,
        nusc_root: str,
        processor,
        *,
        max_length: int = 2560,
        num_past_frames: int = NUM_PAST_FRAMES,
        num_future_waypoints: int = NUM_FUTURE_WP,
        video_fps: float = 2.0,
        vla_loss_mode: str = "answer_and_traj",
        traj_cfg: TrajectoryTokenizerConfig | None = None,
        max_samples: int | None = None,
        require_full_future: bool = True,
        planning_cams: list[str] | None = None,
        require_all_cams: bool = True,
        image_mean: tuple[float, float, float] = QWEN3_VL_IMAGE_MEAN,
        image_std: tuple[float, float, float] = QWEN3_VL_IMAGE_STD,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = True,
        seed: int = 0,
    ):
        super().__init__()
        if planning_cams is None:
            planning_cams = list(DEFAULT_PLANNING_CAMS)

        # We instantiate the HF PlanningDataset but only call its helpers
        # for waypoint / chat / token logic.  We DO NOT call __getitem__
        # (which runs the HF processor); instead we load frames ourselves
        # and emit them in torchtitan's expected (T, H, W, C) format.
        self._inner = PlanningDataset(
            infos_path=infos_path,
            nusc_root=nusc_root,
            processor=processor,
            max_length=max_length,
            num_past_frames=num_past_frames,
            num_future_waypoints=num_future_waypoints,
            video_fps=video_fps,
            vla_loss_mode=vla_loss_mode,
            traj_cfg=traj_cfg,
            max_samples=max_samples,
            require_full_future=require_full_future,
            planning_cams=planning_cams,
            require_all_cams=require_all_cams,
        )

        self.processor = processor
        self.vla_loss_mode = vla_loss_mode
        self.max_length = int(max_length)
        self.image_mean = image_mean
        self.image_std = image_std
        self.planning_cams = planning_cams
        self.dp_rank = int(dp_rank)
        self.dp_world_size = int(dp_world_size)
        self.infinite = bool(infinite)
        self.seed = int(seed)

        # Use a fresh tokenizer for action tokens that mirrors the inner
        # dataset's configuration.
        if traj_cfg is None:
            traj_cfg = TrajectoryTokenizerConfig(
                num_waypoints=num_future_waypoints,
            )
        self.traj_cfg = traj_cfg
        self.tok = TrajectoryTokenizer(self.traj_cfg)

    # ------------------------------------------------------------------
    # Sample construction
    # ------------------------------------------------------------------

    def _build_sample(self, index_in_keep: int) -> dict[str, Any] | None:
        """Build one torchtitan-format sample from a `_keep` index.

        Returns None when the underlying frames are missing (mirrors the
        HF dataset's defensive skipping)."""
        inner = self._inner
        base_idx = inner._keep[index_in_keep]
        info = inner.infos[base_idx]

        try:
            hist = inner._walk_history(base_idx)
            if len(self.planning_cams) == 1:
                clips_pil = [inner._load_frames(hist, self.planning_cams[0])]
            else:
                clips_pil = inner._load_frames_multicam(hist)
        except FileNotFoundError:
            return None

        # 2. Future waypoints in ego frame.
        wp, valid_mask = inner._compute_waypoints(base_idx)

        # 3. Tokenize trajectory.
        action_tokens = self.tok.encode(wp, with_boundaries=True)

        # 4. Build the chat-template prompt EXACTLY like the HF path.
        user_content = _build_user_content_multicam(info, self.planning_cams)
        proc_messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": "Predicted trajectory:"},
        ]
        text = self.processor.apply_chat_template(
            proc_messages, tokenize=False, add_generation_prompt=False
        )
        # 5. Pre-resize the PIL clips with EXACTLY the same smart_resize
        # geometry that step 10's ``process_video`` will apply, THEN feed
        # them to the HF processor.  This makes the <|video_pad|> count
        # the processor bakes into ``input_ids`` agree by construction
        # with the patch count the MultiModalCollator will emit from
        # step 10's (T, H', W', C) tensor.
        #
        # Why this (option-c) over a pure ``videos_kwargs={"size": ...}``
        # override (the earlier e842eba approach):
        #   * The HF Qwen3VLVideoProcessor's volumetric ``longest_edge``
        #     budget vs torchtitan's per-frame ``max_pixels`` budget are
        #     numerically equal only when num_frames=t_bar (which is true
        #     for our 4-frame past-window setting).  Pre-resizing to the
        #     SAME (H', W') decoupled this from the volumetric/per-frame
        #     algorithm choice, so any future change to one budget cannot
        #     drift the other.
        #   * It also drops one round of bicubic resampling inside the
        #     processor (HF resizes from 900x1600 down to 384x704; we now
        #     resize once before the processor sees the frames so HF
        #     finds the dims already round-multiples of 32 and skips its
        #     own resize).  Marginal latency win; main reason is robustness.
        num_frames = max((len(c) for c in clips_pil), default=1)
        factor = QWEN3_VL_PATCH_SIZE * QWEN3_VL_SPATIAL_MERGE_SIZE  # 32

        # Determine the target (H', W') once from the first frame of the
        # first cam (all cams' frames are aligned to a single (h0, w0)
        # upstream in ``_pil_frames_to_uint8_thwc``, and each cam clip
        # shares a single (height, width) within the same scene).
        h0 = clips_pil[0][0].height
        w0 = clips_pil[0][0].width
        target_h, target_w = _titan_smart_resize(
            h0,
            w0,
            factor=factor,
            min_pixels=QWEN3_VL_IMAGE_MIN_PIXELS,
            max_pixels=QWEN3_VL_IMAGE_MAX_PIXELS,
        )

        resized_clips_pil: list[list[Image.Image]] = []
        for clip in clips_pil:
            resized_clip: list[Image.Image] = []
            for frame in clip:
                if frame.mode != "RGB":
                    frame = frame.convert("RGB")
                if (frame.height, frame.width) != (target_h, target_w):
                    # PIL.Image.resize takes (width, height); BICUBIC matches
                    # torchvision.v2's BICUBIC used in process_video.
                    frame = frame.resize(
                        (target_w, target_h), Image.BICUBIC
                    )
                resized_clip.append(frame)
            resized_clips_pil.append(resized_clip)

        # 5b. Tokenize via the FULL processor on the PRE-RESIZED clips: the
        # processor's smart_resize is now a no-op (target_h % factor == 0
        # and target_w % factor == 0 by construction), so its internal
        # grid_thw matches step 10's patch grid exactly.  We discard the
        # processor's video pixel tensors and only keep input_ids; the
        # pixel tensors will be re-emitted from step 10 below using the
        # same (target_h, target_w) geometry.
        from transformers.video_utils import VideoMetadata  # local import
        metadata = [
            VideoMetadata(
                total_num_frames=len(clip),
                fps=2.0,
                frames_indices=list(range(len(clip))),
                height=target_h,
                width=target_w,
            )
            for clip in resized_clips_pil
        ]
        inputs = self.processor(
            text=[text],
            videos=resized_clips_pil,
            video_metadata=metadata,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].squeeze(0)

        # 6. Insert action tokens INSIDE the assistant turn (same as HF).
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

        # 7. Truncate from the prompt-start if too long (NEVER touch action
        # tokens).
        if input_ids.shape[0] > self.max_length:
            overflow = input_ids.shape[0] - self.max_length
            trim_from = max(1, action_insert_start - len(action_tokens) - overflow)
            trim_to = trim_from + overflow
            input_ids = torch.cat(
                [input_ids[:trim_from], input_ids[trim_to:]], dim=0
            )
            action_insert_start -= overflow

        # 8. Build labels: mask everything before the assistant turn.
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

        if self.vla_loss_mode == "traj_only" and action_insert_start is not None:
            labels[:action_insert_start] = -100

        # 9. Mask invalid future waypoints.
        if (valid_mask < 1.0).any() and action_insert_start is not None:
            num_future = wp.shape[0]
            for t in range(num_future):
                if valid_mask[t] == 0.0:
                    bin_x_pos = action_insert_start + 1 + 2 * t
                    bin_y_pos = bin_x_pos + 1
                    if bin_y_pos < labels.shape[0]:
                        labels[bin_x_pos] = -100
                        labels[bin_y_pos] = -100

        # 10. Convert PRE-RESIZED PIL clips to (T, H', W', C) tensors and
        # normalize via torchtitan's shared `process_video` helper.  The
        # clips have already been smart-resized to (target_h, target_w) in
        # step 5 above (both multiples of factor=patch_size*merge_size=32),
        # so process_video's internal smart_resize is a no-op here; it
        # just runs uint8 -> float32 normalization + channel-last permute.
        # By using the same PIL frames the HF processor saw in step 5b, we
        # guarantee the patch count emitted from this tensor agrees
        # bit-for-bit with the <|video_pad|> count in ``input_ids``.
        pixel_values_videos: list[torch.Tensor] = []
        for clip in resized_clips_pil:
            video_uint8 = _pil_frames_to_uint8_thwc(clip)  # (T, H', W', C) uint8
            video = process_video(
                video_uint8,
                patch_size=QWEN3_VL_PATCH_SIZE,
                merge_size=QWEN3_VL_SPATIAL_MERGE_SIZE,
                min_pixels=QWEN3_VL_IMAGE_MIN_PIXELS,
                max_pixels=QWEN3_VL_IMAGE_MAX_PIXELS,
                image_mean=self.image_mean,
                image_std=self.image_std,
            )  # (T, H', W', C) float32, H' and W' multiples of 32
            pixel_values_videos.append(video)

        positions = torch.arange(input_ids.shape[0], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "positions": positions,
            "pixel_values_videos": pixel_values_videos,
        }

    # ------------------------------------------------------------------
    # Iterable protocol
    # ------------------------------------------------------------------

    def _sharded_indices(self) -> Iterable[int]:
        n = len(self._inner)
        # Round-robin shard for DP: rank r visits indices r, r+W, r+2W, ...
        # This is order-preserving across runs (no shuffling) and avoids
        # the need for a samplable map-style dataset wrapper.
        return range(self.dp_rank, n, self.dp_world_size)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        while True:
            for i in self._sharded_indices():
                sample = self._build_sample(i)
                if sample is None:
                    continue
                yield sample
            if not self.infinite:
                break
