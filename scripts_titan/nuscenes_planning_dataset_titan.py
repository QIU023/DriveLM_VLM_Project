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
# Qwen2.5-VL HF processor to produce ALREADY-PATCHIFIED video tensors
# alongside `video_grid_thw`.  torchtitan wants the pre-patchify
# (T, H_pixels, W_pixels, C) form so its own collator can patch.  We
# therefore re-load the raw frames here and skip the processor's vision
# pipeline, while reusing the geometry / waypoint / token logic from the
# HF dataset.

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


# Default image normalisation for the torchtitan vision pipeline.
# Qwen2.5-VL HF processor uses OpenAI-CLIP mean/std; we match those.
QWEN25_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
QWEN25_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)


def _pil_to_thwc_float(frames: list[Image.Image]) -> torch.Tensor:
    """Convert a list of PIL frames into a single (T, H, W, C) float32
    tensor in the [0, 1] range.  Assumes all frames have identical (H, W);
    we resize to the first frame's size if not."""
    if not frames:
        raise ValueError("empty frames list")
    h0, w0 = frames[0].height, frames[0].width
    arrs = []
    for f in frames:
        if (f.height, f.width) != (h0, w0):
            f = f.resize((w0, h0), Image.BILINEAR)
        arrs.append(np.asarray(f, dtype=np.uint8))  # (H, W, C)
    stacked = np.stack(arrs, axis=0)  # (T, H, W, C)
    out = torch.from_numpy(stacked).to(torch.float32) / 255.0
    return out


def _normalise_thwc(
    video: torch.Tensor,
    mean: tuple[float, float, float] = QWEN25_IMAGE_MEAN,
    std: tuple[float, float, float] = QWEN25_IMAGE_STD,
) -> torch.Tensor:
    """In-place-ish normalisation of a (T, H, W, C) float [0,1] tensor."""
    m = torch.tensor(mean, dtype=video.dtype).view(1, 1, 1, 3)
    s = torch.tensor(std, dtype=video.dtype).view(1, 1, 1, 3)
    return (video - m) / s


class NuScenesPlanningDatasetTitan(IterableDataset):
    """torchtitan IterableDataset wrapping the HF PlanningDataset.

    Reuses ``PlanningDataset._walk_history``, ``_load_frames``,
    ``_compute_waypoints``, and ``_build_user_content_multicam`` so the
    per-sample text + waypoint + token logic matches the HF training path
    EXACTLY.  The only difference is that we emit raw (T, H, W, C)
    normalised video tensors plus token IDs, instead of running the
    Qwen2.5-VL HF processor (torchtitan's MultiModalCollator does the
    patching).

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
        image_mean: tuple[float, float, float] = QWEN25_IMAGE_MEAN,
        image_std: tuple[float, float, float] = QWEN25_IMAGE_STD,
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
        # 5. Tokenize TEXT-ONLY (no images): we ignore the processor's
        # vision pipeline.  apply_chat_template has already inserted the
        # <|vision_start|><|video_pad|>*N<|vision_end|> spans, but the
        # number of <|video_pad|> tokens needs to match what torchtitan's
        # collator will produce when it patchifies our videos.  To keep
        # the contract intact, we run the FULL processor here (text + vid
        # pixel tensors that we discard) but only keep input_ids; this
        # guarantees the per-video <|video_pad|> count matches.
        from transformers.video_utils import VideoMetadata  # local import
        metadata = [
            VideoMetadata(
                total_num_frames=len(clip),
                fps=2.0,
                frames_indices=list(range(len(clip))),
                height=clip[0].height,
                width=clip[0].width,
            )
            for clip in clips_pil
        ]
        inputs = self.processor(
            text=[text],
            videos=clips_pil,
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

        # 10. Convert PIL clips to (T, H, W, C) float tensors and
        # normalise.  torchtitan's MultiModalCollator will patchify these.
        pixel_values_videos: list[torch.Tensor] = []
        for clip in clips_pil:
            video = _pil_to_thwc_float(clip)
            video = _normalise_thwc(video, self.image_mean, self.image_std)
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
