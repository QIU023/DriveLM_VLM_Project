"""nuScenes ego-trajectory planning dataset for Qwen2.5-VL VLA training.

Each item is built on-the-fly from UniAD's preprocessed temporal infos
(`nuscenes_infos_temporal_train.pkl` / `_val.pkl`) plus the CAM_FRONT images
that live under `<nusc_root>/samples/CAM_FRONT/...`.

Per-sample contents (matches what the existing `train_lora.collate_fn` expects):
  - 4 past CAM_FRONT frames @ 2 Hz (current + 3 prev keyframes)
  - 6 future ego waypoints @ 2 Hz (3 s horizon) in the *current* ego frame
  - OpenVLA-bin tokenized trajectory appended to the assistant turn
  - Loss mask follows `vla_loss_mode` ("answer_and_traj" | "traj_only")

UniAD infos pkl structure (per entry — verified 2026-05-19):
  token, scene_token, prev, next  -> tokens that chain at 2 Hz
  frame_idx                       -> 0-indexed within-scene keyframe id
  ego2global_translation [x,y,z]  -> global ego position (metres)
  ego2global_rotation    [w,x,y,z]-> ego pose quaternion (Hamilton, w-first)
  cams.CAM_FRONT.data_path        -> "samples/CAM_FRONT/..." relative to nusc_root
  cams.CAM_FRONT.timestamp        -> microseconds (canonical)
  timestamp                       -> lidar canonical timestamp (microseconds)

Coordinate convention (matches VAD/UniAD/AutoVLA planning heads):
  ego frame at current keyframe -> x forward, y left (right-handed, z up).
  Δx, Δy = R_cur^T @ (pos_future - pos_cur), then we drop z.
  R_cur = quaternion-to-3x3 rotation of `ego2global_rotation`.

For temporal endpoints (less than 3 prev / 6 future available), we:
  - history: pad backward with the earliest available frame's image
  - future:  emit zero waypoints for missing steps AND mark them invalid in a
             returned mask so eval can skip them.
"""
from __future__ import annotations

import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

try:
    # Local import; this module sits next to trajectory_tokenizer.py
    from trajectory_tokenizer import (
        TrajectoryTokenizer,
        TrajectoryTokenizerConfig,
    )
except ImportError:  # pragma: no cover
    from .trajectory_tokenizer import (  # type: ignore
        TrajectoryTokenizer,
        TrajectoryTokenizerConfig,
    )


# ============================================================================
# Geometry helpers
# ============================================================================

def quat_to_R(q_wxyz: List[float]) -> np.ndarray:
    """Hamilton (w,x,y,z) -> 3x3 rotation matrix. nuScenes ego2global_rotation
    convention. Reference: nuscenes-devkit pyquaternion.Quaternion(rot).rotation_matrix.
    """
    w, x, y, z = float(q_wxyz[0]), float(q_wxyz[1]), float(q_wxyz[2]), float(q_wxyz[3])
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    return np.array(
        [
            [1.0 - s * (y * y + z * z), s * (x * y - z * w),       s * (x * z + y * w)],
            [s * (x * y + z * w),       1.0 - s * (x * x + z * z), s * (y * z - x * w)],
            [s * (x * z - y * w),       s * (y * z + x * w),       1.0 - s * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


# ============================================================================
# Dataset
# ============================================================================

NUM_PAST_FRAMES = 4    # current + 3 prev
NUM_FUTURE_WP = 6      # 6 waypoints over 3 s @ 2 Hz

PROMPT_TEXT = (
    "Given 4 past front-camera frames @ 2Hz, predict the ego vehicle's "
    "next 6 waypoints at 2 Hz (3 s horizon) in ego frame."
)
SYSTEM_TEXT = (
    "You are an autonomous driving planner. Predict the ego vehicle's "
    "future trajectory as a sequence of 2-D waypoints in the current ego frame."
)

# can_bus layout in UniAD/BEVFormer preprocessing (verified 2026-05-19 against
# this repo's nuscenes_infos_temporal_train.pkl):
#   [0:3]   = global ego translation (x,y,z) (m)
#   [3:7]   = ego rotation quaternion (w,x,y,z)
#   [7:10]  = linear accel in ego frame (ax,ay,az) (m/s^2); slot 9 ≈ g (z accel)
#   [10:13] = rotation rate (wx,wy,wz) (rad/s)
#   [13]    = scalar ego speed magnitude from the CAN bus (m/s)  <-- the field
#             we want; correlates 0.93 with finite-diff |Δp|/Δt across keyframes
#   [14:16] = zero-padded (slots reserved by BEVFormer pipeline)
#   [16:18] = patch angle fields (set later in the BEV pipeline; 0 here)
# This scalar ranges 0..18 m/s on the nuScenes trainval set with mean ~4.9 m/s.
# Ref: "Is Ego Status All You Need?" (arxiv 2312.03031) showed that conditioning
# on this scalar alone is enough to bring open-loop L2 from ~2 m to ~0.7 m on
# the nuScenes planning protocol with vision baselines.
CAN_BUS_SPEED_IDX = 13


def _format_ego_speed_preamble(info: dict) -> str:
    """Return 'Ego speed at current frame: X.XX m/s. ' from can_bus[13].

    Falls back gracefully (returns empty string) if can_bus is missing or
    malformed — that way the dataset doesn't hard-fail on a single bad entry.
    """
    cb = info.get("can_bus")
    if cb is None:
        return ""
    try:
        speed = float(cb[CAN_BUS_SPEED_IDX])
    except (IndexError, TypeError, ValueError):
        return ""
    # Sensor noise sometimes makes a near-zero CAN reading slightly negative
    # (-0.83 m/s min on train); clamp to 0 since the prompt should never imply
    # the car is moving backward by inertia at -0.8 m/s.
    speed = max(0.0, speed)
    return f"Ego speed at current frame: {speed:.2f} m/s. "


class PlanningDataset(Dataset):
    """nuScenes-planning dataset that builds DriveLMDataset-compatible items
    directly from UniAD temporal infos + CAM_FRONT image files.

    Public surface matches the keys produced by
    `scripts/train_lora.DriveLMDataset.__getitem__`:
      input_ids, attention_mask, labels, pixel_values_videos, video_grid_thw,
      second_per_grid_ts, image_name.
    """

    def __init__(
        self,
        infos_path: str,
        nusc_root: str,
        processor,
        max_length: int = 2560,
        num_past_frames: int = NUM_PAST_FRAMES,
        num_future_waypoints: int = NUM_FUTURE_WP,
        video_fps: float = 2.0,
        vla_loss_mode: str = "answer_and_traj",
        traj_cfg: TrajectoryTokenizerConfig | None = None,
        max_samples: Optional[int] = None,
        require_full_future: bool = True,
    ):
        if not os.path.exists(infos_path):
            raise FileNotFoundError(f"Infos pkl not found: {infos_path}")
        with open(infos_path, "rb") as f:
            blob = pickle.load(f)
        # UniAD pkl is {'infos': [...], 'metadata': {...}}
        if isinstance(blob, dict) and "infos" in blob:
            self.infos: List[dict] = blob["infos"]
        else:
            self.infos = blob  # type: ignore[assignment]

        # Token -> index
        self.tok2idx: Dict[str, int] = {info["token"]: i for i, info in enumerate(self.infos)}

        self.nusc_root = nusc_root
        self.processor = processor
        self.max_length = int(max_length)
        self.num_past = int(num_past_frames)
        self.num_future = int(num_future_waypoints)
        self.video_fps = float(video_fps)
        self.vla_loss_mode = vla_loss_mode

        self.traj_cfg = traj_cfg or TrajectoryTokenizerConfig(
            num_waypoints=self.num_future,
        )
        self.tok = TrajectoryTokenizer(self.traj_cfg)

        # Filter to samples that have a full future chain (`require_full_future`).
        # Without this, ~22% of items at scene tails would have padded zeros
        # for future steps, which hurts training signal.
        if require_full_future:
            keep_idx = []
            for i, info in enumerate(self.infos):
                if self._has_full_future(i):
                    keep_idx.append(i)
            self._keep = keep_idx
        else:
            self._keep = list(range(len(self.infos)))

        if max_samples is not None:
            self._keep = self._keep[: int(max_samples)]

        # NOTE: we deliberately do NOT pad history — if fewer than num_past prev
        # frames are available, we replicate the earliest frame to fill in. This
        # mirrors VAD/UniAD which front-pad with the current frame on scene starts.

    def __len__(self) -> int:
        return len(self._keep)

    # ------------------------------------------------------------------
    # Internal: scene-chain walkers and waypoint extraction
    # ------------------------------------------------------------------

    def _has_full_future(self, base_idx: int) -> bool:
        cur = self.infos[base_idx]
        for _ in range(self.num_future):
            nxt = cur.get("next")
            if not nxt or nxt not in self.tok2idx:
                return False
            cur = self.infos[self.tok2idx[nxt]]
        return True

    def _walk_history(self, base_idx: int) -> List[dict]:
        """Return [oldest, ..., current] of length self.num_past.
        Replicates earliest frame if scene starts mid-walk."""
        out = [self.infos[base_idx]]
        cur = self.infos[base_idx]
        for _ in range(self.num_past - 1):
            prev = cur.get("prev")
            if prev and prev in self.tok2idx:
                cur = self.infos[self.tok2idx[prev]]
            # else: keep `cur` so we duplicate the earliest available.
            out.append(cur)
        # out is current,prev1,prev2,prev3 — reverse to oldest-first.
        return out[::-1]

    def _walk_future(self, base_idx: int) -> List[dict]:
        """Return up to self.num_future future infos (next_1, ..., next_N). May
        be shorter than num_future if at scene tail."""
        cur = self.infos[base_idx]
        out: List[dict] = []
        for _ in range(self.num_future):
            nxt = cur.get("next")
            if not nxt or nxt not in self.tok2idx:
                break
            cur = self.infos[self.tok2idx[nxt]]
            out.append(cur)
        return out

    def _compute_waypoints(self, base_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute (T, 2) waypoints in current ego frame plus (T,) valid mask
        (1 where the waypoint comes from a real future frame, 0 if padded)."""
        cur = self.infos[base_idx]
        R_cur = quat_to_R(cur["ego2global_rotation"])
        p_cur = np.asarray(cur["ego2global_translation"], dtype=np.float64)
        R_cur_T = R_cur.T

        future = self._walk_future(base_idx)
        wp = np.zeros((self.num_future, 2), dtype=np.float32)
        mask = np.zeros((self.num_future,), dtype=np.float32)
        for i, info in enumerate(future):
            p_f = np.asarray(info["ego2global_translation"], dtype=np.float64)
            local = R_cur_T @ (p_f - p_cur)
            wp[i, 0] = local[0]
            wp[i, 1] = local[1]
            mask[i] = 1.0
        return wp, mask

    # ------------------------------------------------------------------
    # Frame loading
    # ------------------------------------------------------------------

    def _image_path(self, info: dict) -> str:
        rel = info["cams"]["CAM_FRONT"]["data_path"]
        # data_path is typically "samples/CAM_FRONT/...". Strip a leading
        # "./" or "data/nuscenes/" if some upstream variant included it.
        if rel.startswith("./"):
            rel = rel[2:]
        if rel.startswith("data/nuscenes/"):
            rel = rel[len("data/nuscenes/"):]
        return os.path.join(self.nusc_root, rel)

    def _load_frames(self, hist: List[dict]) -> List[Image.Image]:
        frames: List[Image.Image] = []
        for info in hist:
            p = self._image_path(info)
            img = Image.open(p).convert("RGB")
            frames.append(img)
        return frames

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        base_idx = self._keep[i]
        info = self.infos[base_idx]

        # 1. Past frames (oldest-first).
        hist = self._walk_history(base_idx)
        frames = self._load_frames(hist)

        # 2. Future waypoints in ego frame.
        wp, valid_mask = self._compute_waypoints(base_idx)

        # 3. Tokenize trajectory (OpenVLA-bin, K=256 per-dim).
        action_tokens = self.tok.encode(wp, with_boundaries=True)

        # 4. Chat template: 1 video block + prompt; assistant emits a short
        # text wrapper around the trajectory tokens. We mirror the
        # smoke/extract_ego_trajectory.py output structure (1 video, plain text
        # answer, optional traj tokens appended in-place by DriveLMDataset's
        # action_tokens path).
        user_text = _format_ego_speed_preamble(info) + PROMPT_TEXT
        proc_messages = [
            {"role": "user", "content": [{"type": "video"}, {"type": "text", "text": user_text}]},
            {"role": "assistant", "content": "Predicted trajectory:"},
        ]
        text = self.processor.apply_chat_template(
            proc_messages, tokenize=False, add_generation_prompt=False
        )

        # 5. Run the processor with the video frames.
        from transformers.video_utils import VideoMetadata  # local import
        metadata = [
            VideoMetadata(
                total_num_frames=len(frames),
                fps=self.video_fps,
                frames_indices=list(range(len(frames))),
                height=frames[0].height,
                width=frames[0].width,
            )
        ]
        inputs = self.processor(
            text=[text],
            videos=[frames],
            video_metadata=metadata,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        pixel_values_videos = inputs.get("pixel_values_videos")
        video_grid_thw = inputs.get("video_grid_thw")
        second_per_grid_ts = inputs.get("second_per_grid_ts")

        # 6. Append action tokens INSIDE the assistant turn (before <|im_end|>),
        # matching DriveLMDataset's logic verbatim.
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

        # 7. Truncate if too long: trim from the start of the prompt (NEVER
        # touch the action tokens).
        if input_ids.shape[0] > self.max_length:
            overflow = input_ids.shape[0] - self.max_length
            trim_from = max(1, action_insert_start - len(action_tokens) - overflow)
            trim_to = trim_from + overflow
            keep = torch.cat([input_ids[:trim_from], input_ids[trim_to:]], dim=0)
            input_ids = keep
            attention_mask = torch.ones_like(input_ids)
            action_insert_start -= overflow

        # 8. Labels: mask everything before the assistant turn header tokens.
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

        # If the caller asked for traj-only loss, mask the "Predicted trajectory:"
        # answer text as well.
        if self.vla_loss_mode == "traj_only":
            labels[:action_insert_start] = -100

        # 9. Mask future waypoints whose valid_mask=0 (scene tail). We do this
        # by setting the corresponding bin-token labels to -100. Each waypoint
        # occupies 2 bin tokens, in order, between <traj_start> and <traj_end>.
        if (valid_mask < 1.0).any() and action_insert_start is not None:
            # action_tokens layout: [start, bin_x0, bin_y0, ..., bin_xN-1, bin_yN-1, end]
            for t in range(self.num_future):
                if valid_mask[t] == 0.0:
                    bin_x_pos = action_insert_start + 1 + 2 * t  # +1 skips <traj_start>
                    bin_y_pos = bin_x_pos + 1
                    if bin_y_pos < labels.shape[0]:
                        labels[bin_x_pos] = -100
                        labels[bin_y_pos] = -100

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

        result["image_name"] = os.path.basename(self._image_path(info))
        # Eval-side hooks (NOT used by the trainer's collate; eval reads them via
        # a direct __getitem__ call).
        result["_meta_waypoints"] = torch.tensor(wp, dtype=torch.float32)
        result["_meta_valid_mask"] = torch.tensor(valid_mask, dtype=torch.float32)
        result["_meta_token"] = info["token"]
        return result


# ============================================================================
# Convenience: factory used by train_lora.py
# ============================================================================

def build_planning_dataset(cfg: dict, processor, split: str = "train"):
    """Convenience factory called from train_lora.main when
    cfg.dataset_kind == 'nuscenes_planning'.
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

    max_samples = cfg.get(f"{split}_max_samples", None)
    return PlanningDataset(
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
    )
