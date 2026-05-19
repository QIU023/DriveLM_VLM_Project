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
# AutoVLA (NeurIPS'25) feeds three forward-arc cameras (CAM_FRONT,
# CAM_FRONT_LEFT, CAM_FRONT_RIGHT), each as a separate <video> block of 4
# frames @ 2 Hz. The user prompt names each block so the LM can attribute
# context across cameras. _build_user_text_multicam below produces this format.
DEFAULT_PLANNING_CAMS = ["CAM_FRONT"]
CAM_LABELS_3 = {
    "CAM_FRONT": "Front camera",
    "CAM_FRONT_LEFT": "Front-left camera",
    "CAM_FRONT_RIGHT": "Front-right camera",
    "CAM_BACK": "Back camera",
    "CAM_BACK_LEFT": "Back-left camera",
    "CAM_BACK_RIGHT": "Back-right camera",
}


def _multicam_prompt_suffix(cams: List[str]) -> str:
    """Suffix appended to the per-sample ego-speed preamble for the multi-cam
    variant. Single-cam keeps the original PROMPT_TEXT for back-compat."""
    if len(cams) == 1 and cams[0] == "CAM_FRONT":
        return PROMPT_TEXT
    cam_phrase = ", ".join(
        CAM_LABELS_3.get(c, c).lower() for c in cams
    )
    return (
        f"Given 4 past frames @ 2Hz from each of {len(cams)} cameras "
        f"({cam_phrase}), predict the ego vehicle's next 6 waypoints at 2 Hz "
        f"(3 s horizon) in ego frame."
    )


def _build_user_content_multicam(info: dict, cams: List[str]) -> list:
    """Return the `content` list for the user turn given a list of cameras.

    Single-cam mode: one {"type":"video"} block + ego speed + PROMPT_TEXT.
    Multi-cam mode (e.g. AutoVLA 3-cam): interleaved
        "<CamLabel>: " {video} ... ego_speed + multi-cam prompt text.

    The chat template (apply_chat_template) replaces each {"type":"video"}
    entry with a <|vision_start|><|video_pad|><|vision_end|> token triple in
    insertion order. The processor's `videos=[clip1, clip2, ...]` argument
    must contain one clip per video block, in the same order.
    """
    if len(cams) == 1 and cams[0] == "CAM_FRONT":
        return [
            {"type": "video"},
            {"type": "text", "text": _format_ego_speed_preamble(info) + PROMPT_TEXT},
        ]
    content: list = []
    for cam in cams:
        label = CAM_LABELS_3.get(cam, cam)
        content.append({"type": "text", "text": f"{label}: "})
        content.append({"type": "video"})
        content.append({"type": "text", "text": " "})
    # Trailing task prompt + ego speed
    content.append({
        "type": "text",
        "text": _format_ego_speed_preamble(info) + _multicam_prompt_suffix(cams),
    })
    return content
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
        planning_cams: Optional[List[str]] = None,
        require_all_cams: bool = True,
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
        # Multi-camera config. CAM_FRONT-only is the default and exactly
        # matches the pre-3cam dataset behavior (back-compat for existing
        # configs). With multiple cams, each cam is emitted as a separate
        # <video> block in the user turn (AutoVLA-style).
        self.planning_cams: List[str] = list(planning_cams or DEFAULT_PLANNING_CAMS)
        self.require_all_cams = bool(require_all_cams)

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

        # In multi-cam mode, the FL/FR images may not exist on disk for every
        # keyframe (depending on which nuScenes blob tarballs were extracted).
        # Filter to indices for which *every* requested cam file exists across
        # all num_past history frames. This is stricter than CAM_FRONT-only
        # filtering and prevents __getitem__ from hitting a FileNotFoundError
        # mid-training.
        if self.require_all_cams and len(self.planning_cams) > 1:
            ok_idx = []
            for i in self._keep:
                if self._all_cams_present(i):
                    ok_idx.append(i)
            dropped = len(self._keep) - len(ok_idx)
            self._keep = ok_idx
            if dropped:
                print(
                    f"[PlanningDataset] cams={self.planning_cams}: "
                    f"dropped {dropped} samples missing one or more cam files; "
                    f"keeping {len(self._keep)}"
                )

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

    def _image_path(self, info: dict, cam: str = "CAM_FRONT") -> str:
        rel = info["cams"][cam]["data_path"]
        # data_path is typically "samples/<cam>/...". Strip a leading
        # "./" or "data/nuscenes/" if some upstream variant included it.
        if rel.startswith("./"):
            rel = rel[2:]
        if rel.startswith("data/nuscenes/"):
            rel = rel[len("data/nuscenes/"):]
        return os.path.join(self.nusc_root, rel)

    def _load_frames(self, hist: List[dict], cam: str = "CAM_FRONT") -> List[Image.Image]:
        frames: List[Image.Image] = []
        for info in hist:
            p = self._image_path(info, cam)
            img = Image.open(p).convert("RGB")
            frames.append(img)
        return frames

    def _load_frames_multicam(self, hist: List[dict]) -> List[List[Image.Image]]:
        """Return one frame list per cam (in self.planning_cams order). Used
        when len(planning_cams)>1; each per-cam clip is passed as a separate
        entry to `processor(videos=[...])`."""
        return [self._load_frames(hist, cam) for cam in self.planning_cams]

    def _all_cams_present(self, base_idx: int) -> bool:
        """True iff every requested cam has an on-disk file for every
        history frame at this index (current + num_past-1 prev)."""
        hist = self._walk_history(base_idx)
        for h in hist:
            for cam in self.planning_cams:
                if cam not in h.get("cams", {}):
                    return False
                p = self._image_path(h, cam)
                if not os.path.exists(p):
                    return False
        return True

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        base_idx = self._keep[i]
        info = self.infos[base_idx]

        # 1. Past frames (oldest-first). Single-cam returns one clip; multi-cam
        # returns one clip per camera in self.planning_cams order.
        hist = self._walk_history(base_idx)
        if len(self.planning_cams) == 1:
            clips: List[List[Image.Image]] = [self._load_frames(hist, self.planning_cams[0])]
        else:
            clips = self._load_frames_multicam(hist)

        # 2. Future waypoints in ego frame.
        wp, valid_mask = self._compute_waypoints(base_idx)

        # 3. Tokenize trajectory (OpenVLA-bin, K=256 per-dim).
        action_tokens = self.tok.encode(wp, with_boundaries=True)

        # 4. Chat template: one <video> block per cam + ego-speed/prompt;
        # assistant emits a short text wrapper around the trajectory tokens.
        user_content = _build_user_content_multicam(info, self.planning_cams)
        proc_messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": "Predicted trajectory:"},
        ]
        text = self.processor.apply_chat_template(
            proc_messages, tokenize=False, add_generation_prompt=False
        )

        # 5. Run the processor with one video per cam. Qwen2.5-VL's processor
        # builds a per-cam vision_start/vision_end block in token order matching
        # the order of {"type": "video"} entries in user_content.
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
    planning_cams_cfg = cfg.get("planning_cams", None)
    # Tolerate the common YAML mistake of providing a single string instead of
    # a list (`planning_cams: CAM_FRONT_LEFT` -> ["CAM_FRONT_LEFT"]).
    if isinstance(planning_cams_cfg, str):
        planning_cams_cfg = [planning_cams_cfg]
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
        planning_cams=planning_cams_cfg,
        require_all_cams=bool(cfg.get("planning_require_all_cams", True)),
    )
