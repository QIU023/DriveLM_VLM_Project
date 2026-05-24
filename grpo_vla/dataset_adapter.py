"""veRL-compatible dataset adapter for the nuScenes planning VLA.

veRL expects each training/eval row to be a dict with the structure::

    {
        "prompt": str | list[dict],     # un-expanded chat-template text OR
                                         # OpenAI-style messages list
        "multi_modal_data": {
            "image": [PIL.Image, ...],   # zero or more images
            "video": [list[PIL.Image]],  # zero or more frame lists (one per <video>)
        },
        "ground_truth": ...,             # opaque; passed to reward fn
        "extra_info": {...},             # opaque; passed to reward fn
    }

The "prompt" field is the *unexpanded* chat-template string with a single
``<|video_pad|>`` per video block and a single ``<|image_pad|>`` per image
block — this is what the SGLang / vLLM rollout server consumes. The server
itself runs the processor to expand video/image pad tokens into the per-patch
token sequence, given the PIL frames in ``multi_modal_data``.

This adapter wraps ``scripts/multimodal_planning_dataset.MultiModalPlanningDataset``
(B.5 / B.6) but BYPASSES that class's ``__getitem__`` (which runs the processor
to produce pre-tokenized tensors — exactly what veRL does NOT want). Instead
we replicate the prompt-construction logic (history walk + bbox lookup + HD-map
load + chat template render) and emit the un-expanded text + raw PIL frames.

The reward-side info (GT waypoints, bbox dicts, ego speed) is packed into
``extra_info`` and consumed by ``grpo_vla.reward.compute_reward_scalar``.

NO new learnable module is introduced — the adapter is read-only over the
upstream dataset and reuses ``MultiModalPlanningDataset.__init__`` for sample
filtering / bbox cache / HD map loading, then overrides only the per-item
prompt assembly.

CPU-only; no GPU side effects. Safe to import from veRL Ray workers.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

# Add scripts/ to path so we can import the upstream dataset class without
# modification.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS_DIR = os.path.join(_REPO_ROOT, "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from multimodal_planning_dataset import (  # noqa: E402
    MultiModalPlanningDataset,
    _build_user_content_multimodal,
    BBOX_NONE_TEXT,
)
from planning_dataset import CAN_BUS_SPEED_IDX  # noqa: E402

# Sibling-module import. When executed as a script (test_reward.py adds
# grpo_vla/ to sys.path) ``reward`` is a top-level module; when executed as
# part of the ``grpo_vla`` package (veRL workers) it lives at
# ``grpo_vla.reward``. Try both, prefer the package form so we don't shadow
# any upstream ``reward`` module.
try:
    from grpo_vla.reward import parse_bbox_text  # noqa: E402
except ImportError:  # pragma: no cover -- script-style import fallback
    from reward import parse_bbox_text  # noqa: E402


class VeRLNuScenesDataset(Dataset):
    """veRL-compatible wrapper over ``MultiModalPlanningDataset``.

    Constructor accepts the SAME kwargs as ``MultiModalPlanningDataset`` so it
    can be built from the same B.5 / B.6 config blob, except ``processor`` is
    only used to call ``apply_chat_template`` (text-only, no tokenization /
    no pixel processing).

    Each __getitem__ returns a veRL-shaped dict:

        {
            "prompt":          str (chat-template text with <|video_pad|> /
                                <|image_pad|> single placeholders),
            "multi_modal_data": {
                "video": List[List[PIL.Image]],    # one frame-list per cam
                "image": List[PIL.Image],          # [HD-map BEV]
            },
            "ground_truth":     List[List[float]],  # gt waypoints (T, 2)
            "extra_info": {
                "sample_token":   str,
                "gt_waypoints":   List[List[float]],   # mirror of ground_truth
                "valid_mask":     List[float],         # (T,)
                "ego_state":      {"speed_mps": float},
                "bbox_3d_list":   List[dict],          # parsed bbox dicts
                "bbox_text":      str,                  # raw text from jsonl
                "horizon_s":      float,
            },
        }
    """

    def __init__(self, *args, **kwargs):
        # Defer to MultiModalPlanningDataset for filtering / caching.
        self._base = MultiModalPlanningDataset(*args, **kwargs)

    def __len__(self) -> int:
        return len(self._base)

    # ------------------------------------------------------------------

    def _ego_speed_mps(self, info: dict) -> float:
        cb = info.get("can_bus")
        if cb is None or len(cb) <= CAN_BUS_SPEED_IDX:
            return 0.0
        try:
            return max(0.0, float(cb[CAN_BUS_SPEED_IDX]))
        except (TypeError, ValueError):
            return 0.0

    def __getitem__(self, i: int) -> Dict[str, Any]:
        base = self._base
        base_idx = base._keep[i]
        info = base.infos[base_idx]
        sample_token = info["token"]

        # 1. Camera frames (List[List[PIL.Image]] — one per planning cam)
        hist = base._walk_history(base_idx)
        if len(base.planning_cams) == 1:
            clips: List[List[Image.Image]] = [
                base._load_frames(hist, base.planning_cams[0])
            ]
        else:
            clips = base._load_frames_multicam(hist)

        # 2. GT waypoints + valid mask
        wp, valid_mask = base._compute_waypoints(base_idx)

        # 3. HD-map + bbox text (no dropout in eval/RL rollout; the policy
        #    should see the canonical sample. veRL training rollouts also
        #    re-evaluate the same prompt many times, so per-sample randomness
        #    here would inject variance unrelated to the policy.)
        hdmap_img = base._load_hdmap(sample_token)
        bbox_text = base._lookup_bbox(sample_token)
        if not bbox_text.strip():
            bbox_text = BBOX_NONE_TEXT

        # 4. Build chat-template text WITHOUT tokenization / no image expansion.
        user_content = _build_user_content_multimodal(
            info, base.planning_cams, bbox_text
        )
        proc_messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": "Predicted trajectory:"},
        ]
        prompt_text: str = base.processor.apply_chat_template(
            proc_messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # 5. Pack reward-side info
        bbox_3d_list = parse_bbox_text(bbox_text)
        ego_state = {"speed_mps": self._ego_speed_mps(info)}
        gt_wp_list = wp.astype(np.float32).tolist()
        valid_list = valid_mask.astype(np.float32).tolist()

        return {
            "prompt": prompt_text,
            "multi_modal_data": {
                "video": clips,
                "image": [hdmap_img],
            },
            "ground_truth": gt_wp_list,
            "extra_info": {
                "sample_token": sample_token,
                "gt_waypoints": gt_wp_list,
                "valid_mask": valid_list,
                "ego_state": ego_state,
                "bbox_3d_list": bbox_3d_list,
                "bbox_text": bbox_text,
                "horizon_s": float(base.num_future) / float(base.video_fps),
            },
        }


# ---------------------------------------------------------------------------
# Factory: build VeRLNuScenesDataset from a config dict (B.5 / B.6 yaml schema).
# ---------------------------------------------------------------------------


def build_verl_nuscenes_dataset(
    cfg: Dict[str, Any],
    processor,
    split: str = "val",
) -> VeRLNuScenesDataset:
    """Convenience builder. Mirrors
    ``multimodal_planning_dataset.build_multimodal_planning_dataset`` but
    returns the veRL-shaped wrapper instead."""
    from planning_dataset import NUM_PAST_FRAMES, NUM_FUTURE_WP  # noqa: E402

    infos_key = "infos_train" if split == "train" else "infos_val"
    infos_path = cfg.get(infos_key)
    if not infos_path:
        raise KeyError(f"Missing config key {infos_key!r}")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(
        os.path.join(_SCRIPTS_DIR, "planning_dataset.py")
    )))
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

    return VeRLNuScenesDataset(
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
        hdmap_dir=hdmap_dir,
        bbox_jsonl=bbox_path,
        split=split,
        modality_dropout_p=0.0,   # always dropout-free for RL rollouts
    )
