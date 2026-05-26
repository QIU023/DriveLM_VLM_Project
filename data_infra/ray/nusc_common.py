#!/usr/bin/env python3
"""Shared, dependency-light nuScenes-planning sample logic for the Ray Data
ingestion DAG.

This module re-implements the *per-sample transform* of the legacy
``grpo_vla/build_parquet.py`` path WITHOUT pulling in the heavy SFT dataset
machinery (``scripts/multimodal_planning_dataset.MultiModalPlanningDataset``)
or the transformers ``AutoProcessor``. The legacy path only used the processor
to render a chat-template *text* prompt; the veRL parquet row schema does not
require that, so we build an equivalent plain-text prompt here.

The output row schema EXACTLY mirrors ``build_parquet._process_one`` (the "OK"
payload):

    prompt        : list[dict]   [{"role": "user", "content": <str>}]
    images        : list[dict]   [{"bytes": <jpeg>} ...]  (CAM_FRONT current + HD-map)
    extra_info    : dict         (sample_token, gt_waypoints, valid_mask, ego_state,
                                  bbox_3d_list, bbox_text, horizon_s)
    reward_model  : dict         {"style": "rule", "ground_truth": gt_waypoints}
    data_source   : str          "nusc_planning"

Everything here is CPU-only and import-safe inside Ray workers (no torch /
no transformers / no GPU side effects).
"""
from __future__ import annotations

import io
import json
import os
import pickle
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Constants (mirror scripts/planning_dataset.py + grpo_vla/build_parquet.py)
# ---------------------------------------------------------------------------
NUM_PAST_FRAMES = 4
NUM_FUTURE_WP = 6
VIDEO_FPS = 2.0
CAN_BUS_SPEED_IDX = 13
HDMAP_SIDE_PX = 224
BBOX_NONE_TEXT = "Detected objects: (none)\n"

DEFAULT_PLANNING_CAMS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"]

# Default planning prompt (matches build_parquet._process_one fallback).
PLANNING_PROMPT = (
    "Given the above observations, predict the next 6 ego waypoints "
    "as <traj_start>{12 bin tokens}<traj_end>."
)

_BBOX_LINE_RE = re.compile(
    r"^-\s*(?P<cls>[a-zA-Z_]+)\s+at\s+"
    r"\((?P<cx>-?\d+\.?\d*),\s*(?P<cy>-?\d+\.?\d*),\s*(?P<cz>-?\d+\.?\d*)\)\s*m,\s*"
    r"size\s+(?P<l>-?\d+\.?\d*)x(?P<w>-?\d+\.?\d*)x(?P<h>-?\d+\.?\d*),\s*"
    r"yaw\s+(?P<yaw>-?\d+\.?\d*)\s*rad"
    r"(?:,\s*vel\s+\((?P<vx>-?\d+\.?\d*),\s*(?P<vy>-?\d+\.?\d*)\)\s*m/s)?"
)


# ---------------------------------------------------------------------------
# Geometry / parsing helpers (copied verbatim in spirit from the repo)
# ---------------------------------------------------------------------------
def quat_to_R(q_wxyz) -> np.ndarray:
    """Hamilton (w,x,y,z) -> 3x3 rotation matrix (nuScenes ego2global convention)."""
    w, x, y, z = (float(q_wxyz[0]), float(q_wxyz[1]),
                  float(q_wxyz[2]), float(q_wxyz[3]))
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


def parse_bbox_text(bbox_text: str) -> List[dict]:
    """Parse a bbox_egostate jsonl bbox_text block into structured dicts."""
    if not bbox_text or "none" in bbox_text.lower():
        return []
    out: List[dict] = []
    for raw in bbox_text.splitlines():
        line = raw.strip()
        if not line.startswith("-"):
            continue
        m = _BBOX_LINE_RE.match(line)
        if not m:
            continue
        d = {
            "cls": m.group("cls"),
            "cx": float(m.group("cx")), "cy": float(m.group("cy")),
            "cz": float(m.group("cz")),
            "l": float(m.group("l")), "w": float(m.group("w")),
            "h": float(m.group("h")), "yaw": float(m.group("yaw")),
        }
        if m.group("vx") is not None:
            d["vx"] = float(m.group("vx"))
            d["vy"] = float(m.group("vy"))
        out.append(d)
    return out


def jpeg_bytes(img: Image.Image, max_edge: int = 448, q: int = 75) -> bytes:
    """Force-resize to square (max_edge x max_edge) + JPEG encode (q75).

    Mirrors build_parquet._jpeg_bytes: square images avoid Qwen2.5-VL
    grid_thw drift when mixing 16:9 cam + 1:1 HD-map aspect ratios.
    """
    img = img.convert("RGB").resize((max_edge, max_edge))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=q, optimize=False)
    return buf.getvalue()


def _ego_speed_mps(info: dict) -> float:
    cb = info.get("can_bus")
    if cb is None or len(cb) <= CAN_BUS_SPEED_IDX:
        return 0.0
    try:
        return max(0.0, float(cb[CAN_BUS_SPEED_IDX]))
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# SampleIndex: the dependency-light replacement for the SFT dataset class.
# Builds the keep-list (full-future + all-cams present) and exposes a single
# read_sample(keep_i) that returns a fully-built veRL row.
# ---------------------------------------------------------------------------
class SampleIndex:
    """Holds infos + token->idx map + bbox cache + the filtered keep-list.

    A single instance is built per Ray map worker (lazily, see ray_ingest). It
    is intentionally picklable-friendly but in practice we construct it once per
    worker via an actor-style closure to amortize the ~0.7 s pkl load.
    """

    def __init__(
        self,
        infos_path: str,
        nusc_root: str,
        hdmap_dir: str,
        bbox_jsonl: str,
        split: str,
        planning_cams: Optional[List[str]] = None,
        num_past: int = NUM_PAST_FRAMES,
        num_future: int = NUM_FUTURE_WP,
        require_full_future: bool = True,
        require_all_cams: bool = True,
    ):
        with open(infos_path, "rb") as f:
            blob = pickle.load(f)
        self.infos: List[dict] = blob["infos"] if isinstance(blob, dict) and "infos" in blob else blob
        self.tok2idx: Dict[str, int] = {info["token"]: i for i, info in enumerate(self.infos)}
        self.nusc_root = nusc_root
        self.hdmap_split_dir = os.path.join(hdmap_dir, split)
        self.split = split
        self.planning_cams = list(planning_cams or DEFAULT_PLANNING_CAMS)
        self.num_past = int(num_past)
        self.num_future = int(num_future)

        # bbox cache
        self._bbox: Dict[str, str] = {}
        with open(bbox_jsonl, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                tok = row.get("sample_token")
                if tok is not None:
                    self._bbox[tok] = row.get("bbox_text", "")

        # keep-list: full-future + (optionally) all-cams-present
        keep = []
        for i, info in enumerate(self.infos):
            if require_full_future and not self._has_full_future(i):
                continue
            keep.append(i)
        if require_all_cams and len(self.planning_cams) > 1:
            keep = [i for i in keep if self._all_cams_present(i)]
        self.keep: List[int] = keep

    # ---- chain walkers (mirror planning_dataset) ----
    def _has_full_future(self, base_idx: int) -> bool:
        cur = self.infos[base_idx]
        for _ in range(self.num_future):
            nxt = cur.get("next")
            if not nxt or nxt not in self.tok2idx:
                return False
            cur = self.infos[self.tok2idx[nxt]]
        return True

    def _walk_history(self, base_idx: int) -> List[dict]:
        out = [self.infos[base_idx]]
        cur = self.infos[base_idx]
        for _ in range(self.num_past - 1):
            prev = cur.get("prev")
            if prev and prev in self.tok2idx:
                cur = self.infos[self.tok2idx[prev]]
            out.append(cur)
        return out[::-1]

    def _walk_future(self, base_idx: int) -> List[dict]:
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

    def _image_path(self, info: dict, cam: str) -> str:
        rel = info["cams"][cam]["data_path"]
        if rel.startswith("./"):
            rel = rel[2:]
        if rel.startswith("data/nuscenes/"):
            rel = rel[len("data/nuscenes/"):]
        return os.path.join(self.nusc_root, rel)

    def _all_cams_present(self, base_idx: int) -> bool:
        hist = self._walk_history(base_idx)
        for h in hist:
            for cam in self.planning_cams:
                if cam not in h.get("cams", {}):
                    return False
                if not os.path.exists(self._image_path(h, cam)):
                    return False
        return True

    def _load_hdmap(self, sample_token: str) -> Image.Image:
        path = os.path.join(self.hdmap_split_dir, f"{sample_token}.png")
        if not os.path.isfile(path):
            return Image.new("RGB", (HDMAP_SIDE_PX, HDMAP_SIDE_PX), color=(0, 0, 0))
        return Image.open(path).convert("RGB")

    def _lookup_bbox(self, sample_token: str) -> str:
        return self._bbox.get(sample_token, "")

    # ---- the full per-sample build (mirrors build_parquet._process_one) ----
    def __len__(self) -> int:
        return len(self.keep)

    def token_for(self, keep_i: int) -> str:
        return self.infos[self.keep[keep_i]]["token"]

    def build_row(self, keep_i: int, max_edge: int = 448, jpeg_q: int = 75) -> Dict[str, Any]:
        base_idx = self.keep[keep_i]
        info = self.infos[base_idx]
        sample_token = info["token"]

        # --- stage: build trajectory (gt waypoints + valid mask) ---
        wp, valid_mask = self._compute_waypoints(base_idx)
        gt_wp_list = wp.astype(np.float32).tolist()
        valid_list = valid_mask.astype(np.float32).tolist()

        # --- stage: decode + resize cameras (CAM_FRONT current frame only) ---
        # build_parquet keeps ONLY CAM_FRONT current (last history frame).
        hist = self._walk_history(base_idx)
        cam0 = self.planning_cams[0]
        cur_frame = Image.open(self._image_path(hist[-1], cam0)).convert("RGB")
        images_payload: List[dict] = [{"bytes": jpeg_bytes(cur_frame, max_edge, jpeg_q)}]
        marker_blocks: List[str] = ["Camera FRONT (current): <image>"]

        # --- stage: render/attach HD-map ---
        hdmap_img = self._load_hdmap(sample_token)
        images_payload.append({"bytes": jpeg_bytes(hdmap_img, max_edge, jpeg_q)})
        marker_blocks.append("HD-map BEV: <image>")

        # --- stage: serialize bbox + ego ---
        bbox_text = self._lookup_bbox(sample_token)
        if not bbox_text.strip():
            bbox_text = BBOX_NONE_TEXT
        bbox_3d_list = parse_bbox_text(bbox_text)
        ego_speed_mps = _ego_speed_mps(info)

        content = "\n".join(marker_blocks)
        content += f"\n\nDetected objects in ego frame:\n{bbox_text}\n"
        content += f"\nEgo speed at current frame: {ego_speed_mps:.2f} m/s\n"
        content += f"\n{PLANNING_PROMPT}"

        extra_info = {
            "sample_token": sample_token,
            "gt_waypoints": gt_wp_list,
            "valid_mask": valid_list,
            "ego_state": {"speed_mps": ego_speed_mps},
            "bbox_3d_list": bbox_3d_list,
            "bbox_text": bbox_text,
            "horizon_s": float(self.num_future) / float(VIDEO_FPS),
        }
        return {
            "prompt": [{"role": "user", "content": content}],
            "images": images_payload,
            "extra_info": extra_info,
            "reward_model": {"style": "rule", "ground_truth": gt_wp_list},
            "data_source": "nusc_planning",
        }


# ---------------------------------------------------------------------------
# Path resolution (shared default locations)
# ---------------------------------------------------------------------------
REPO_ROOT = "/workspace/DriveLM_VLM_Project"


def default_paths(split: str = "val") -> Dict[str, str]:
    return {
        "infos_path": os.path.join(
            REPO_ROOT, "data", "uniad_infos", f"nuscenes_infos_temporal_{split}.pkl"
        ),
        "nusc_root": os.path.join(REPO_ROOT, "data", "nuscenes"),
        "hdmap_dir": os.path.join(REPO_ROOT, "data", "preproc", "hdmap_bev"),
        "bbox_jsonl": os.path.join(
            REPO_ROOT, "data", "preproc", f"bbox_egostate_{split}.jsonl"
        ),
    }
