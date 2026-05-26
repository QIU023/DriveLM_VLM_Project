"""torch.utils.data.Dataset over the Lance multimodal lakehouse (P0).

This is the TRAIN-TIME path that SKIPS the ViT: instead of decoding JPEGs and
running the vision tower per sample, it reads the pre-cached int8 vision tokens
(written by cache_vision_tokens.py) by row index using Lance's zero-copy random
``take`` and dequantizes them on the CPU.

Per item returns:
    vis_tokens  (torch.float32, (n_tok, 2560))  dequantized cached ViT tokens
    hdmap       (torch.uint8,   (224, 224, 3))   decoded HD-map BEV
    bbox_text   (str)
    ego_speed   (float)
    traj_gt     (torch.float32, (6, 2))          future xy waypoints
    scenario    (str)
    sample_token(str)

Lance opens the dataset once per worker (in __init__ of the forked process the
handle is cheap; the actual page reads are lazy + memory-mapped). ``take`` with
a single row index is the documented zero-copy random-access pattern.
"""
from __future__ import annotations

import io
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

import lance

_VIS_COLS = ["vis_tokens_int8", "vis_tokens_scale", "vis_tokens_shape"]
_META_COLS = ["sample_token", "scenario", "hdmap_png", "bbox_text", "ego_speed", "traj_gt"]


def _dequant(buf: bytes, scale: float, shape: List[int]) -> np.ndarray:
    return np.frombuffer(buf, dtype=np.int8).reshape(shape).astype(np.float32) * scale


class LanceMMDataset(Dataset):
    """Reads cached vision tokens + modalities from a Lance dataset by row index."""

    def __init__(self, lance_path: str, columns: Optional[List[str]] = None,
                 decode_hdmap: bool = True):
        self.lance_path = lance_path
        self.decode_hdmap = decode_hdmap
        self.columns = columns or (_VIS_COLS + _META_COLS)
        # Validate the vis columns exist (fail loud — a missing cache is a bug,
        # not a runtime-recovery scenario).
        ds = lance.dataset(lance_path)
        missing = [c for c in _VIS_COLS if c not in ds.schema.names]
        if missing:
            raise RuntimeError(
                f"{lance_path} is missing cached vision columns {missing}; "
                f"run cache_vision_tokens.py first."
            )
        self._n = ds.count_rows()
        self._ds = ds  # reused in the main process; workers re-open lazily below.

    def __len__(self) -> int:
        return self._n

    def _dataset(self) -> "lance.LanceDataset":
        # Each DataLoader worker is a forked process; the Lance handle is not
        # guaranteed fork-safe, so re-open lazily per worker (cheap — metadata
        # only; page reads stay memory-mapped).
        info = torch.utils.data.get_worker_info()
        if info is None:
            return self._ds
        cache = getattr(self, "_worker_ds", None)
        if cache is None:
            cache = lance.dataset(self.lance_path)
            self._worker_ds = cache
        return cache

    def __getitem__(self, i: int) -> Dict[str, object]:
        ds = self._dataset()
        row = ds.take([i], columns=self.columns).to_pylist()[0]

        vis = _dequant(row["vis_tokens_int8"], row["vis_tokens_scale"],
                       row["vis_tokens_shape"])
        item: Dict[str, object] = {
            "vis_tokens": torch.from_numpy(vis.copy()),  # (n_tok, 2560) f32
            "sample_token": row["sample_token"],
            "scenario": row["scenario"],
            "bbox_text": row["bbox_text"],
            "ego_speed": float(row["ego_speed"]),
            "traj_gt": torch.tensor(row["traj_gt"], dtype=torch.float32).reshape(-1, 2),
        }
        if self.decode_hdmap:
            from PIL import Image

            im = Image.open(io.BytesIO(row["hdmap_png"])).convert("RGB")
            item["hdmap"] = torch.from_numpy(np.array(im, dtype=np.uint8))
        return item


def collate(batch: List[Dict[str, object]]) -> Dict[str, object]:
    """Stack vis_tokens (same shape across rows) + hdmap + traj; keep text as lists."""
    out: Dict[str, object] = {
        "vis_tokens": torch.stack([b["vis_tokens"] for b in batch]),
        "traj_gt": torch.stack([b["traj_gt"] for b in batch]),
        "ego_speed": torch.tensor([b["ego_speed"] for b in batch], dtype=torch.float32),
        "sample_token": [b["sample_token"] for b in batch],
        "scenario": [b["scenario"] for b in batch],
        "bbox_text": [b["bbox_text"] for b in batch],
    }
    if "hdmap" in batch[0]:
        out["hdmap"] = torch.stack([b["hdmap"] for b in batch])
    return out
