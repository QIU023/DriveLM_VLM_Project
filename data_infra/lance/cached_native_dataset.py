"""Dataset + collate over the Tier-2 NATIVE 3-cam cache (cache_3cam_native.py).

Yields rows whose ViT has ALREADY been run + FasterVLM-compressed offline. The
training loop feeds the collated batch straight to
``train_lora.forward_with_cached_vision_tokens`` — no pixel decode, no vision
tower. Each ``__getitem__`` dequantizes the int8 vision tokens on CPU.
"""
from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

import lance


def _dequant(buf: bytes, scale: float, shape: List[int]) -> np.ndarray:
    return np.frombuffer(buf, dtype=np.int8).reshape(shape).astype(np.float32) * float(scale)


class CachedNativeDataset(Dataset):
    def __init__(self, lance_path: str):
        self.lance_path = lance_path
        # Only read the row count in __init__; do NOT hold a Lance handle on the
        # instance — the handle is not fork-safe and a forked copy shared across
        # DataLoader workers / DDP ranks panics the Lance IO scheduler. Each
        # process/worker opens its own handle lazily in _dataset() (keyed by pid).
        self._n = lance.dataset(lance_path).count_rows()
        self._handle = None
        self._handle_pid = None

    def __len__(self) -> int:
        return self._n

    def _dataset(self):
        pid = os.getpid()
        if self._handle is None or self._handle_pid != pid:
            self._handle = lance.dataset(self.lance_path)
            self._handle_pid = pid
        return self._handle

    def __getitem__(self, i: int) -> Dict[str, object]:
        row = self._dataset().take([i]).to_pylist()[0]
        vp = _dequant(row["video_pooler_int8"], row["video_pooler_scale"], row["video_pooler_shape"])
        vd = _dequant(row["video_deepstack_int8"], row["video_deepstack_scale"], row["video_deepstack_shape"])
        ip = _dequant(row["image_pooler_int8"], row["image_pooler_scale"], row["image_pooler_shape"])
        idp = _dequant(row["image_deepstack_int8"], row["image_deepstack_scale"], row["image_deepstack_shape"])
        out = {
            "sample_token": row["sample_token"],
            "input_ids": torch.tensor(row["input_ids"], dtype=torch.long),
            "labels": torch.tensor(row["labels"], dtype=torch.long),
            "attention_mask": torch.tensor(row["attention_mask"], dtype=torch.long),
            "video_grid_thw": torch.tensor(row["video_grid_thw"], dtype=torch.long).reshape(-1, 3),
            "cached_video_pooler": torch.from_numpy(vp.copy()),
            "cached_video_deepstack": torch.from_numpy(vd.copy()),
            "cached_image_pooler": torch.from_numpy(ip.copy()),
            "cached_image_deepstack": torch.from_numpy(idp.copy()),
        }
        if row.get("mm_token_type_ids"):
            out["mm_token_type_ids"] = torch.tensor(row["mm_token_type_ids"], dtype=torch.long)
        if row.get("image_grid_thw"):
            out["image_grid_thw"] = torch.tensor(row["image_grid_thw"], dtype=torch.long).reshape(-1, 3)
        return out


def collate_cached(batch: List[Dict[str, object]]) -> Dict[str, object]:
    """Right-pad input_ids/labels/attention_mask/mm to common len; stack the
    fixed-shape cached vision tensors along a new batch dim."""
    max_len = max(b["input_ids"].shape[0] for b in batch)
    pad_id = 0

    def _pad(key, fill, dtype=None):
        rows = []
        for b in batch:
            t = b[key]
            p = max_len - t.shape[0]
            if p > 0:
                t = torch.cat([t, torch.full((p,), fill, dtype=t.dtype)])
            rows.append(t)
        return torch.stack(rows)

    has_mm = "mm_token_type_ids" in batch[0]
    out = {
        "input_ids": _pad("input_ids", pad_id),
        "labels": _pad("labels", -100),
        "attention_mask": _pad("attention_mask", 0),
        "video_grid_thw": torch.cat([b["video_grid_thw"] for b in batch], dim=0),
        "cached_video_pooler": torch.stack([b["cached_video_pooler"] for b in batch]),
        "cached_video_deepstack": torch.stack([b["cached_video_deepstack"] for b in batch]),
        "cached_image_pooler": torch.stack([b["cached_image_pooler"] for b in batch]),
        "cached_image_deepstack": torch.stack([b["cached_image_deepstack"] for b in batch]),
        "sample_token": [b["sample_token"] for b in batch],
    }
    if has_mm:
        out["mm_token_type_ids"] = _pad("mm_token_type_ids", 0)
    if "image_grid_thw" in batch[0]:
        out["image_grid_thw"] = torch.cat([b["image_grid_thw"] for b in batch], dim=0)
    return out
