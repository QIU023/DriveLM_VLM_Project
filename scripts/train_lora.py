"""LoRA fine-tuning of Qwen2.5-VL on DriveLM data.

All hyperparameters are loaded from a YAML config file.
Supports visual token compression experiments via compress_method / compress_ratio.

Usage:
  python train_lora.py --config configs/gh200.yaml --mini
  python train_lora.py --config configs/baseline.yaml
  python train_lora.py --config configs/avg_pool_c4.yaml
  python train_lora.py --config configs/gh200.yaml --bs 4 --epochs 1
"""
import argparse
import json
import math
import os
import sys
import time
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from collections import deque
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from tqdm import tqdm

# ---- Distributed / FSDP via HuggingFace Accelerate -------------------------
# Imports are intentionally light at module scope; FSDP-specific symbols are
# imported lazily inside main() if a distributed launch is detected, so the
# single-GPU smoke-test path continues to work on machines without a fully
# functional FSDP build.
from accelerate import Accelerator

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_config(config_path):
    """Load YAML config with optional base_config inheritance."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    if "base_config" in cfg:
        base_path = cfg.pop("base_config")
        if not os.path.isabs(base_path):
            base_path = os.path.join(os.path.dirname(config_path), base_path)
        base_cfg = load_config(base_path)
        base_cfg.update(cfg)
        cfg = base_cfg
    return cfg


class DriveLMDataset(Dataset):
    """Dataset for DriveLM QA fine-tuning with Qwen2.5-VL.

    Supports three input-modality modes:
      * ``video_mode=False`` (default): each sample contains ONE ``{"type": "image"}``
        message and a single CAM_FRONT PIL image is fed to ``processor(images=...)``.
      * ``video_mode=True`` (Tier-1 video): each sample contains ONE ``{"type": "video"}``
        message holding ``num_frames`` CAM_FRONT frames; we materialise the list as
        ``[PIL.Image, ...]`` and pass to ``processor(videos=[frames], fps=video_fps, ...)``.
        Transformers >= 5.x routes that through ``Qwen2_5_VLVideoProcessor`` which
        returns ``pixel_values_videos`` + ``video_grid_thw`` + ``second_per_grid_ts``.

    Tier-2 VLA add-on:
      * If a record contains an ``action_tokens: [int...]`` field, those token ids
        are appended (verbatim) to the assistant turn AFTER the answer text and
        the chat template is closed. Loss masking is controlled by
        ``vla_loss_mode``:
            "answer_and_traj" -> compute loss on the answer text + traj tokens (OpenVLA-style)
            "traj_only"       -> compute loss only on the traj tokens (AutoVLA-style)
    """

    def __init__(self, data_path, processor, max_length=512,
                 video_mode=False, num_frames=4, video_fps=2.0,
                 vla_mode: bool = False,
                 vla_loss_mode: str = "answer_and_traj",
                 traj_start_id: int | None = None,
                 traj_end_id: int | None = None):
        with open(data_path, "r") as f:
            self.data = json.load(f)
        self.processor = processor
        self.max_length = max_length
        self.video_mode = video_mode
        self.num_frames = num_frames
        self.video_fps = video_fps
        self.vla_mode = vla_mode
        self.vla_loss_mode = vla_loss_mode
        self.traj_start_id = traj_start_id
        self.traj_end_id = traj_end_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        messages = item["messages"]

        # Build messages (skip system message for simplicity)
        proc_messages = []
        for msg in messages:
            if msg["role"] == "system":
                continue
            proc_messages.append(msg)

        # Extract images / videos and build clean messages
        images = []           # list of PIL.Image — single-image branch
        videos = []           # list of list[PIL.Image] — video branch (per-clip frames)
        image_name = ""
        clean_messages = []
        for msg in proc_messages:
            if isinstance(msg["content"], list):
                new_content = []
                for part in msg["content"]:
                    if part.get("type") == "image":
                        image_path = part["image"]
                        if image_path.startswith("file://"):
                            image_path = image_path[7:]
                        images.append(Image.open(image_path).convert("RGB"))
                        image_name = os.path.basename(image_path)
                        new_content.append({"type": "image"})
                    elif part.get("type") == "video":
                        # part["video"] is list of absolute frame paths (see convert_data_video.py).
                        frame_paths = part["video"]
                        if isinstance(frame_paths, str):
                            frame_paths = [frame_paths]
                        frames = []
                        for fp in frame_paths:
                            if fp.startswith("file://"):
                                fp = fp[7:]
                            frames.append(Image.open(fp).convert("RGB"))
                        videos.append(frames)
                        image_name = os.path.basename(frame_paths[-1])
                        new_content.append({"type": "video"})
                    else:
                        new_content.append(part)
                clean_messages.append({"role": msg["role"], "content": new_content})
            else:
                clean_messages.append(msg)

        # Apply chat template
        text = self.processor.apply_chat_template(
            clean_messages, tokenize=False, add_generation_prompt=False
        )

        # Tokenize with processor.
        # NOTE on processor signature (transformers 5.x, processing_qwen2_5_vl.py):
        #   __call__(images=None, text=None, videos=None, **kwargs)
        # The processor routes images -> image_processor (returns pixel_values + image_grid_thw)
        # and videos -> video_processor (returns pixel_values_videos + video_grid_thw +
        # second_per_grid_ts). fps is consumed inside videos_kwargs.
        proc_kwargs = {
            "text": [text],
            "return_tensors": "pt",
        }
        if self.video_mode and videos:
            proc_kwargs["videos"] = videos
            # Tell video_processor how to compute second_per_grid_ts. Supplying a
            # VideoMetadata with fps lets it derive `sampled_fps` correctly.
            from transformers.video_utils import VideoMetadata
            metadata = []
            for frames in videos:
                metadata.append(VideoMetadata(
                    total_num_frames=len(frames),
                    fps=float(self.video_fps) if self.video_fps else 2.0,
                    frames_indices=list(range(len(frames))),
                    height=frames[0].height,
                    width=frames[0].width,
                ))
            proc_kwargs["video_metadata"] = metadata
        elif images:
            proc_kwargs["images"] = images
        inputs = self.processor(**proc_kwargs)

        # Squeeze batch dimension
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        pixel_values = inputs.get("pixel_values")
        image_grid_thw = inputs.get("image_grid_thw")
        pixel_values_videos = inputs.get("pixel_values_videos")
        video_grid_thw = inputs.get("video_grid_thw")
        second_per_grid_ts = inputs.get("second_per_grid_ts")

        # ============ Tier-2 VLA: append trajectory tokens to assistant turn ============
        # We append the raw action token IDs BEFORE the <|im_end|> closer so they live
        # inside the assistant turn. Append in-place at the end of input_ids; if the
        # last token is <|im_end|>, insert action tokens just before it.
        action_tokens = item.get("action_tokens", []) if self.vla_mode else []
        action_insert_start = None
        if action_tokens:
            im_end_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
            ids_list = input_ids.tolist()
            atok = list(action_tokens)
            # Find the last <|im_end|> (assistant turn closer)
            insert_at = None
            for i in range(len(ids_list) - 1, -1, -1):
                if ids_list[i] == im_end_id:
                    insert_at = i
                    break
            if insert_at is None:
                # No assistant closer found — append at the end.
                insert_at = len(ids_list)
            action_insert_start = insert_at
            new_ids = ids_list[:insert_at] + atok + ids_list[insert_at:]
            input_ids = torch.tensor(new_ids, dtype=input_ids.dtype)
            attention_mask = torch.ones_like(input_ids)

        # Truncate if too long (DON'T cut action tokens — if the prompt is too long,
        # we trim from the prompt side instead so the trajectory target survives).
        if input_ids.shape[0] > self.max_length:
            if action_tokens:
                # Drop tokens just before assistant action block (keep the leading
                # system + question if possible; this is a coarse last-resort trim).
                overflow = input_ids.shape[0] - self.max_length
                # Trim from the start of the user turn (after position 0 system header).
                trim_from = max(1, action_insert_start - len(action_tokens) - overflow)
                trim_to = trim_from + overflow
                keep = torch.cat([input_ids[:trim_from], input_ids[trim_to:]], dim=0)
                input_ids = keep
                attention_mask = torch.ones_like(input_ids)
                action_insert_start -= overflow
            else:
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]

        # Create labels: mask everything before the assistant's response
        labels = input_ids.clone()
        assistant_token_str = "<|im_start|>assistant\n"
        assistant_tokens = self.processor.tokenizer.encode(
            assistant_token_str, add_special_tokens=False
        )
        input_list = input_ids.tolist()
        assistant_start = -1
        for i in range(len(input_list) - len(assistant_tokens) + 1):
            if input_list[i : i + len(assistant_tokens)] == assistant_tokens:
                assistant_start = i + len(assistant_tokens)
                break

        if assistant_start > 0:
            labels[:assistant_start] = -100
        labels[attention_mask == 0] = -100

        # Tier-2 VLA loss mode: optionally mask the answer text and only learn
        # the trajectory tokens. The trajectory block starts where the bin tokens
        # were inserted (action_insert_start) up to action_insert_start + len(atok).
        if self.vla_mode and action_tokens and self.vla_loss_mode == "traj_only":
            if action_insert_start is not None:
                # Mask everything up to but not including the action tokens.
                labels[:action_insert_start] = -100

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        if pixel_values is not None:
            result["pixel_values"] = pixel_values.squeeze(0) if pixel_values.dim() > 3 else pixel_values
        if image_grid_thw is not None:
            result["image_grid_thw"] = image_grid_thw.squeeze(0) if image_grid_thw.dim() > 1 else image_grid_thw
        if pixel_values_videos is not None:
            # shape: (num_video_patches, hidden) — keep as-is, batched in collate_fn.
            result["pixel_values_videos"] = pixel_values_videos.squeeze(0) if pixel_values_videos.dim() > 2 else pixel_values_videos
        if video_grid_thw is not None:
            result["video_grid_thw"] = video_grid_thw.squeeze(0) if video_grid_thw.dim() > 1 else video_grid_thw
        if second_per_grid_ts is not None:
            # Tensor or list[float]; standardize to tensor.
            if not isinstance(second_per_grid_ts, torch.Tensor):
                second_per_grid_ts = torch.tensor(second_per_grid_ts, dtype=torch.float32)
            result["second_per_grid_ts"] = second_per_grid_ts

        # Store image name for CRP importance lookup
        result["image_name"] = image_name if 'image_name' in dir() else ""

        return result


def collate_fn(batch):
    """Custom collate that handles variable-size sequences and pixel_values."""
    max_len = max(item["input_ids"].shape[0] for item in batch)
    pad_token_id = 0  # Qwen uses 0 as pad

    padded_input_ids = []
    padded_attention_mask = []
    padded_labels = []

    for item in batch:
        seq_len = item["input_ids"].shape[0]
        pad_len = max_len - seq_len
        padded_input_ids.append(
            torch.cat([item["input_ids"], torch.full((pad_len,), pad_token_id, dtype=item["input_ids"].dtype)])
        )
        padded_attention_mask.append(
            torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=item["attention_mask"].dtype)])
        )
        padded_labels.append(
            torch.cat([item["labels"], torch.full((pad_len,), -100, dtype=item["labels"].dtype)])
        )

    result = {
        "input_ids": torch.stack(padded_input_ids),
        "attention_mask": torch.stack(padded_attention_mask),
        "labels": torch.stack(padded_labels),
    }

    if "pixel_values" in batch[0]:
        result["pixel_values"] = torch.cat(
            [item["pixel_values"].unsqueeze(0) if item["pixel_values"].dim() == 3 else item["pixel_values"] for item in batch],
            dim=0,
        )
    if "image_grid_thw" in batch[0]:
        result["image_grid_thw"] = torch.cat(
            [item["image_grid_thw"].unsqueeze(0) if item["image_grid_thw"].dim() == 1 else item["image_grid_thw"] for item in batch],
            dim=0,
        )
    # Video tensors. pixel_values_videos has shape (num_patches, hidden) per sample;
    # we concatenate along dim 0 since Qwen2.5-VL flattens patches across the batch
    # and uses video_grid_thw to recover per-sample slices.
    if "pixel_values_videos" in batch[0]:
        result["pixel_values_videos"] = torch.cat(
            [item["pixel_values_videos"] for item in batch], dim=0,
        )
    if "video_grid_thw" in batch[0]:
        result["video_grid_thw"] = torch.cat(
            [item["video_grid_thw"].unsqueeze(0) if item["video_grid_thw"].dim() == 1 else item["video_grid_thw"] for item in batch],
            dim=0,
        )
    if "second_per_grid_ts" in batch[0]:
        result["second_per_grid_ts"] = torch.cat(
            [item["second_per_grid_ts"].view(-1) for item in batch], dim=0,
        )
    if "image_name" in batch[0]:
        result["image_names"] = [item["image_name"] for item in batch]
    return result


# --------------- Visual token compression ---------------

def get_base_model(model):
    """Unwrap PEFT to get the original Qwen2.5-VL model."""
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        return model.base_model.model
    return model


_CRP_IMPORTANCE = {}  # Global cache for CRP precomputed importance


def forward_with_compression(model, batch, compress_method, compress_ratio, image_token_id):
    """Forward pass with optional visual token compression.

    For compress_method == "none", falls through to the normal model forward.
    Otherwise:
      1. Run vision encoder on the base model
      2. Compress visual tokens
      3. Adjust input_ids (remove excess image placeholders)
      4. Build inputs_embeds with compressed visual tokens
      5. Forward through LoRA-wrapped LLM with proper 3D RoPE positions
    """
    # Strip non-tensor keys before model forward
    image_names = batch.pop("image_names", [])
    if compress_method == "none" or "pixel_values" not in batch:
        return model(**batch)

    from visual_compress import compress_visual_tokens

    base = get_base_model(model)
    device = batch["input_ids"].device

    # 1. Vision encoder (no grad — LoRA is only on LLM layers)
    vis_dtype = next(base.model.visual.parameters()).dtype
    with torch.no_grad():
        vis_out = base.model.visual(batch["pixel_values"].to(vis_dtype), grid_thw=batch["image_grid_thw"])
        image_embeds = vis_out.pooler_output if hasattr(vis_out, "pooler_output") else vis_out
        if isinstance(image_embeds, (tuple, list)):
            image_embeds = image_embeds[0]
        image_embeds = image_embeds.detach()
    del vis_out  # free last_hidden_state

    # 2. Compress — use post-merger grid (the merger does 2x2 spatial merge,
    #    so actual token grid is grid_thw with h/2, w/2)
    raw_grid = batch["image_grid_thw"]
    merge_size = getattr(base.model.visual, "spatial_merge_size", 2)
    post_grid = raw_grid.clone()
    post_grid[:, 1] = raw_grid[:, 1] // merge_size
    post_grid[:, 2] = raw_grid[:, 2] // merge_size
    # Build importance list for CRP methods
    importance_list = None
    if compress_method in ("crp", "crp_merge") and _CRP_IMPORTANCE:
        importance_list = [_CRP_IMPORTANCE.get(n) for n in image_names]

    compressed, new_grid_thw = compress_visual_tokens(image_embeds, post_grid, compress_method, compress_ratio, importance_list=importance_list)
    del image_embeds  # free pre-compression tokens

    # per-image token counts
    orig_counts = (post_grid[:, 0] * post_grid[:, 1] * post_grid[:, 2]).tolist()
    new_counts = (new_grid_thw[:, 0] * new_grid_thw[:, 1] * new_grid_thw[:, 2]).tolist()

    input_ids = batch["input_ids"]
    attn_mask = batch["attention_mask"]
    labels = batch["labels"]
    B = input_ids.shape[0]

    # 3. Remove excess image-placeholder tokens from each sample
    new_ids_list, new_mask_list, new_lab_list = [], [], []
    img_idx = 0  # pointer into the per-image counts

    for b in range(B):
        ids = input_ids[b]
        msk = attn_mask[b]
        lab = labels[b]

        img_pos = (ids == image_token_id).nonzero(as_tuple=True)[0]
        n_img = len(img_pos)

        if n_img == 0:
            new_ids_list.append(ids)
            new_mask_list.append(msk)
            new_lab_list.append(lab)
            continue

        n_keep = int(new_counts[img_idx])
        img_idx += 1
        n_remove = n_img - n_keep

        if n_remove <= 0:
            new_ids_list.append(ids)
            new_mask_list.append(msk)
            new_lab_list.append(lab)
            continue

        # remove from the END of the image-placeholder block
        remove_pos = img_pos[n_keep:]
        keep = torch.ones(len(ids), dtype=torch.bool, device=device)
        keep[remove_pos] = False
        new_ids_list.append(ids[keep])
        new_mask_list.append(msk[keep])
        new_lab_list.append(lab[keep])

    # 4. Pad to max length
    max_len = max(t.shape[0] for t in new_ids_list)
    for i in range(B):
        pad = max_len - new_ids_list[i].shape[0]
        if pad > 0:
            new_ids_list[i] = torch.cat([new_ids_list[i], torch.zeros(pad, dtype=new_ids_list[i].dtype, device=device)])
            new_mask_list[i] = torch.cat([new_mask_list[i], torch.zeros(pad, dtype=new_mask_list[i].dtype, device=device)])
            new_lab_list[i] = torch.cat([new_lab_list[i], torch.full((pad,), -100, dtype=new_lab_list[i].dtype, device=device)])

    new_input_ids = torch.stack(new_ids_list)
    new_attn_mask = torch.stack(new_mask_list)
    new_labels = torch.stack(new_lab_list)

    # 5. Build inputs_embeds
    inputs_embeds = base.model.language_model.embed_tokens(new_input_ids).clone()
    img_mask = new_input_ids == image_token_id
    inputs_embeds[img_mask] = compressed.to(inputs_embeds.dtype)

    # 6. Forward — pass input_ids for 3D RoPE position computation,
    #    inputs_embeds for actual content, new_grid_thw for spatial dims
    outputs = model(
        input_ids=new_input_ids,
        inputs_embeds=inputs_embeds,
        attention_mask=new_attn_mask,
        image_grid_thw=new_grid_thw,
        labels=new_labels,
    )
    return outputs


# --------------- Cross-frame video token compression (planning VLA) ---------------


def _factor_grid_thw_for_count(target: int, merge_size: int = 2) -> "tuple[int, int, int]":
    """Choose a ``video_grid_thw = (T, H_pre, W_pre)`` that maps to ``target``
    LM-side video-pad placeholders after the spatial merger.

    The Qwen2.5-VL video pipeline emits exactly
    ``T_pre * (H_pre // merge) * (W_pre // merge)`` placeholders for one video
    of grid ``(T_pre, H_pre, W_pre)``. After cross-frame compression we always
    collapse to ``T_pre = 1`` (we treat the compressed bag of tokens as one
    temporal slice), so we need ``H_post * W_post == target`` where
    ``H_post = H_pre // merge`` etc.

    We pick the factor pair ``(h, w)`` of ``target`` with the smallest aspect
    ratio (most square-ish). Returns ``(1, h * merge, w * merge)`` so that the
    grid is exactly representable in ``video_grid_thw`` (which is stored in
    pre-merger units).
    """
    best = None
    for h in range(1, int(target ** 0.5) + 1):
        if target % h == 0:
            w = target // h
            ar = max(h, w) / min(h, w)
            if best is None or ar < best[0]:
                best = (ar, h, w)
    if best is None:
        # Shouldn't happen for target >= 1; fall back to (1, target).
        return (1, 1 * merge_size, target * merge_size)
    _, h, w = best
    return (1, h * merge_size, w * merge_size)


def forward_with_video_xframe_compression(
    model,
    batch,
    compressor,
    video_token_id: int,
    num_past_frames: int,
    merge_size: int = 2,
):
    """Forward pass with cross-frame visual token compression.

    Designed for the Qwen2.5-VL **video** branch of nuScenes planning. The
    dataset emits one video clip per sample with grid ``(T_pre, H_pre, W_pre)``
    in ``video_grid_thw``; the vision tower's merger collapses each 2x2 spatial
    block into one token, so the post-merger token count per sample is
    ``T_pre * H_pre/2 * W_pre/2``. We:

    1. Use the model's own ``get_video_features`` to run the (frozen) vision
       tower in the standard FSDP-aware way. The model's forward path
       internally summons unsharded params; we call ``get_video_features``
       through a monkey-patch so the downstream ``model(...)`` invocation
       sees the compressed tokens directly instead of recomputing.
    2. Reshape to ``(B, T_post=T_pre, N, D)`` where ``N = (H_pre/2)*(W_pre/2)``
       is the per-frame-group spatial token count.
    3. Apply ``compressor(frames)`` -> ``(B, N', D)``. The compressor has its
       own learnable parameters (gradient flows through it).
    4. Drop the (``T_post - 1``) excess ``<|video_pad|>`` placeholders per
       sample from ``input_ids`` / ``attention_mask`` / ``labels`` so that the
       LM sees exactly ``N'`` video tokens.
    5. Build a fake ``pixel_values_videos`` (1-patch tensor; ignored by our
       patched ``get_video_features``) so the model's forward enters the
       video-branch and runs its built-in masked_scatter using OUR compressed
       tokens.
    6. Construct a new ``video_grid_thw = (1, h, w)`` with
       ``(h//merge)*(w//merge) == N'``.

    The monkey-patch is restored in a try/finally so a raise inside the LM
    forward never leaves the model in a corrupted state.
    """
    # Strip non-tensor / non-model keys before forward
    batch.pop("image_names", None)

    base = get_base_model(model)
    device = batch["input_ids"].device

    pv = batch["pixel_values_videos"]
    grid = batch["video_grid_thw"]  # (num_videos, 3) — concat'd by collate; one video per sample
    if grid.dim() == 1:
        grid = grid.unsqueeze(0)

    B = grid.shape[0]
    t_pre = grid[:, 0].tolist()
    h_post = (grid[:, 1] // merge_size).tolist()
    w_post = (grid[:, 2] // merge_size).tolist()
    n_per_group = h_post[0] * w_post[0]
    for b in range(B):
        if h_post[b] * w_post[b] != n_per_group:
            raise RuntimeError(
                f"cross-frame compression requires per-frame N to match across batch; "
                f"sample {b} has N={h_post[b]*w_post[b]} != {n_per_group}"
            )
    T_post_max = t_pre[0]
    for b in range(B):
        if t_pre[b] != T_post_max:
            raise RuntimeError(
                f"cross-frame compression requires uniform T across batch; "
                f"sample {b} has T={t_pre[b]} != {T_post_max}"
            )

    # 1. Determine N_new (compressor output token count) WITHOUT running the
    # vision tower. We need this up-front to trim input_ids before the model
    # forward (the model uses input_ids' <|video_pad|> count to scatter the
    # compressed embeds). Each compressor exposes ``output_token_count(T, N)``;
    # VTM's signature has an extra kwarg.
    try:
        if hasattr(compressor, "target_tokens"):
            N_new = int(compressor.target_tokens)
        else:
            N_new = int(type(compressor).output_token_count(T_post_max, n_per_group))
    except TypeError:
        # Fallback for compressors with non-static output_token_count.
        N_new = int(compressor.output_token_count(T_post_max, n_per_group))

    # 2. Drop excess <|video_pad|> tokens. Per sample: keep first N_new
    # video-pad positions, drop the rest.
    input_ids = batch["input_ids"]
    attn_mask = batch["attention_mask"]
    labels = batch["labels"]

    new_ids_list, new_mask_list, new_lab_list = [], [], []
    for b in range(B):
        ids = input_ids[b]
        msk = attn_mask[b]
        lab = labels[b]
        vid_pos = (ids == video_token_id).nonzero(as_tuple=True)[0]
        n_vid = len(vid_pos)
        if n_vid == 0:
            new_ids_list.append(ids)
            new_mask_list.append(msk)
            new_lab_list.append(lab)
            continue
        if n_vid < N_new:
            raise RuntimeError(
                f"sample {b}: only {n_vid} video-pad tokens but compressor emits {N_new}; "
                f"max_length truncation may have eaten visual placeholders"
            )
        n_remove = n_vid - N_new
        if n_remove == 0:
            new_ids_list.append(ids)
            new_mask_list.append(msk)
            new_lab_list.append(lab)
            continue
        remove_pos = vid_pos[N_new:]
        keep = torch.ones(len(ids), dtype=torch.bool, device=device)
        keep[remove_pos] = False
        new_ids_list.append(ids[keep])
        new_mask_list.append(msk[keep])
        new_lab_list.append(lab[keep])

    # Pad to max length
    max_len = max(t.shape[0] for t in new_ids_list)
    for i in range(B):
        pad = max_len - new_ids_list[i].shape[0]
        if pad > 0:
            new_ids_list[i] = torch.cat([
                new_ids_list[i],
                torch.zeros(pad, dtype=new_ids_list[i].dtype, device=device),
            ])
            new_mask_list[i] = torch.cat([
                new_mask_list[i],
                torch.zeros(pad, dtype=new_mask_list[i].dtype, device=device),
            ])
            new_lab_list[i] = torch.cat([
                new_lab_list[i],
                torch.full((pad,), -100, dtype=new_lab_list[i].dtype, device=device),
            ])
    new_input_ids = torch.stack(new_ids_list)
    new_attn_mask = torch.stack(new_mask_list)
    new_labels = torch.stack(new_lab_list)

    # 4. New video_grid_thw with T=1 and a clean (h, w) factorization of N_new.
    _, h_pre_new, w_pre_new = _factor_grid_thw_for_count(N_new, merge_size=merge_size)
    new_grid_thw = torch.tensor(
        [[1, h_pre_new, w_pre_new]] * B,
        dtype=grid.dtype, device=device,
    )

    # 5. Monkey-patch `inner.get_video_features` so the model.forward call
    #    runs OUR pipeline (vision encoder under no_grad -> compressor) and
    #    returns the compressed features. Calling the vision tower from
    #    INSIDE model.forward (vs from our pre-step) is the only safe way to
    #    interact with FSDP-sharded visual params: the FSDP root summons them
    #    automatically at the model.forward entry.
    #
    #    The model expects a return with a `.pooler_output` attribute holding
    #    the (total_post_tokens, D) tensor. We mimic that with an ad-hoc
    #    namespace.
    inner = base.model  # Qwen2_5_VLModel
    _orig_get_video_features = inner.get_video_features

    class _FakeVisOut:
        def __init__(self, t):
            self.pooler_output = t

    def _patched_get_video_features(_pv, _grid):  # noqa: ARG001
        # Run original encoder under no_grad (vision tower frozen).
        with torch.no_grad():
            real = _orig_get_video_features(pv, grid)
            embeds = real.pooler_output
            if isinstance(embeds, (tuple, list)):
                # Some HF versions split per-video into a list/tuple.
                embeds = torch.cat([e for e in embeds], dim=0)
            embeds = embeds.detach()
        D_inner = embeds.shape[-1]
        expected = B * T_post_max * n_per_group
        if embeds.shape[0] != expected:
            raise RuntimeError(
                f"vision pooler_output {embeds.shape[0]} != B*T*N {expected}"
            )
        frames_local = embeds.view(B, T_post_max, n_per_group, D_inner)
        compressed_local = compressor(frames_local)  # (B, N_new, D)
        if compressed_local.shape != (B, N_new, D_inner):
            raise RuntimeError(
                f"compressor output {tuple(compressed_local.shape)} != "
                f"expected (B={B}, N_new={N_new}, D={D_inner})"
            )
        # Return per-video tensors as a list so the caller's torch.cat works.
        # We treat the compressed bag as ONE video per sample with N_new tokens.
        per_sample = [compressed_local[b] for b in range(B)]
        return _FakeVisOut(per_sample)

    inner.get_video_features = _patched_get_video_features
    try:
        outputs = model(
            input_ids=new_input_ids,
            attention_mask=new_attn_mask,
            labels=new_labels,
            pixel_values_videos=pv,        # passed through (the patch ignores it)
            video_grid_thw=new_grid_thw,
        )
    finally:
        inner.get_video_features = _orig_get_video_features
    return outputs


# --------------- Q-Former projector (Track A.1, redo) ---------------


def forward_with_video_qformer_projector(
    model,
    batch,
    projector,
    video_token_id: int,
    merge_size: int = 2,
):
    """Forward pass with a BLIP-2-style Q-Former PROJECTOR on top of the
    Qwen2.5-VL in-encoder ``PatchMerger``.

    THIS IS A PROJECTOR REPLACEMENT, not a cross-frame compressor. The Q-Former
    is a **fusion mechanism** (visual<->LM projector) along the same axis as
    A.2 (PixelShuffle) and A.3 (Perceiver Resampler). It is mutually exclusive
    with ``cross_frame_compressor`` (which is for TEMPORAL compression like VTM
    / LongVU / mean-pool — a different ablation axis entirely).

    The HF Qwen2.5-VL flow applies the in-encoder ``PatchMerger`` (2x2 spatial
    merge -> 4x token reduction) inside ``get_video_features``; the output is
    ``(total_post_tokens, lm_dim)`` flattened across all video items in the
    batch. We feed each visual ITEM (one cam-clip) independently through the
    Q-Former — the 64 learnable queries cross-attend the whole flattened bag
    of (T_post * H_post * W_post) post-merger features for that item and emit
    a fixed 64-token output, regardless of input length.

    Per-item output token count is FIXED at ``projector.num_queries`` (default
    64). With 3 cams that's 192 LM placeholders / sample (down from ~1680 raw
    post-merger tokens at min_pixels=max_pixels=109760).

    Multi-cam policy: Q-Former has NO temporal positional encoding, so
    feeding each cam independently is equivalent to feeding the concatenation
    minus the cross-cam attention. We chose the per-item path for parity
    with the resampler shim (A.3) — it's the same code shape and makes the
    A.1/A.3 comparison apples-to-apples.

    The placeholder-trim logic mirrors `forward_with_video_resampler_projector`:
    keep the first ``num_queries`` ``<|video_pad|>`` tokens per visual item,
    drop the rest; rebuild ``video_grid_thw`` as ``(1, h_pre, w_pre)`` with
    ``h_post * w_post == num_queries``.

    Param grad: every projector param (queries + cross-attn + FFN + out_proj)
    is learnable; added to the optimizer by ``main()`` the same way the
    PixelShuffle / Resampler projector params are.

    Args:
        model: PEFT-wrapped Qwen2_5_VL model.
        batch: collated batch dict from the planning dataset.
        projector: ``Qwen2VLQFormerProjector`` instance.
        video_token_id: ``<|video_pad|>`` token id.
        merge_size: in-encoder spatial merge size (Qwen2.5-VL: 2).
    """
    batch.pop("image_names", None)

    base = get_base_model(model)
    device = batch["input_ids"].device

    pv = batch["pixel_values_videos"]
    grid = batch["video_grid_thw"]
    if grid.dim() == 1:
        grid = grid.unsqueeze(0)

    num_items = grid.shape[0]

    # Post-merger per-item token counts.
    t_per_item = grid[:, 0].tolist()
    h_post = (grid[:, 1] // merge_size).tolist()
    w_post = (grid[:, 2] // merge_size).tolist()
    n_post_per_item = [t_per_item[i] * h_post[i] * w_post[i] for i in range(num_items)]
    n_post_total = sum(n_post_per_item)

    # Q-Former does not require uniform per-item shape (the cross-attn handles
    # variable N_vision via key_padding_mask). We still assert uniformity here
    # so the per-item reshape is well-defined and so this code path mirrors
    # the resampler/pixelshuffle invariants — 3-cam x 4f nuScenes satisfies it
    # by construction.
    first_thw = (t_per_item[0], h_post[0], w_post[0])
    for i in range(num_items):
        if (t_per_item[i], h_post[i], w_post[i]) != first_thw:
            raise RuntimeError(
                f"Q-Former projector currently requires identical post-merger "
                f"(t, h, w) across all items. Item {i}="
                f"{(t_per_item[i], h_post[i], w_post[i])} != item 0={first_thw}. "
                f"Mixed-resolution batches require per-shape grouping with "
                f"key_padding_mask (not implemented)."
            )
    t0, h0, w0 = first_thw
    n_compressed_per_item = int(projector.num_queries)

    # ---- 1. Build trimmed input_ids / attention_mask / labels --------------
    input_ids = batch["input_ids"]
    attn_mask = batch["attention_mask"]
    labels = batch["labels"]
    B_lm = input_ids.shape[0]

    items_per_sample = num_items // B_lm
    if items_per_sample * B_lm != num_items:
        raise RuntimeError(
            f"video_grid_thw num_items={num_items} not divisible by LM batch "
            f"size B_lm={B_lm}; items_per_sample is non-uniform."
        )
    n_per_item = n_post_per_item[0]

    new_ids_list, new_mask_list, new_lab_list = [], [], []
    for b in range(B_lm):
        ids = input_ids[b]
        msk = attn_mask[b]
        lab = labels[b]
        vid_pos = (ids == video_token_id).nonzero(as_tuple=True)[0]
        n_vid = len(vid_pos)
        if n_vid == 0:
            new_ids_list.append(ids)
            new_mask_list.append(msk)
            new_lab_list.append(lab)
            continue
        expected_uncompressed = items_per_sample * n_per_item
        if n_vid != expected_uncompressed:
            raise RuntimeError(
                f"sample {b}: found {n_vid} video-pad tokens but expected "
                f"{expected_uncompressed} ({items_per_sample} items x "
                f"{n_per_item} post-merger tokens). max_length truncation may "
                f"have eaten visual placeholders."
            )
        # Per item k, the k-th contiguous block of placeholders is
        # [k*n_per_item .. (k+1)*n_per_item); keep first n_compressed_per_item,
        # drop the rest.
        drop_positions = []
        for k in range(items_per_sample):
            item_start = k * n_per_item
            item_end = item_start + n_per_item
            keep_until = item_start + n_compressed_per_item
            drop_positions.extend(vid_pos[keep_until:item_end].tolist())
        if drop_positions:
            keep = torch.ones(len(ids), dtype=torch.bool, device=device)
            keep[torch.tensor(drop_positions, device=device)] = False
            new_ids_list.append(ids[keep])
            new_mask_list.append(msk[keep])
            new_lab_list.append(lab[keep])
        else:
            new_ids_list.append(ids)
            new_mask_list.append(msk)
            new_lab_list.append(lab)

    # Pad to common length.
    max_len = max(t.shape[0] for t in new_ids_list)
    for i in range(B_lm):
        pad = max_len - new_ids_list[i].shape[0]
        if pad > 0:
            new_ids_list[i] = torch.cat([
                new_ids_list[i],
                torch.zeros(pad, dtype=new_ids_list[i].dtype, device=device),
            ])
            new_mask_list[i] = torch.cat([
                new_mask_list[i],
                torch.zeros(pad, dtype=new_mask_list[i].dtype, device=device),
            ])
            new_lab_list[i] = torch.cat([
                new_lab_list[i],
                torch.full((pad,), -100, dtype=new_lab_list[i].dtype, device=device),
            ])
    new_input_ids = torch.stack(new_ids_list)
    new_attn_mask = torch.stack(new_mask_list)
    new_labels = torch.stack(new_lab_list)

    # ---- 2. New video_grid_thw with compressed shape per item -------------
    # Q-Former emits a 1-D bag of queries (no spatial structure), so we encode
    # as (t=1, h_pre, w_pre) where h_post * w_post == num_queries. The factor
    # helper picks the most-square (h, w) pair.
    _, h_pre_new, w_pre_new = _factor_grid_thw_for_count(
        n_compressed_per_item, merge_size=merge_size,
    )
    new_grid = torch.tensor(
        [[1, h_pre_new, w_pre_new]] * num_items,
        dtype=grid.dtype, device=device,
    )

    # ---- 3. Monkey-patch get_video_features to inject Q-Former embeds -----
    inner = base.model  # Qwen2_5_VLModel
    _orig_get_video_features = inner.get_video_features

    class _FakeVisOut:
        def __init__(self, t):
            self.pooler_output = t

    def _patched_get_video_features(_pv, _grid):  # noqa: ARG001
        # Run original encoder under no_grad (vision tower frozen).
        with torch.no_grad():
            real = _orig_get_video_features(pv, grid)
            embeds = real.pooler_output
            if isinstance(embeds, (tuple, list)):
                embeds = torch.cat([e for e in embeds], dim=0)
            embeds = embeds.detach()
        if embeds.shape[0] != n_post_total:
            raise RuntimeError(
                f"vision pooler_output rows {embeds.shape[0]} != expected "
                f"post-merger total {n_post_total}"
            )
        D = embeds.shape[-1]
        # Per-item reshape (all items share shape, asserted above).
        per_item = embeds.view(num_items, t0 * h0 * w0, D)
        # Cast to projector dtype before forward.
        proj_dtype = next(projector.parameters()).dtype
        # Run Q-Former per-item. Each call returns (1, num_queries, lm_dim);
        # since there's no temporal pos / cross-item state in the Q-Former,
        # we could also stack all items into one projector call — but the
        # per-item loop mirrors the resampler shim and keeps memory bounded.
        compressed_items = []
        for i in range(num_items):
            out_i = projector(
                per_item[i : i + 1].to(proj_dtype),
            )  # (1, num_queries, lm_dim)
            compressed_items.append(out_i.squeeze(0))
        return _FakeVisOut(compressed_items)

    inner.get_video_features = _patched_get_video_features
    try:
        outputs = model(
            input_ids=new_input_ids,
            attention_mask=new_attn_mask,
            labels=new_labels,
            pixel_values_videos=pv,
            video_grid_thw=new_grid,
        )
    finally:
        inner.get_video_features = _orig_get_video_features
    return outputs


@torch.no_grad()
def validate(model, val_loader, compress_method, compress_ratio, image_token_id, val_batches, device,
             *, xframe_compressor=None, video_token_id=None, num_past_frames=None,
             qformer_projector=None):
    """Run validation for val_batches batches.

    Returns ``(val_loss, val_acc, l2_dict)`` where ``l2_dict`` carries
    teacher-forced L2 (metres) at horizons 1/2/3 s and the average over all
    6 waypoints, decoded via the trajectory tokenizer's per-dim bin centres.

    ``l2_dict`` is an empty dict when L2 cannot be computed (xframe mode, or
    no trajectory tokens found in the val batches).
    """
    # Build a CPU-side trajectory tokenizer so we can map bin token-ids back to
    # (Δx, Δy) metres. This mirrors the planning_eval.py decode path but works
    # on teacher-forced labels/logits instead of generated tokens.
    from trajectory_tokenizer import TrajectoryTokenizer, TrajectoryTokenizerConfig
    _traj_cfg = TrajectoryTokenizerConfig()
    _traj_tok = TrajectoryTokenizer(_traj_cfg)
    _bin_base = _traj_cfg.bin_base
    _bin_hi = _bin_base + _traj_cfg.num_bins  # exclusive
    _num_wp = _traj_cfg.num_waypoints  # 6
    _dx_centers = torch.from_numpy(_traj_tok.dx_centers).float()  # (256,)
    _dy_centers = torch.from_numpy(_traj_tok.dy_centers).float()  # (256,)

    model.eval()
    total_loss, count = 0.0, 0
    correct_tokens, total_tokens = 0, 0
    # Mean-of-batch-means aggregation (ST-P3 TemAvg, batched).
    l2_1s_sum = l2_2s_sum = l2_3s_sum = l2_avg_sum = 0.0
    l2_n_batches = 0
    l2_n_samples = 0
    l2_skipped = (xframe_compressor is not None) or (qformer_projector is not None)
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= val_batches:
                break
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            try:
                if qformer_projector is not None:
                    outputs = forward_with_video_qformer_projector(
                        model, batch, qformer_projector, video_token_id,
                    )
                elif xframe_compressor is not None:
                    outputs = forward_with_video_xframe_compression(
                        model, batch, xframe_compressor, video_token_id, num_past_frames,
                    )
                else:
                    outputs = forward_with_compression(model, batch, compress_method, compress_ratio, image_token_id)
                total_loss += outputs.loss.item()
                count += 1
                logits = outputs.logits[:, :-1, :]
                labels = batch["labels"][:, 1:]
                # NOTE: in xframe compression mode, logits seq_len is the
                # COMPRESSED length (after T*N -> N video token reduction) while
                # batch["labels"] is the uncompressed length. The shapes diverge
                # and we cannot align without re-doing the input_ids surgery
                # that forward_with_video_xframe_compression did. Skip token-acc
                # in that case; val_loss is still meaningful.
                if logits.shape[1] == labels.shape[1]:
                    mask = labels != -100
                    if mask.any():
                        preds = logits.argmax(dim=-1)
                        correct_tokens += (preds[mask] == labels[mask]).sum().item()
                        total_tokens += mask.sum().item()

                    # ---- Teacher-forced L2 (metres) over trajectory tokens ----
                    # Identify GT trajectory bin positions in labels. Bin token
                    # ids live in [BIN_BASE, BIN_BASE+256). -100 (ignore) is
                    # already excluded by the half-open range.
                    traj_mask = (labels >= _bin_base) & (labels < _bin_hi)  # (B, T-1)
                    # Require every sample in the batch to expose exactly
                    # 2 * num_waypoints (=12) trajectory token positions.
                    per_sample_counts = traj_mask.sum(dim=1)  # (B,)
                    if per_sample_counts.numel() > 0 and bool((per_sample_counts == 2 * _num_wp).all()):
                        B = labels.shape[0]
                        # GT bin ids reshaped to (B, num_wp, 2)
                        gt_ids = labels[traj_mask].view(B, _num_wp, 2)
                        # Predicted bin ids at the SAME positions (shifted-by-1
                        # alignment already applied above when we sliced logits).
                        pred_ids_full = logits.argmax(dim=-1)  # (B, T-1)
                        pred_ids = pred_ids_full[traj_mask].view(B, _num_wp, 2)
                        # Decode bin id -> bin index -> metre via per-dim centres.
                        gt_bins = (gt_ids - _bin_base).clamp_(0, _traj_cfg.num_bins - 1)
                        pred_bins = (pred_ids - _bin_base).clamp_(0, _traj_cfg.num_bins - 1)
                        dx_c = _dx_centers.to(labels.device)
                        dy_c = _dy_centers.to(labels.device)
                        gt_dx = dx_c[gt_bins[..., 0]]
                        gt_dy = dy_c[gt_bins[..., 1]]
                        pred_dx = dx_c[pred_bins[..., 0]]
                        pred_dy = dy_c[pred_bins[..., 1]]
                        # L2 per waypoint -> (B, num_wp)
                        ddx = pred_dx - gt_dx
                        ddy = pred_dy - gt_dy
                        dist = torch.sqrt(ddx * ddx + ddy * ddy)
                        # Horizon indices: [0.5s, 1s, 1.5s, 2s, 2.5s, 3s]
                        l2_1s_sum += float(dist[:, 1].mean().item())
                        l2_2s_sum += float(dist[:, 3].mean().item())
                        l2_3s_sum += float(dist[:, 5].mean().item())
                        l2_avg_sum += float(dist.mean().item())
                        l2_n_batches += 1
                        l2_n_samples += B
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    continue
                raise
    model.train()
    val_loss = total_loss / max(count, 1)
    val_acc = correct_tokens / max(total_tokens, 1)

    l2_dict: dict = {}
    if not l2_skipped and l2_n_batches > 0:
        l2_dict = {
            "L2_1s": l2_1s_sum / l2_n_batches,
            "L2_2s": l2_2s_sum / l2_n_batches,
            "L2_3s": l2_3s_sum / l2_n_batches,
            "L2_avg": l2_avg_sum / l2_n_batches,
            "n_samples": l2_n_samples,
        }
    return val_loss, val_acc, l2_dict


def _save_model_and_state(accelerator, model, optimizer, scheduler,
                          train_mode, save_path, global_step, epoch, batch_idx,
                          save_processor=None):
    """Distributed-safe checkpoint writer.

    For LoRA / QLoRA we save the adapter only (small, single rank writes).
    For full_sft under FSDP we gather a full state_dict on rank 0 and write
    via HF `save_pretrained` so the result is a drop-in HF checkpoint dir.
    """
    if accelerator.is_main_process:
        os.makedirs(save_path, exist_ok=True)
    accelerator.wait_for_everyone()

    unwrapped = accelerator.unwrap_model(model)
    if train_mode == "full_sft":
        # `accelerator.get_state_dict` gathers FSDP shards onto rank 0 (or
        # returns the local state dict in single-GPU mode).
        state_dict = accelerator.get_state_dict(model)
        if accelerator.is_main_process:
            unwrapped.save_pretrained(
                save_path,
                is_main_process=True,
                save_function=accelerator.save,
                state_dict=state_dict,
                safe_serialization=True,
            )
            if save_processor is not None:
                save_processor.save_pretrained(save_path)
    else:
        # LoRA / QLoRA: only adapter weights, main process writes.
        if accelerator.is_main_process:
            unwrapped.save_pretrained(save_path, safe_serialization=True)
            if save_processor is not None:
                save_processor.save_pretrained(save_path)

    # Optimizer / scheduler state — single rank dump (sharded optim under
    # FSDP would need accelerator.save_state; we keep the simple legacy
    # path for compatibility with the existing --resume code).
    if accelerator.is_main_process:
        try:
            torch.save({
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "global_step": global_step,
                "epoch": epoch,
                "batch_idx": batch_idx,
            }, os.path.join(save_path, "training_state.pt"))
        except Exception as e:
            # Under FSDP the optimizer state can be sharded — fall back to
            # accelerator.save_state so we get a proper distributed dump.
            print(f"[warn] direct optimizer.state_dict() failed ({e}); "
                  f"using accelerator.save_state instead")
    accelerator.wait_for_everyone()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--mini", action="store_true", help="Use mini dataset for testing")
    parser.add_argument("--epochs", type=int, default=None, help="Override num_epochs from config")
    parser.add_argument("--lr", type=float, default=None, help="Override learning_rate from config")
    parser.add_argument("--bs", type=int, default=None, help="Override batch_size from config")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--compress-method", type=str, default=None, help="Override compress_method")
    parser.add_argument("--compress-ratio", type=int, default=None, help="Override compress_ratio")
    parser.add_argument("--experiment", type=str, default=None, help="Override experiment name")
    parser.add_argument("--val-every", type=int, default=None, help="Validate every N opt steps")
    parser.add_argument("--val-batches", type=int, default=None, help="Number of val batches")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint dir (e.g. checkpoints_qwen25/crp_c8/checkpoint-66000)")
    parser.add_argument("--max-steps", type=int, default=None, help="Stop training after N optimizer steps")
    parser.add_argument("--save-every", type=int, default=None, help="Override save_every from config (smoke: pass huge value to skip ckpts)")
    parser.add_argument("--train-max-samples", type=int, default=None, help="Cap train dataset to first N samples (planning branch only)")
    parser.add_argument("--no-validate", action="store_true", help="Disable in-loop validation (smoke runs)")
    parser.add_argument("--no-final-save", action="store_true", help="Skip the post-training _save_model_and_state final dump (smoke runs)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Tier-2 VLA: load model + processor + dataset (1 sample), print "
                             "param counts and memory estimate, then exit. Does NOT train.")
    parser.add_argument("--train-mode", type=str, default=None,
                        choices=["lora", "full_sft", "qlora"],
                        help="Override train_mode from config. Defaults to 'lora'.")
    args = parser.parse_args()

    # ============ Distributed / Accelerator setup ============
    # Detect torchrun / `accelerate launch` environment. When RANK/WORLD_SIZE
    # are present we assume a multi-process distributed launch; the actual
    # FSDP wiring depends on `fsdp: true` in the YAML (parsed below).
    _is_distributed_env = ("RANK" in os.environ and "WORLD_SIZE" in os.environ
                           and int(os.environ.get("WORLD_SIZE", "1")) > 1)

    # ============ Load config ============
    cfg = load_config(args.config)
    if not _is_distributed_env or os.environ.get("RANK", "0") == "0":
        print(f"Config: {args.config}")

    model_id = cfg["model_id"]
    lora_r = cfg["lora_r"]
    lora_alpha = cfg["lora_alpha"]
    lora_dropout = cfg["lora_dropout"]
    lora_targets = cfg["lora_target_modules"]
    quantize = cfg.get("quantize", False)
    dtype_str = cfg.get("dtype", "bfloat16")
    compute_dtype = getattr(torch, dtype_str)
    lr = args.lr if args.lr is not None else float(cfg.get("learning_rate", cfg.get("lr", 2e-4)))
    batch_size = args.bs if args.bs is not None else cfg["batch_size"]
    grad_accum = cfg.get("grad_accum_steps", 1)
    num_epochs = args.epochs if args.epochs is not None else cfg.get("num_epochs", cfg.get("epochs", 1))
    max_length = cfg.get("max_length", 512)
    num_workers = cfg.get("num_workers", 0)
    save_every = args.save_every if args.save_every is not None else cfg.get("save_every", 500)
    keep_latest_k = cfg.get("keep_latest_k", 3)  # disk discipline; 0 disables pruning
    if args.train_max_samples is not None:
        cfg["train_max_samples"] = int(args.train_max_samples)
    if args.no_validate:
        cfg["val_every"] = 0
    min_pixels = cfg.get("min_pixels", 256 * 28 * 28)
    max_pixels = cfg.get("max_pixels", 512 * 28 * 28)

    # Video-mode settings (Tier-1 multi-frame video).
    # When video_mode=True, the dataset emits {"type":"video"} messages and we route
    # `max_pixels` into the **video processor** (per-frame). For a constant *total*
    # visual-token budget across an N-frame clip, the YAML should set
    # `max_pixels: <single-frame budget> // N` — documented in configs/gb200_video.yaml.
    video_mode = cfg.get("video_mode", False)
    num_frames = cfg.get("num_frames", 4)
    video_fps = cfg.get("video_fps", 2.0)
    data_path_video = cfg.get("data_path_video", None)

    # Tier-2 VLA settings
    train_mode = (args.train_mode or cfg.get("train_mode", "lora")).lower()
    if train_mode not in ("lora", "full_sft", "qlora"):
        print(f"ERROR: invalid train_mode={train_mode!r}", file=sys.stderr)
        sys.exit(2)
    vla_mode = cfg.get("vla_mode", False)
    vla_loss_mode = cfg.get("vla_loss_mode", "answer_and_traj")
    freeze_vision = cfg.get("freeze_vision", True)
    data_path_vla = cfg.get("data_path_vla", None)
    fsdp_enabled = cfg.get("fsdp", False)  # informational; launching FSDP is done via torchrun + accelerate
    activation_checkpointing = cfg.get("activation_checkpointing", cfg.get("gradient_checkpointing", False))
    # TODO(v2): visual token compression for video — needs a per-frame variant of
    # forward_with_compression (currently compress_visual_tokens only handles image
    # tokens via image_grid_thw). For Tier-1 we leave compress_method='none' in video
    # configs and only exercise the data + forward path.
    lora_target_modules_vision = cfg.get("lora_target_modules_vision", []) or []

    # Compression & experiment settings
    compress_method = args.compress_method or cfg.get("compress_method", "none")
    compress_ratio = args.compress_ratio or cfg.get("compress_ratio", 1)
    experiment = args.experiment or cfg.get("experiment", "default")
    val_every = 0 if args.no_validate else (args.val_every or cfg.get("val_every", 0))
    val_batches = args.val_batches or cfg.get("val_batches", 50)

    data_dir = os.path.join(_BASE_DIR, "data_processed")
    output_dir = os.path.join(_BASE_DIR, "checkpoints_qwen25", experiment)
    os.makedirs(output_dir, exist_ok=True)

    # ---- Build Accelerator (FSDP if requested, else default DDP/single-GPU) ----
    # When `fsdp_enabled` (YAML `fsdp: true`) AND we are inside a distributed
    # launch we construct a FullyShardedDataParallelPlugin and pass it to the
    # Accelerator. Otherwise we still create an Accelerator (so the training
    # loop has a single code path) but it stays in `no` distributed mode or
    # plain multi-GPU DDP depending on launch.
    accelerator = None
    use_fsdp = bool(fsdp_enabled) and _is_distributed_env
    if use_fsdp:
        from accelerate.utils import FullyShardedDataParallelPlugin
        from torch.distributed.fsdp import (
            MixedPrecision, BackwardPrefetch, ShardingStrategy,
        )
        # Try to locate the actual decoder-layer class to enable a
        # cls-name-based auto-wrap policy. Fall back to string-based
        # name (which the plugin also accepts) if the import fails.
        try:
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
                Qwen2_5_VLDecoderLayer,
            )
            transformer_cls_names = ["Qwen2_5_VLDecoderLayer"]
        except Exception as e:  # pragma: no cover
            print(f"[FSDP] Could not import Qwen2_5_VLDecoderLayer ({e}); "
                  f"falling back to string name only")
            transformer_cls_names = ["Qwen2_5_VLDecoderLayer"]

        # IMPORTANT: do NOT enable activation_checkpointing in the FSDP plugin
        # for transformers >= 5.x — `Qwen2_5_VLDecoderLayer` inherits from
        # `transformers.modeling_layers.GradientCheckpointingLayer` and already
        # handles AC internally when the model has gradient checkpointing on.
        # If we also wrap each layer with FSDP's `checkpoint_wrapper`, the
        # double-wrap causes a `CheckpointError: A different number of tensors
        # was saved during the original forward and recomputation` on backward.
        # Instead, leave AC off in the plugin and let HF's built-in
        # `model.gradient_checkpointing_enable()` (called below) do it.
        fsdp_plugin = FullyShardedDataParallelPlugin(
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            mixed_precision_policy=MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.bfloat16,
            ),
            transformer_cls_names_to_wrap=transformer_cls_names,
            use_orig_params=True,
            sync_module_states=True,
            cpu_ram_efficient_loading=True,
            forward_prefetch=False,
            activation_checkpointing=False,  # see comment above
            state_dict_type="SHARDED_STATE_DICT",
        )
        accelerator = Accelerator(fsdp_plugin=fsdp_plugin,
                                  gradient_accumulation_steps=grad_accum,
                                  mixed_precision="bf16")
    else:
        # Plain Accelerator: single-GPU or DDP. When fsdp YAML flag is set but
        # we were not launched via accelerate/torchrun, fall through here and
        # warn so the user sees that FSDP is silently off.
        if fsdp_enabled and not _is_distributed_env:
            print("[FSDP] WARNING: cfg.fsdp=true but no distributed launch detected; "
                  "running in single-process mode. Use `accelerate launch ...` to enable FSDP.")
        accelerator = Accelerator(gradient_accumulation_steps=grad_accum)

    eff_bs = batch_size * grad_accum
    # World-size aware global batch when distributed.
    world_size = (accelerator.num_processes if accelerator is not None else 1)
    global_bs = eff_bs * world_size

    accelerator.print(f"Experiment: {experiment}")
    accelerator.print(f"Model: {model_id} | dtype: {dtype_str} | quantize: {quantize}")
    accelerator.print(f"LoRA: r={lora_r} alpha={lora_alpha} dropout={lora_dropout}")
    accelerator.print(
        f"BS={batch_size} x accum={grad_accum} = eff_bs={eff_bs} per-rank | "
        f"world_size={world_size} -> global_bs={global_bs} | LR={lr} | max_len={max_length}"
    )
    accelerator.print(f"Image pixels: {min_pixels} ~ {max_pixels} | workers={num_workers}")
    accelerator.print(f"Compression: {compress_method} ratio={compress_ratio}")
    accelerator.print(f"Distributed: type={accelerator.distributed_type} "
                      f"num_processes={accelerator.num_processes} "
                      f"FSDP={'on' if use_fsdp else 'off'}")
    if val_every > 0:
        accelerator.print(f"Validation: every {val_every} opt steps, {val_batches} batches")

    # ============ Model setup ============
    print(f"Training mode: {train_mode}  |  vla_mode={vla_mode}  |  freeze_vision={freeze_vision}")
    # For dry-runs (and any time the configured path points to a non-existent
    # local checkpoint), fall back to a smaller available Qwen2.5-VL on disk.
    # Real training jobs that need the merged warm-init should set the correct
    # model_id explicitly — this only protects smoke / dry-run.
    if args.dry_run and model_id.startswith("/") and not os.path.exists(model_id):
        _FALLBACK_LOCAL = [
            "/workspace/models/Qwen2.5-VL-3B-drivelm-merged",
            "/workspace/models/Qwen2.5-VL-3B-Instruct",
            "/workspace/models/Qwen2.5-VL-7B-Instruct",
        ]
        for cand in _FALLBACK_LOCAL:
            if os.path.exists(cand):
                print(f"[dry-run] model_id {model_id!r} missing; using fall-back {cand}")
                model_id = cand
                break
    # Under FSDP we must load on CPU (or meta) so the plugin can shard;
    # `device_map="auto"` would pre-place weights on a single GPU and
    # collide with FSDP. For single-GPU / DDP keep the auto-placement path.
    if use_fsdp:
        load_kwargs = {}
        # cpu_ram_efficient_loading: only rank 0 holds the full weights, other
        # ranks load on meta-device — the plugin will broadcast at wrap time.
        if int(os.environ.get("RANK", "0")) != 0 and accelerator is not None:
            # rank>0 should load on meta to save host RAM. Use low_cpu_mem_usage
            # to defer materialization; the FSDP plugin's
            # `cpu_ram_efficient_loading` will sync_module_states from rank0.
            load_kwargs["low_cpu_mem_usage"] = True
    else:
        load_kwargs = {"device_map": "auto"}

    # qlora implies quantize-on-load even if YAML didn't set quantize:true
    if train_mode == "qlora":
        quantize = True
        cfg.setdefault("quant_bits", 4)

    if quantize:
        accelerator.print(f"Loading with {cfg.get('quant_bits', 4)}-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=(cfg.get("quant_bits", 4) == 4),
            load_in_8bit=(cfg.get("quant_bits", 4) == 8),
            bnb_4bit_use_double_quant=cfg.get("double_quant", True),
            bnb_4bit_quant_type=cfg.get("quant_type", "nf4"),
            bnb_4bit_compute_dtype=compute_dtype,
        )
        load_kwargs["quantization_config"] = bnb_config
    else:
        accelerator.print(f"Loading in {dtype_str} (no quantization)...")
        load_kwargs["torch_dtype"] = compute_dtype

    load_kwargs["attn_implementation"] = "sdpa"
    model = AutoModelForImageTextToText.from_pretrained(model_id, **load_kwargs)
    processor = AutoProcessor.from_pretrained(model_id)

    if hasattr(processor, "image_processor") and processor.image_processor is not None:
        processor.image_processor.min_pixels = min_pixels
        processor.image_processor.max_pixels = max_pixels
    # In video_mode, also push min/max pixels into the video processor — this caps
    # PER-FRAME resolution (the size dict's shortest/longest_edge in pixels). When the
    # YAML sets max_pixels = base_max // num_frames, the total visual token budget per
    # clip stays close to the single-image budget.
    if video_mode and hasattr(processor, "video_processor") and processor.video_processor is not None:
        vp = processor.video_processor
        # vp.size is a SizeDict dataclass (no __setitem__) — use setattr.
        if hasattr(vp, "size") and vp.size is not None:
            if hasattr(vp.size, "shortest_edge"):
                setattr(vp.size, "shortest_edge", min_pixels)
            if hasattr(vp.size, "longest_edge"):
                setattr(vp.size, "longest_edge", max_pixels)
        for attr, val in (("min_pixels", min_pixels), ("max_pixels", max_pixels)):
            if hasattr(vp, attr):
                setattr(vp, attr, val)
        print(f"Video processor caps: min_pixels={min_pixels} max_pixels={max_pixels} (per frame); size={vp.size}")

    # Prepare for training
    if quantize:
        model = prepare_model_for_kbit_training(model)
    else:
        # enable_input_require_grads installs a forward hook that attaches a
        # grad-requiring view to the embedding output. That is necessary for
        # LoRA/PEFT (so gradients flow into adapters through frozen embeddings)
        # but under FSDP + activation_checkpointing it injects a "phantom"
        # saved-tensor that does not reappear during the recompute pass and
        # triggers `CheckpointError: A different number of tensors was saved
        # during the original forward and recomputation`. For full_sft (no
        # frozen embeddings on the LLM side) we can safely skip it.
        _need_input_require_grads = train_mode != "full_sft"
        if _need_input_require_grads:
            model.enable_input_require_grads()

    # Gradient checkpointing: trade compute for memory.
    # Strategy: ALWAYS use HF-side gradient_checkpointing_enable (which routes
    # through `Qwen2_5_VLDecoderLayer`'s built-in `GradientCheckpointingLayer`
    # base). The FSDP plugin's `activation_checkpointing` is deliberately left
    # OFF (see fsdp_plugin construction above) to avoid double-wrap mismatches.
    # We enable gradient checkpointing whenever either YAML knob asks for it
    # OR we are running full_sft under FSDP (memory pressure on shards).
    _want_gc = (cfg.get("gradient_checkpointing", False)
                or activation_checkpointing
                or (use_fsdp and train_mode == "full_sft"))
    if _want_gc:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        accelerator.print("Gradient checkpointing enabled (HF-side, non-reentrant)")

    # Optionally extend LoRA targets to vision-tower modules. Off by default since
    # video Tier-1 starts with LLM-only LoRA; if a user wants to also adapt the
    # SigLIP/ViT side, set `lora_target_modules_vision: [...]` in the YAML (e.g.
    # ["qkv", "proj"] for Qwen2.5-VL's vision blocks).
    effective_lora_targets = list(lora_targets)
    if lora_target_modules_vision:
        for t in lora_target_modules_vision:
            if t not in effective_lora_targets:
                effective_lora_targets.append(t)
        print(f"Vision-tower LoRA targets added: {lora_target_modules_vision}")

    if train_mode == "full_sft":
        # Skip LoRA entirely; train all (non-frozen) parameters.
        if args.resume:
            accelerator.print(f"NOTE: --resume with train_mode=full_sft loads model weights from {args.resume}")
            # Caller is responsible for pointing model_id at the resume checkpoint or
            # using accelerate/torch.distributed checkpoint loading.
        # Freeze vision tower if requested.
        if freeze_vision:
            base_for_freeze = model
            visual = getattr(getattr(base_for_freeze, "model", base_for_freeze), "visual", None)
            if visual is None and hasattr(base_for_freeze, "visual"):
                visual = base_for_freeze.visual
            if visual is not None:
                frozen = 0
                for p in visual.parameters():
                    p.requires_grad = False
                    frozen += p.numel()
                accelerator.print(f"Froze vision tower: {frozen / 1e6:.1f} M params")
            else:
                accelerator.print("WARNING: train_mode=full_sft, freeze_vision=true but could not locate visual module")
        # Param accounting
        n_total = sum(p.numel() for p in model.parameters())
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        accelerator.print(f"Full SFT: {n_train/1e9:.3f} B trainable / {n_total/1e9:.3f} B total "
                          f"({100 * n_train / n_total:.2f}%)")
    else:
        # LoRA / QLoRA
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=effective_lora_targets,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        if args.resume:
            # Resume: load LoRA weights from checkpoint instead of init new
            resume_path = args.resume if os.path.isabs(args.resume) else os.path.join(_BASE_DIR, args.resume)
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, resume_path, is_trainable=True)
            accelerator.print(f"Resumed LoRA from {resume_path}")
        else:
            model = get_peft_model(model, lora_config)
        if accelerator.is_main_process:
            model.print_trainable_parameters()

    # Image token id for compression
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    video_token_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
    accelerator.print(f"Image token id: {image_token_id} | Video token id: {video_token_id}")

    # ============ Cross-frame video token compressor (planning VLA) ============
    # When `cross_frame_compressor` is set in cfg we build a compressor module
    # (scripts/compressors registry) and route the planning forward through
    # `forward_with_video_xframe_compression`. Compressor params are added to
    # the optimizer; most variants are zero-param but `temporal_pool` with
    # pool_type='weighted' or `vtm` with `use_learnable_key=True` do learn.
    xframe_compressor = None
    xframe_cfg = cfg.get("cross_frame_compressor", None)
    if xframe_cfg is not None:
        try:
            from scripts.compressors import make_compressor  # noqa: E402
        except ImportError:
            # When train_lora.py is invoked directly with cwd inside scripts/,
            # the package path resolves as `compressors` instead.
            from compressors import make_compressor  # type: ignore  # noqa: E402
        comp_name = xframe_cfg.get("name")
        comp_kwargs = xframe_cfg.get("kwargs", {}) or {}
        xframe_compressor = make_compressor(comp_name, **comp_kwargs)
        # Move to device with same dtype as LM weights.
        xframe_compressor = xframe_compressor.to(device=accelerator.device, dtype=compute_dtype)
        n_comp = sum(p.numel() for p in xframe_compressor.parameters())
        n_comp_train = sum(p.numel() for p in xframe_compressor.parameters() if p.requires_grad)
        accelerator.print(
            f"[xframe] Built compressor '{comp_name}' kwargs={comp_kwargs}: "
            f"{n_comp_train}/{n_comp} trainable params"
        )
        # NOTE on FSDP: the compressor sits OUTSIDE the FSDP wrap. For zero-param
        # variants (mean / last / cosine / norm-only) this is a non-issue. For
        # learnable variants (pool_type=weighted, vtm use_learnable_key, longvu
        # similarity_metric=learned) gradients are computed per-rank with no
        # implicit DDP all-reduce — caller must add an explicit reduce or wrap the
        # compressor in DDP/FSDP if it is to be trained under multi-GPU launch.
        if n_comp_train > 0 and _is_distributed_env:
            accelerator.print(
                f"[xframe] WARNING: compressor has {n_comp_train} trainable params under "
                f"distributed launch; cross-rank gradient sync is NOT wired. Set "
                f"learnable variants only after wiring DDP / FSDP wrap for the compressor."
            )

    # ============ Q-Former projector (Track A.1, redo) ============
    # When `projector_type: qformer` is set in cfg we build the BLIP-2-style
    # Q-Former projector (64 learnable queries cross-attending post-merger
    # visual features) and route the planning forward through
    # `forward_with_video_qformer_projector`. Projector params are added to
    # the optimizer like the pixelshuffle / resampler paths. Mutually
    # exclusive with `cross_frame_compressor` (different ablation axis:
    # fusion mechanism vs cross-frame temporal compression).
    qformer_projector = None
    projector_type = str(cfg.get("projector_type", "linear")).lower()
    if projector_type == "qformer":
        if xframe_compressor is not None:
            raise ValueError(
                "Cannot combine projector_type=qformer with "
                "cross_frame_compressor; these are different ablation axes "
                "(fusion mechanism vs cross-frame temporal compression). "
                "Pick one."
            )
        try:
            from scripts.qformer_projector_hf import (  # noqa: E402
                Qwen2VLQFormerProjector,
            )
        except ImportError:
            from qformer_projector_hf import (  # type: ignore  # noqa: E402
                Qwen2VLQFormerProjector,
            )
        qf_cfg = cfg.get("qformer", {}) or {}
        # Resolve LM hidden dim (= post-merger in_features for Qwen2.5-VL).
        try:
            lm_dim_default = int(model.config.text_config.hidden_size)
        except Exception:
            lm_dim_default = int(qf_cfg.get("lm_dim", 2048))
        # vit_dim: when running POST-merger (the default forward shim entry),
        # the Q-Former's KV input dim is the LM hidden size, since
        # get_video_features outputs (total_post_tokens, lm_dim).
        vit_dim_default = lm_dim_default
        qformer_projector = Qwen2VLQFormerProjector(
            vit_dim=int(qf_cfg.get("vit_dim", vit_dim_default)),
            internal_dim=int(qf_cfg.get("internal_dim", 1024)),
            lm_dim=int(qf_cfg.get("lm_dim", lm_dim_default)),
            num_queries=int(qf_cfg.get("num_queries", 64)),
            num_layers=int(qf_cfg.get("num_layers", 6)),
            n_heads=int(qf_cfg.get("n_heads", 8)),
            ffn_mult=int(qf_cfg.get("ffn_mult", 4)),
            layer_norm_eps=float(qf_cfg.get("layer_norm_eps", 1e-6)),
            dropout=float(qf_cfg.get("dropout", 0.0)),
        )
        qformer_projector = qformer_projector.to(
            device=accelerator.device, dtype=compute_dtype,
        )
        n_proj = sum(p.numel() for p in qformer_projector.parameters())
        n_proj_train = sum(p.numel() for p in qformer_projector.parameters() if p.requires_grad)
        accelerator.print(
            f"[qformer] Built projector "
            f"vit={qformer_projector.vit_dim} "
            f"lm={qformer_projector.lm_dim} "
            f"internal={qformer_projector.internal_dim} "
            f"queries={qformer_projector.num_queries} "
            f"layers={qformer_projector.num_layers}: "
            f"{n_proj_train}/{n_proj} trainable params "
            f"({n_proj/1e6:.2f}M)"
        )
        # Same FSDP caveat as pixelshuffle / resampler: projector sits OUTSIDE
        # the FSDP wrap so per-rank gradients are not cross-rank-reduced. OK
        # for single-node smoke; must be wired (DDP wrap / manual all-reduce)
        # before multi-node SFT.
        if n_proj_train > 0 and _is_distributed_env:
            accelerator.print(
                f"[qformer] WARNING: projector has {n_proj_train} "
                f"trainable params under distributed launch; cross-rank "
                f"gradient sync is NOT wired. Same caveat as pixelshuffle / "
                f"resampler."
            )

    gpu_mem = torch.cuda.memory_allocated() / 1024**3
    accelerator.print(f"GPU memory after model load (pre-shard): {gpu_mem:.2f} GB")

    # ============ Load CRP importance (if needed) ============
    global _CRP_IMPORTANCE
    if compress_method in ("crp", "crp_merge"):
        crp_path = cfg.get("crp_importance_path", os.path.join(_BASE_DIR, "precomputed", "crp_importance.pt"))
        if not os.path.isabs(crp_path):
            crp_path = os.path.join(_BASE_DIR, crp_path)
        if os.path.exists(crp_path):
            _CRP_IMPORTANCE = torch.load(crp_path, weights_only=True)
            accelerator.print(f"Loaded CRP importance for {len(_CRP_IMPORTANCE)} images")
        else:
            accelerator.print(f"[WARN] CRP importance not found at {crp_path}, falling back to L2 norm")

    # ============ Data setup ============
    # Resolve trajectory tokenizer (only if vla_mode) so we can wire boundary ids
    # into the dataset for traj-only loss masking.
    traj_start_id = traj_end_id = None
    if vla_mode:
        from trajectory_tokenizer import TrajectoryTokenizerConfig
        traj_cfg = TrajectoryTokenizerConfig()
        traj_start_id = traj_cfg.traj_start_id
        traj_end_id = traj_cfg.traj_end_id
        accelerator.print(f"VLA mode: traj_start_id={traj_start_id} traj_end_id={traj_end_id} "
                          f"loss={vla_loss_mode}")

    # ============ nuScenes-planning branch (Phase B) ============
    # If dataset_kind == 'nuscenes_planning', skip the JSON loader entirely
    # and build the dataset on the fly from UniAD's preprocessed temporal infos
    # + the CAM_FRONT samples under data/nuscenes/samples/CAM_FRONT/.
    dataset_kind = str(cfg.get("dataset_kind", "drivelm")).lower()
    use_planning = (dataset_kind == "nuscenes_planning")
    if use_planning:
        from planning_dataset import build_planning_dataset  # noqa: E402
        train_dataset = build_planning_dataset(cfg, processor, split="train")
        val_dataset = None
        if val_every > 0 and cfg.get("infos_val"):
            val_dataset = build_planning_dataset(cfg, processor, split="val")
        accelerator.print(
            f"[nusc-planning] train_samples={len(train_dataset)} "
            f"val_samples={len(val_dataset) if val_dataset is not None else 0}"
        )

    if vla_mode and not use_planning:
        v_path = data_path_vla or f"data_processed/v1_1_video_n{num_frames}_with_traj.json"
        if not os.path.isabs(v_path):
            v_path = os.path.join(_BASE_DIR, v_path)
        train_file = v_path
        val_file = v_path.replace(".json", "_val.json")
    elif video_mode and not use_planning:
        # Prefer explicit data_path_video; fall back to v1_1_video_n{N}.json
        v_path = data_path_video or f"data_processed/v1_1_video_n{num_frames}.json"
        if not os.path.isabs(v_path):
            v_path = os.path.join(_BASE_DIR, v_path)
        train_file = v_path
        val_file = v_path.replace(".json", "_val.json")  # convention; smoke uses same file
    elif not use_planning:
        default_train = "train_mini.json" if args.mini else "train.json"
        train_file = os.path.join(data_dir, cfg.get("train_file", default_train))
        val_file = os.path.join(data_dir, "val.json")
    if not use_planning:
        accelerator.print(f"Loading dataset: {train_file} (video_mode={video_mode}, num_frames={num_frames}, vla={vla_mode})")

        train_dataset = DriveLMDataset(
            train_file, processor, max_length=max_length,
            video_mode=video_mode, num_frames=num_frames, video_fps=video_fps,
            vla_mode=vla_mode, vla_loss_mode=vla_loss_mode,
            traj_start_id=traj_start_id, traj_end_id=traj_end_id,
        )

    if args.dry_run:
        # Sanity-print one sample + memory estimate; do NOT build optimizer / train.
        accelerator.print("\n========= DRY RUN =========")
        sample0 = train_dataset[0]
        accelerator.print(f"  dataset size      : {len(train_dataset)}")
        accelerator.print(f"  sample input_ids  : {tuple(sample0['input_ids'].shape)}")
        if "pixel_values_videos" in sample0:
            accelerator.print(f"  pixel_values_vids : {tuple(sample0['pixel_values_videos'].shape)}")
        if "video_grid_thw" in sample0:
            accelerator.print(f"  video_grid_thw    : {sample0['video_grid_thw'].tolist()}")
        n_action = sum(1 for tid in sample0['input_ids'].tolist()
                       if (traj_start_id is not None and tid >= traj_start_id - 256))
        accelerator.print(f"  ~action tokens    : {n_action}  (rough heuristic)")
        n_total = sum(p.numel() for p in model.parameters())
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # Memory estimate: bf16 weights = 2 bytes; AdamW (states m + v in fp32) = 8 bytes/trainable
        weight_gb = 2 * n_total / 1024**3
        grad_gb = 2 * n_train / 1024**3
        opt_gb = 8 * n_train / 1024**3
        accelerator.print(f"  params total      : {n_total/1e9:.3f} B  (~{weight_gb:.2f} GB bf16 weights)")
        accelerator.print(f"  params trainable  : {n_train/1e9:.3f} B  (~{grad_gb:.2f} GB grads, ~{opt_gb:.2f} GB Adam states)")
        accelerator.print(f"  rough train mem   : ~{weight_gb + grad_gb + opt_gb:.1f} GB (excl. activations / VLA)")
        accelerator.print(f"  fsdp flag         : {fsdp_enabled}  activation_ckpt={activation_checkpointing}  use_fsdp={use_fsdp}")
        accelerator.print(f"  accelerator       : type={accelerator.distributed_type} world={accelerator.num_processes}")
        accelerator.print("DRY RUN OK — exiting before optimizer construction.")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = None
    if use_planning:
        if val_every > 0 and val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=True,
            )
            accelerator.print(f"Validation samples: {len(val_dataset)}")
    elif val_every > 0 and os.path.exists(val_file):
        val_dataset = DriveLMDataset(
            val_file, processor, max_length=max_length,
            video_mode=video_mode, num_frames=num_frames, video_fps=video_fps,
            vla_mode=vla_mode, vla_loss_mode=vla_loss_mode,
            traj_start_id=traj_start_id, traj_end_id=traj_end_id,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        accelerator.print(f"Validation samples: {len(val_dataset)}")

    # World-size aware step count. Accelerator.prepare() will shard the
    # dataloader across ranks, so per-rank batches = unsharded_len / world.
    # We compute total_steps against the per-rank length so the scheduler's
    # cosine decay aligns with the actual optimizer.step() cadence (an opt
    # step fires every `grad_accum` *per-rank* micro-batches, regardless of
    # world size — gradients are all-reduced at the sync boundary).
    _unsharded_batches = len(train_loader)
    _per_rank_batches = max(1, _unsharded_batches // max(1, accelerator.num_processes))
    num_batches = _per_rank_batches
    total_steps = num_batches * num_epochs // grad_accum
    accelerator.print(f"Training samples: {len(train_dataset)}")
    accelerator.print(
        f"Batches (unsharded): {_unsharded_batches} | "
        f"per-rank batches: {num_batches} | "
        f"world_size: {accelerator.num_processes} | "
        f"Total opt steps: {total_steps}"
    )

    # ============ Optimizer ============
    # NB: Optimizer must be built AFTER FSDP wraps the model when use_orig_params=True;
    # but Accelerator.prepare() wraps the model first, then the optimizer — so we build
    # the optimizer here against the raw (CPU) params, and accelerator.prepare() will
    # rebind them after sharding. This is the supported path in accelerate>=1.x.
    _opt_params = list(model.parameters())
    if xframe_compressor is not None:
        _opt_params = _opt_params + list(xframe_compressor.parameters())
    if qformer_projector is not None:
        _opt_params = _opt_params + list(qformer_projector.parameters())
    optimizer = torch.optim.AdamW(_opt_params, lr=lr, weight_decay=0.01)

    lr_schedule = str(cfg.get("lr_schedule", "cosine"))
    if lr_schedule == "autovla_stepdecay":
        # AutoVLA recipe: linear warmup for N steps, then ×gamma every step_freq.
        # Replicates `LambdaLR` from ucla-mobility/AutoVLA tools/run_sft.py.
        autovla_warmup = int(cfg.get("warmup_steps", 500))
        autovla_step_freq = int(cfg.get("lr_step_freq", 2000))
        autovla_step_gamma = float(cfg.get("lr_step_gamma", 0.98))

        def _autovla_lr_lambda(current_step: int) -> float:
            if current_step < autovla_warmup:
                return float(current_step) / float(max(1, autovla_warmup))
            decay_count = (current_step - autovla_warmup) // autovla_step_freq
            return autovla_step_gamma ** decay_count

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_autovla_lr_lambda)
        accelerator.print(
            f"Scheduler: AutoVLA step-decay -> warmup={autovla_warmup} linear, "
            f"then x{autovla_step_gamma} every {autovla_step_freq} steps / {total_steps} total"
        )
    else:
        warmup_ratio = float(cfg.get("warmup_ratio", 0.05))
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=max(1, int(total_steps * warmup_ratio)),
            num_training_steps=total_steps,
        )
        accelerator.print(f"Scheduler: cosine with warmup_ratio={warmup_ratio:.3f} -> "
                          f"{int(total_steps * warmup_ratio)} warmup steps / {total_steps} total")

    # ============ Accelerator.prepare (FSDP sharding happens here) ============
    # NOTE: deliberately DO NOT pass `scheduler` to prepare(). accelerate's
    # AcceleratedScheduler wrapper multiplies scheduler.step() by
    # num_processes (world_size) per call — designed for DP semantics. With
    # our FSDP + manual sync_gradients-gated step, that "fast-forwards" the
    # cosine decay by 8×, causing lr → ~0 by step ~500 of 4106 (verified
    # 2026-05-18: lr=5.26e-10 at step 510). Keep scheduler vanilla; the loop
    # calls .step() exactly once per opt step (gated by `if accelerator.sync_gradients`).
    if val_loader is not None:
        model, optimizer, train_loader, val_loader = accelerator.prepare(
            model, optimizer, train_loader, val_loader,
        )
    else:
        model, optimizer, train_loader = accelerator.prepare(
            model, optimizer, train_loader,
        )
    if use_fsdp and freeze_vision and train_mode == "full_sft":
        # Re-apply the vision freeze AFTER FSDP wrap, because FSDP's flat-param
        # layout (with use_orig_params=True) can re-attach requires_grad to params
        # during wrap. Walk the unwrapped module tree and zero the flag again.
        _unwrapped = accelerator.unwrap_model(model)
        _visual = getattr(getattr(_unwrapped, "model", _unwrapped), "visual", None)
        if _visual is None and hasattr(_unwrapped, "visual"):
            _visual = _unwrapped.visual
        if _visual is not None:
            _frozen = 0
            for p in _visual.parameters():
                if p.requires_grad:
                    p.requires_grad = False
                _frozen += p.numel()
            accelerator.print(f"[post-prepare] re-froze vision tower: {_frozen / 1e6:.1f} M params")

    if torch.cuda.is_available():
        post_shard_gb = torch.cuda.memory_allocated() / 1024**3
        accelerator.print(f"GPU memory after FSDP prepare: {post_shard_gb:.2f} GB / rank")

    # ============ Resume training state ============
    resume_step = 0
    resume_epoch = 0
    if args.resume:
        state_path = os.path.join(
            args.resume if os.path.isabs(args.resume) else os.path.join(_BASE_DIR, args.resume),
            "training_state.pt"
        )
        if os.path.exists(state_path):
            state = torch.load(state_path, weights_only=True)
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            resume_step = state["global_step"]
            resume_epoch = state.get("epoch", 0)
            accelerator.print(f"Resumed optimizer/scheduler from step {resume_step}, epoch {resume_epoch}")
        else:
            accelerator.print(f"[WARN] No training_state.pt found, resuming LoRA weights only (optimizer reset)")

    # ============ Optional wandb ============
    if args.wandb and accelerator.is_main_process:
        import wandb
        wandb.init(project="drivelm-qwen25vl", name=experiment, config={
            **cfg, "mini": args.mini, "lr": lr, "batch_size": batch_size,
            "compress_method": compress_method, "compress_ratio": compress_ratio,
        })

    # ============ Training loop ============
    accelerator.print(f"\n{'='*60}")
    accelerator.print(f"  Starting training | {experiment} | {num_epochs} epoch(s) | {total_steps} opt steps")
    accelerator.print(f"  Compression: {compress_method} ratio={compress_ratio}")
    accelerator.print(f"{'='*60}\n")
    model.train()
    global_step = resume_step
    accum_loss = 0.0
    skip_batches = resume_step * grad_accum if resume_step > 0 else 0
    _device = accelerator.device

    # Sliding-window loss (per user 2026-05-19): show mean of last N batch losses
    # instead of cumulative epoch average. Epoch-average smears initial-spike
    # losses across all later steps and hides recent dynamics; the window shows
    # the loss the optimizer is currently seeing.
    loss_window_size = int(cfg.get("loss_window", 100))
    batch_loss_window: deque = deque(maxlen=loss_window_size)

    for epoch in range(resume_epoch, num_epochs):
        epoch_loss_sum = 0.0
        epoch_loss_count = 0

        # Only render the progress bar on the main process to avoid 8x stdout spam.
        pbar = tqdm(
            enumerate(train_loader),
            total=num_batches,
            desc=f"Epoch {epoch+1}/{num_epochs}",
            bar_format="{l_bar}{bar:30}{r_bar}",
            dynamic_ncols=True,
            disable=not accelerator.is_main_process,
        )

        for step, batch in pbar:
            # Skip already-trained batches on resume
            if skip_batches > 0:
                skip_batches -= 1
                if skip_batches % 1000 == 0 and accelerator.is_main_process:
                    pbar.set_postfix_str(f"skipping... {skip_batches} left")
                continue

            # Accelerate's prepared dataloader already places tensors on the
            # right device, but our custom collate may produce extra keys
            # (image_names, second_per_grid_ts) — sweep .to(device) defensively.
            batch = {k: v.to(_device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            try:
                with accelerator.accumulate(model):
                    if qformer_projector is not None:
                        outputs = forward_with_video_qformer_projector(
                            model, batch, qformer_projector, video_token_id,
                        )
                    elif xframe_compressor is not None:
                        outputs = forward_with_video_xframe_compression(
                            model, batch, xframe_compressor, video_token_id,
                            int(cfg.get("planning_num_past_frames", 4)),
                        )
                    else:
                        outputs = forward_with_compression(
                            model, batch, compress_method, compress_ratio, image_token_id
                        )
                    loss = outputs.loss
                    batch_loss = loss.detach().float().item()

                    # NaN guard: skip bad batches before they poison the model
                    if not math.isfinite(batch_loss):
                        if accelerator.is_main_process:
                            tqdm.write(f"[NaN] batch {step+1}/{num_batches}, loss={batch_loss}, skipping")
                        optimizer.zero_grad(set_to_none=True)
                        accum_loss = 0.0
                        del outputs, loss
                        torch.cuda.empty_cache()
                        continue

                    accelerator.backward(loss)
                    accum_loss += batch_loss / grad_accum
                    epoch_loss_sum += batch_loss
                    epoch_loss_count += 1
                    batch_loss_window.append(batch_loss)

                    if accelerator.sync_gradients:
                        # Gradient clipping (FSDP-aware).
                        grad_norm = accelerator.clip_grad_norm_(model.parameters(), 1.0)
                        # grad_norm may be a tensor returned by accelerate; coerce to float
                        try:
                            _gn = float(grad_norm)
                        except Exception:
                            _gn = float('nan')
                        if not math.isfinite(_gn):
                            if accelerator.is_main_process:
                                tqdm.write(f"[NaN grad] step {global_step}, grad_norm={_gn}, skipping update")
                            optimizer.zero_grad(set_to_none=True)
                            accum_loss = 0.0
                            del outputs, loss
                            continue
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                        global_step += 1
                        _did_step = True
                    else:
                        _did_step = False

                    del outputs, loss
            except RuntimeError as e:
                if "out of memory" in str(e):
                    if accelerator.is_main_process:
                        tqdm.write(f"[OOM] batch {step+1}/{num_batches}, skipping")
                    for _v in ("outputs", "loss", "batch"):
                        try:
                            del locals()[_v]
                        except (KeyError, NameError):
                            pass
                    optimizer.zero_grad(set_to_none=True)
                    accum_loss = 0.0
                    torch.cuda.empty_cache()
                    continue
                raise

            # Update tqdm postfix every batch (main process only)
            if accelerator.is_main_process:
                # Sliding-window mean over last `loss_window_size` batches.
                window_loss = sum(batch_loss_window) / max(len(batch_loss_window), 1)
                gpu_mem = torch.cuda.memory_allocated() / 1024**3
                cur_lr = scheduler.get_last_lr()[0] if global_step > 0 else lr
                pbar.set_postfix_str(
                    f"batch_loss={batch_loss:.4f} | loss(w{loss_window_size})={window_loss:.4f} | "
                    f"lr={cur_lr:.2e} | opt_step={global_step}/{total_steps} | "
                    f"GPU={gpu_mem:.1f}GB"
                )

            if _did_step:
                if args.wandb and accelerator.is_main_process:
                    import wandb
                    cur_lr = scheduler.get_last_lr()[0]
                    wandb.log({
                        "loss": accum_loss, "batch_loss": batch_loss,
                        "loss_window": sum(batch_loss_window) / max(len(batch_loss_window), 1),
                        "avg_loss": epoch_loss_sum / max(epoch_loss_count, 1),
                        "lr": cur_lr, "gpu_mem": gpu_mem,
                    }, step=global_step)
                accum_loss = 0.0

                if global_step % save_every == 0:
                    save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                    _save_model_and_state(
                        accelerator, model, optimizer, scheduler,
                        train_mode, save_path, global_step, epoch, step,
                    )
                    if accelerator.is_main_process:
                        tqdm.write(f"  [SAVE] checkpoint-{global_step}")
                        # Disk discipline: keep only the latest K ckpts.
                        # At save_every=100 and ckpt~18GB this caps disk use
                        # to ~K * 18GB instead of 41 * 18GB = 738GB.
                        if keep_latest_k > 0:
                            import glob, shutil
                            ckpts = sorted(
                                glob.glob(os.path.join(output_dir, "checkpoint-*")),
                                key=lambda p: int(p.rsplit("-", 1)[-1]),
                            )
                            for old in ckpts[:-keep_latest_k]:
                                tqdm.write(f"  [PRUNE] removing {os.path.basename(old)}")
                                shutil.rmtree(old, ignore_errors=True)
                    accelerator.wait_for_everyone()

                # Validation
                if val_every > 0 and val_loader is not None and global_step % val_every == 0:
                    val_loss, val_acc, l2_dict = validate(
                        model, val_loader, compress_method, compress_ratio,
                        image_token_id, val_batches, _device,
                        xframe_compressor=xframe_compressor,
                        video_token_id=video_token_id,
                        num_past_frames=int(cfg.get("planning_num_past_frames", 4)),
                        qformer_projector=qformer_projector,
                    )
                    if accelerator.is_main_process:
                        base = f"  [VAL] step={global_step} val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
                        if qformer_projector is not None:
                            tqdm.write(f"{base} L2=skipped(qformer)")
                        elif xframe_compressor is not None:
                            tqdm.write(f"{base} L2=skipped(xframe)")
                        elif l2_dict:
                            tqdm.write(
                                f"{base} L2 avg={l2_dict['L2_avg']:.3f} "
                                f"(1s={l2_dict['L2_1s']:.3f} "
                                f"2s={l2_dict['L2_2s']:.3f} "
                                f"3s={l2_dict['L2_3s']:.3f})"
                            )
                        else:
                            tqdm.write(f"{base} L2=skipped(no_traj_tokens)")
                    if args.wandb and accelerator.is_main_process:
                        import wandb
                        log_payload = {"val_loss": val_loss, "val_acc": val_acc}
                        if l2_dict:
                            log_payload.update({
                                "val_l2_avg": l2_dict["L2_avg"],
                                "val_l2_1s": l2_dict["L2_1s"],
                                "val_l2_2s": l2_dict["L2_2s"],
                                "val_l2_3s": l2_dict["L2_3s"],
                            })
                        wandb.log(log_payload, step=global_step)

                if args.max_steps and global_step >= args.max_steps:
                    if accelerator.is_main_process:
                        tqdm.write(f"  [STOP] Reached max_steps={args.max_steps}")
                    pbar.close()
                    break

        pbar.close()
        avg_loss = epoch_loss_sum / max(epoch_loss_count, 1)
        cur_lr = scheduler.get_last_lr()[0]
        accelerator.print(
            f"\n  Epoch {epoch+1} done | "
            f"avg_loss={avg_loss:.4f} | LR={cur_lr:.2e} | "
            f"opt_steps={global_step}/{total_steps}\n"
        )
        if args.max_steps and global_step >= args.max_steps:
            break

    # ============ Save final model ============
    if args.no_final_save:
        accelerator.print(f"\nTraining complete! (--no-final-save -> skipping final dump)")
    else:
        final_path = os.path.join(output_dir, "final")
        _save_model_and_state(
            accelerator, model, optimizer, scheduler,
            train_mode, final_path, global_step, num_epochs - 1, -1,
            save_processor=processor,
        )
        accelerator.print(f"\nTraining complete! Final model saved to {final_path}")

    if args.wandb and accelerator.is_main_process:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
