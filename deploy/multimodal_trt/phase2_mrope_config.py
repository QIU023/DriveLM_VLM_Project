"""Build mrope_config in the EXACT format TRT-LLM 1.3.0rc15's
``Qwen3VLModelBase.prepare_mrope_config`` consumes per request.

Strategy: delegate the M-RoPE computation to Hugging Face's canonical
``Qwen3VLModel.get_rope_index`` (the same routine ``model.forward`` runs
internally to feed position_ids to the LM). We bind the unbound method
against a tiny stub object so we don't have to instantiate the 8B model.

Reference functions (called, NOT reimplemented):

- ``transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLModel.get_rope_index``
  ``/usr/local/lib/python3.12/dist-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:1033``
  Returns ``(position_ids[3, B, L] int64, mrope_position_deltas[B, 1])``.
- ``transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLModel.get_vision_position_ids``
  same file, line 975. Called by ``get_rope_index`` for each image/video span.

TRT consumer (verified at ``/venv/trt_llm/lib/python3.12/site-packages/tensorrt_llm/_torch/models/modeling_qwen3vl.py:1056-1101``):
- Reads ``mrope_config["mrope_position_ids"]`` — uses ``shape[-1]`` to slice
  into ``mrope_position_ids_padding_cuda`` (preallocated ``(3, 1, max_pos)``
  int32 CUDA buffer at line 1045). So we must produce shape ``(3, 1, seq_len)``
  and the dtype should be int32 to match the destination buffer (and TRT's
  own input-processor output at line 348 also does ``.to(torch.int32)``).
- Reads ``mrope_config["mrope_position_deltas"]`` — concatenated along dim 0
  across requests at line 1099, so shape ``(1, 1)`` int32 per request.
"""

from __future__ import annotations

import sys
import types
from typing import Any, Dict

import torch
from transformers import AutoConfig, AutoProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel


def _make_stub(model_config: Any) -> Any:
    """Build the minimal duck-typed object that ``Qwen3VLModel.get_rope_index``
    expects: it only accesses ``self.config.vision_config.spatial_merge_size``
    and calls ``self.get_vision_position_ids(...)``.
    """
    stub = types.SimpleNamespace()
    stub.config = model_config
    # Bind the unbound vision-position helper to the stub so the ``self``
    # call inside ``get_rope_index`` resolves correctly.
    stub.get_vision_position_ids = types.MethodType(
        Qwen3VLModel.get_vision_position_ids, stub
    )
    return stub


def build_mrope_config(
    *,
    model_config: Any,
    input_ids: torch.Tensor,                 # (seq_len,) or (1, seq_len)
    mm_token_type_ids: torch.Tensor,         # (seq_len,) or (1, seq_len); 0=text 1=img 2=vid
    image_grid_thw: torch.Tensor | None,     # (n_img, 3) or (3,) — pre-merger grid
    video_grid_thw: torch.Tensor | None,     # (n_vid, 3) or (3,) — pre-merger grid
    attention_mask: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    """Produce ``{"mrope_position_ids": (3,1,L) int32 cpu,
                  "mrope_position_deltas": (1,1) int32 cpu}``.

    Inputs match the keys returned by ``MultiModalPlanningDataset.__getitem__``.
    Grids must be the *pre-merger* values from the HF/TRT processor (e.g.
    ``[1, 22, 22]`` for an image that yields 121 LM tokens after merger=2).
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if mm_token_type_ids.dim() == 1:
        mm_token_type_ids = mm_token_type_ids.unsqueeze(0)
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    elif attention_mask.dim() == 1:
        attention_mask = attention_mask.unsqueeze(0)

    def _ensure_2d(g):
        if g is None:
            return None
        g = g.to(dtype=torch.long)
        if g.dim() == 1:
            g = g.unsqueeze(0)
        return g

    image_grid_thw = _ensure_2d(image_grid_thw)
    video_grid_thw = _ensure_2d(video_grid_thw)

    stub = _make_stub(model_config)
    # Call HF's get_rope_index against the stub. We do NOT instantiate the
    # full Qwen3VLModel (~8B params) — get_rope_index only uses
    # self.config.vision_config.spatial_merge_size and
    # self.get_vision_position_ids(), both satisfied by the stub.
    position_ids, mrope_position_deltas = Qwen3VLModel.get_rope_index(
        stub,
        input_ids=input_ids.to(dtype=torch.long),
        mm_token_type_ids=mm_token_type_ids.to(dtype=torch.long),
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        attention_mask=attention_mask.to(dtype=torch.long),
    )

    return {
        "mrope_position_ids": position_ids.to("cpu", torch.int32).contiguous().clone(),
        "mrope_position_deltas": mrope_position_deltas.to("cpu", torch.int32).contiguous().clone(),
    }


# ---------------------------------------------------------------------------
# Smoke on a real B.5'' val sample.
# ---------------------------------------------------------------------------

CKPT = "/workspace/DriveLM_VLM_Project/checkpoints_qwen25/nusc_planning_b5pp_1cam_qwen3vl_multimodal/final"


def _smoke() -> None:
    sys.path.insert(0, "/workspace/DriveLM_VLM_Project/scripts")
    from multimodal_planning_dataset import MultiModalPlanningDataset  # noqa: E402

    proc = AutoProcessor.from_pretrained(CKPT)
    ds = MultiModalPlanningDataset(
        infos_path="/workspace/DriveLM_VLM_Project/data/uniad_infos/nuscenes_infos_temporal_val.pkl",
        nusc_root="/workspace/DriveLM_VLM_Project/data/nuscenes",
        processor=proc,
        max_length=12288,
        num_past_frames=4,
        num_future_waypoints=6,
        video_fps=2.0,
        vla_loss_mode="answer_and_traj",
        max_samples=2,
        require_full_future=True,
        planning_cams=["CAM_FRONT"],
        require_all_cams=True,
        hdmap_dir="/workspace/DriveLM_VLM_Project/data/preproc/hdmap_bev",
        bbox_jsonl="/workspace/DriveLM_VLM_Project/data/preproc/bbox_egostate_val.jsonl",
        split="val",
        modality_dropout_p=0.0,
    )
    sample = ds[0]
    input_ids = sample["input_ids"]
    image_grid_thw = sample.get("image_grid_thw")
    video_grid_thw = sample.get("video_grid_thw")
    mm_token_type_ids = sample["mm_token_type_ids"]

    print("[sample] input_ids        :", tuple(input_ids.shape), input_ids.dtype)
    print("[sample] image_grid_thw   :", None if image_grid_thw is None else image_grid_thw.tolist())
    print("[sample] video_grid_thw   :", None if video_grid_thw is None else video_grid_thw.tolist())
    print("[sample] mm_token_type    :", tuple(mm_token_type_ids.shape), mm_token_type_ids.dtype)
    types_ = mm_token_type_ids
    n_text = int((types_ == 0).sum())
    n_img = int((types_ == 1).sum())
    n_vid = int((types_ == 2).sum())
    print(f"[sample] type counts      : text={n_text} image={n_img} video={n_vid}")
    # find modality boundaries
    diffs = (types_[1:] != types_[:-1]).nonzero(as_tuple=True)[0].tolist()
    print(f"[sample] modality bounds  : changes at idx {diffs[:12]}")

    cfg = AutoConfig.from_pretrained(CKPT)
    print("[cfg]    spatial_merge    :", cfg.vision_config.spatial_merge_size)
    print("[cfg]    image_token_id   :", cfg.image_token_id)
    print("[cfg]    video_token_id   :", cfg.video_token_id)

    mrope_config = build_mrope_config(
        model_config=cfg,
        input_ids=input_ids,
        mm_token_type_ids=mm_token_type_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        attention_mask=None,
    )

    pids = mrope_config["mrope_position_ids"]
    deltas = mrope_config["mrope_position_deltas"]
    print()
    print("=== mrope_config ===")
    print("mrope_position_ids    shape:", tuple(pids.shape), "dtype:", pids.dtype, "device:", pids.device)
    print("mrope_position_deltas shape:", tuple(deltas.shape), "dtype:", deltas.dtype, "device:", deltas.device)
    print()
    print("mrope_position_ids[:, 0, :5] (first 5):")
    print("  T:", pids[0, 0, :5].tolist())
    print("  H:", pids[1, 0, :5].tolist())
    print("  W:", pids[2, 0, :5].tolist())
    print("mrope_position_ids[:, 0, -5:] (last 5):")
    print("  T:", pids[0, 0, -5:].tolist())
    print("  H:", pids[1, 0, -5:].tolist())
    print("  W:", pids[2, 0, -5:].tolist())
    print("Around video block (idx 8..16):")
    print("  T:", pids[0, 0, 8:16].tolist())
    print("  H:", pids[1, 0, 8:16].tolist())
    print("  W:", pids[2, 0, 8:16].tolist())
    print("Around image block (idx 54..62):")
    print("  T:", pids[0, 0, 54:62].tolist())
    print("  H:", pids[1, 0, 54:62].tolist())
    print("  W:", pids[2, 0, 54:62].tolist())
    print("mrope_position_deltas:", deltas.tolist())


if __name__ == "__main__":
    _smoke()
