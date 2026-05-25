"""Shared helpers for B.5''' TRT deploy scripts.

Centralises:
  - venv import shims (FLASHINFER_DISABLE_VERSION_CHECK + nvidia-cuda-tileiras
    metadata.files() workaround) — must run BEFORE `import tensorrt_llm`
  - default ckpt / dataset paths
  - calibration loader (256 train samples through MultiModalPlanningDataset)
  - quantization summary printer wrapper
  - HF model loader with bf16

Usage:
    from _common import apply_venv_shims; apply_venv_shims()  # FIRST
    from _common import (
        DEFAULT_CKPT, DEFAULT_CONFIG_YAML, build_calib_dataset, ...
    )
"""
from __future__ import annotations

import importlib.metadata as _md
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_BASE = _HERE.parent.parent  # /workspace/DriveLM_VLM_Project


# ---------------------------------------------------------------------------
# Defaults — all callers default to B.5''' ckpt
# ---------------------------------------------------------------------------

# B.5'' v2 = 1-cam Qwen3-VL-4B multimodal (native, not 3-cam). 3-cam ckpt
# (nusc_planning_b5ppp_3cam_qwen3vl_multimodal) is the prior B.5''' baseline;
# this deploy script targets the 1-cam v2 successor.
DEFAULT_CKPT = str(
    _BASE / "checkpoints_qwen25/nusc_planning_b5pp_1cam_qwen3vl_multimodal/final"
)
DEFAULT_PARENT = str(_BASE / "checkpoints_qwen25/nusc_planning_b5pp_1cam_qwen3vl_multimodal")
DEFAULT_CONFIG_YAML = str(
    _BASE / "configs/nuscenes_planning_1cam_qwen3vl_multimodal.yaml"
)
DEFAULT_BENCH_OUT_DIR = str(_BASE / "deploy/trt_bench")
DEFAULT_ENGINE_OUT_DIR = str(_BASE / "deploy/trt_b5ppp/engines")

# Vision payload shape for B.5'' v2 (1-cam + HD-map @ Qwen3-VL patch=16).
# Per the 1-cam multimodal config (configs/nuscenes_planning_1cam_qwen3vl_multimodal.yaml):
#   video: 1 camera × T_grid=2 × 56 × 100 -> 1 × 2800 = 2800 video tokens
#         (native 1600x900 / 16 -> ~57x100 patches; post 2x2 merger 2*(57//2)*(100//2)=2800)
#   image: 1 HD-map @ 224x224 upscaled to min_pixels=109760 cap -> 22x22 -> 121 image tokens
#   total visual = 2921 LM tokens (vs 3-cam B.5''' = 8464)
EXPECTED_VIDEO_TOKENS = 2800
EXPECTED_IMAGE_TOKENS = 121
EXPECTED_TOTAL_VISUAL = EXPECTED_VIDEO_TOKENS + EXPECTED_IMAGE_TOKENS

# Qwen3-VL native video pass-through default (no video_max_pixels cap in cfg)
# = 25_165_824 px = 4096*6144. If a cfg sets video_max_pixels we assert that;
# otherwise we assert the HF default holds.
HF_QWEN3VL_VIDEO_DEFAULT_LONGEST_EDGE = 25_165_824


# ---------------------------------------------------------------------------
# venv shims — MUST run before `import tensorrt_llm`
# ---------------------------------------------------------------------------

def apply_venv_shims() -> None:
    """Apply the two workarounds the TRT-LLM 1.3.0rc15 venv needs on RTX 5090.

    1. FLASHINFER_DISABLE_VERSION_CHECK=1 — flashinfer's runtime check rejects
       sm_120, so we disable it (the kernels still load via the JIT path).
    2. shim importlib.metadata.files() for nvidia-cuda-tileiras — TRT-LLM
       unconditionally calls files() on this package even though the .dist-info
       is missing in this venv. Returning None makes the lookup silently skip.
    """
    os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
    _orig_files = _md.files

    def _files_shim(name):
        try:
            return _orig_files(name)
        except _md.PackageNotFoundError:
            if "tileiras" in name:
                return None
            raise

    _md.files = _files_shim


# ---------------------------------------------------------------------------
# Path bootstrapping for DriveLM_VLM_Project scripts/
# ---------------------------------------------------------------------------

def add_project_paths() -> None:
    """Add DriveLM_VLM_Project/scripts and deploy/multimodal_trt to sys.path."""
    sys.path.insert(0, str(_BASE / "scripts"))
    sys.path.insert(0, str(_BASE / "deploy/multimodal_trt"))


def project_base() -> Path:
    return _BASE


# ---------------------------------------------------------------------------
# Calibration dataset — 256 train samples (same loader as training)
# ---------------------------------------------------------------------------

def build_calib_dataset(
    *,
    processor,
    n_samples: int = 256,
    split: str = "train",
    config_yaml: str = DEFAULT_CONFIG_YAML,
):
    """Construct MultiModalPlanningDataset capped at `n_samples` for PTQ calib.

    Mirrors the training config (3-cam, 4 past frames, HD-map, bbox, ego state,
    max_length=12288, video_fps=2.0, full-future requirement). Reads the same
    yaml so any future change to the SFT recipe propagates here automatically.
    """
    import yaml

    add_project_paths()
    from multimodal_planning_dataset import MultiModalPlanningDataset

    with open(config_yaml) as f:
        cfg = yaml.safe_load(f)

    base = project_base()
    # Resolve paths relative to project root (yaml uses repo-relative strings)
    def _resolve(p):
        if not p:
            return p
        return p if os.path.isabs(p) else str(base / p)

    hdmap_dir = _resolve(cfg.get("hdmap_dir", "data/preproc/hdmap_bev"))
    bbox_template = cfg.get("bbox_jsonl", "data/preproc/bbox_egostate_{split}.jsonl")
    bbox_jsonl = _resolve(bbox_template.format(split=split))
    infos_path = _resolve(
        cfg.get(f"infos_{split}", f"data/uniad_infos/nuscenes_infos_temporal_{split}.pkl")
    )
    nusc_root = _resolve(cfg.get("nusc_root", "data/nuscenes"))

    planning_cams = cfg.get("planning_cams", ["CAM_FRONT"])
    if isinstance(planning_cams, str):
        planning_cams = [planning_cams]

    ds = MultiModalPlanningDataset(
        infos_path=infos_path,
        nusc_root=nusc_root,
        processor=processor,
        max_length=int(cfg.get("max_length", 12288)),
        num_past_frames=int(cfg.get("planning_num_past_frames", 4)),
        num_future_waypoints=int(cfg.get("planning_num_future_wp", 6)),
        video_fps=float(cfg.get("video_fps", 2.0)),
        vla_loss_mode=cfg.get("vla_loss_mode", "answer_and_traj"),
        max_samples=int(n_samples),
        require_full_future=bool(cfg.get("planning_require_full_future", True)),
        planning_cams=planning_cams,
        require_all_cams=bool(cfg.get("planning_require_all_cams", True)),
        hdmap_dir=hdmap_dir,
        bbox_jsonl=bbox_jsonl,
        split=split,
        modality_dropout_p=0.0,  # ALWAYS 0 for calib + eval; only train uses dropout
    )

    # -- F2 GATE: processor caps actually match what the cfg yaml declares. --
    # Catches silent failures where the wrong preprocessor_config landed in the
    # ckpt dir (e.g. wrong max_pixels -> wrong HD-map token count -> mm-disagg
    # block-count mismatch at runtime).
    cfg_max_pixels = cfg.get("max_pixels")
    if cfg_max_pixels is not None and hasattr(processor, "image_processor"):
        # transformers 5.x: Qwen2VL/Qwen3VL image processors no longer expose a
        # scalar `.max_pixels`; the cap lives in `.size` (a SizeDict, key
        # `longest_edge`). Mirror the video branch below. Fall back to the old
        # `.max_pixels` attr for pre-5.x compat.
        ip = processor.image_processor
        actual_max_px = int(getattr(ip, "max_pixels", 0) or 0)
        if actual_max_px == 0:
            try:
                actual_max_px = int(ip.size.get("longest_edge"))
            except Exception:
                actual_max_px = int(getattr(getattr(ip, "size", None), "longest_edge", 0) or 0)
        assert actual_max_px == int(cfg_max_pixels), (
            f"[F2 GATE] processor.image_processor.max_pixels={actual_max_px} "
            f"!= cfg max_pixels={cfg_max_pixels}. The preprocessor_config in the "
            f"ckpt does not match {config_yaml}. Will produce wrong HD-map token "
            f"count and break mm-disagg block math. Fix by re-copying the correct "
            f"preprocessor_config.json into the ckpt dir."
        )

    # Qwen3-VL has a SEPARATE video_processor; assert it matches cfg (if set)
    # or holds the HF native pass-through default (~25M longest_edge).
    cfg_video_max_pixels = cfg.get("video_max_pixels")
    if hasattr(processor, "video_processor"):
        vp = processor.video_processor
        try:
            actual_video_longest = int(vp.size.get("longest_edge"))
        except Exception:
            actual_video_longest = int(getattr(vp.size, "longest_edge", 0))
        if cfg_video_max_pixels is not None:
            assert actual_video_longest == int(cfg_video_max_pixels), (
                f"[F2 GATE] processor.video_processor.size.longest_edge="
                f"{actual_video_longest} != cfg video_max_pixels="
                f"{cfg_video_max_pixels}. Will compress video to wrong resolution "
                f"and break per-cam token count."
            )
        else:
            # Expect HF Qwen3-VL native pass-through default
            assert actual_video_longest == HF_QWEN3VL_VIDEO_DEFAULT_LONGEST_EDGE, (
                f"[F2 GATE] cfg does not set video_max_pixels and "
                f"processor.video_processor.size.longest_edge={actual_video_longest} "
                f"!= HF Qwen3-VL default {HF_QWEN3VL_VIDEO_DEFAULT_LONGEST_EDGE}. "
                f"Native video pass-through assumption violated; per-cam token "
                f"count will not match EXPECTED_VIDEO_TOKENS={EXPECTED_VIDEO_TOKENS}."
            )
    return ds


# ---------------------------------------------------------------------------
# Tokenizer / processor copy helper — quantized save_pretrained skips these
# ---------------------------------------------------------------------------

def copy_processor_and_tokenizer(src_ckpt: str, dst_dir: str) -> None:
    """Copy tokenizer + processor config from src ckpt to dst (after quant save).

    modelopt's save_pretrained writes the quantized model weights + config but
    does NOT copy tokenizer.json / processor_config.json / chat_template.jinja
    etc. Those are needed for downstream inference, so we copy them verbatim.
    """
    import shutil

    src = Path(src_ckpt)
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)

    # Tokenizer + processor + special tokens + chat template — copy if present
    candidates = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        "chat_template.json",
        "chat_template.jinja",
        "processor_config.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "generation_config.json",
    ]
    copied = []
    for name in candidates:
        s = src / name
        if s.is_file():
            shutil.copy2(s, dst / name)
            copied.append(name)
    print(f"[copy] copied {len(copied)} processor/tokenizer files: {copied}")


# ---------------------------------------------------------------------------
# HF model loader (single GPU bf16, vision tower stays bf16)
# ---------------------------------------------------------------------------

def load_hf_model_bf16(ckpt: str, device: str = "cuda:0"):
    """Load Qwen3VLForConditionalGeneration in bf16. Returns (model, processor)."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"[load] loading HF model from {ckpt} (bf16, sdpa) ...")
    model = AutoModelForImageTextToText.from_pretrained(
        ckpt, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(ckpt)
    print(f"[load] OK on {device}")
    return model, processor


# ---------------------------------------------------------------------------
# CUDA mem helper
# ---------------------------------------------------------------------------

def reset_peak_mem():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def peak_mem_gb() -> float:
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / (1024 ** 3)
    return 0.0
