# Agent A — torchtitan config_registry for Qwen3-VL-8B (dense) nuScenes
# planning SFT.
#
# Acts as both:
#   (a) a `config_registry` module (consumed by torchtitan's
#       ConfigManager._load_config via `--module scripts_titan.train_titan_qwen3_vl`
#       and `--config qwen3_vl_8b_planning_fsdp`), AND
#   (b) a thin programmatic entry point — `import` the factory directly
#       from tests / smoke scripts to get a fully-built Trainer.Config.
#
# CPU-only NOTE: this module only constructs configs.  All device-heavy
# work (FSDP shard, model init, dataloader spin-up) is deferred to the
# torchtitan Trainer at runtime.  No GPU operations happen at import time.
#
# Pivot history (2026-05-20):
#   The previous iteration tried to subclass `Qwen3VLModel` to imitate
#   Qwen2.5-VL (RMSNorm vs LayerNorm, gated MLP, windowed attention) and
#   port HF Qwen2.5-VL weights via a lossy state-dict adapter.  This was
#   abandoned because (a) the vision tower port is unavoidably lossy and
#   (b) torchtitan now ships a numerically-verified native Qwen3-VL-8B
#   (see torchtitan_qwen25/torchtitan/models/qwen3_vl/README.md: end-to-end
#   KL ~5e-8..5e-5 vs HF Transformers).  We now use the upstream native
#   spec straight from `Qwen/Qwen3-VL-8B-Instruct`.  See
#   docs/2026-05-20_vision_cp_open_problem.md for why CP stays at 1.
#
# Documented smoke command (DO NOT run from here — Agent A is CPU-only;
# another agent runs it later):
#
#   /usr/bin/python3 -m torchtitan.train \
#       --module scripts_titan.train_titan_qwen3_vl \
#       --config qwen3_vl_8b_planning_fsdp \
#       --training.steps 4 \
#       --checkpoint.no-enable

from __future__ import annotations

import os

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.components.tokenizer import MultiModalTokenizer
from torchtitan.config import (
    ActivationCheckpointConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.hf_datasets import DatasetConfig
from torchtitan.hf_datasets.multimodal.mm_datasets import MM_DATASETS, MMDataLoader
from torchtitan.models.qwen3_vl import QWEN3_VL_SPECIAL_TOKENS, model_registry
from torchtitan.trainer import Trainer


# ============================================================================
# Constants
# ============================================================================

# Verified 2026-05-20 against the HF model page (Qwen/Qwen3-VL-8B-Instruct).
QWEN3_VL_8B_HF_ID = "Qwen/Qwen3-VL-8B-Instruct"

# Image normalization for the Qwen3-VL vision tower.  Qwen3-VL switched from
# OpenAI-CLIP mean/std (Qwen2.5-VL) to (0.5, 0.5, 0.5) / (0.5, 0.5, 0.5);
# this matches torchtitan's qwen3_vl.config_registry._qwen3_vl_dataloader.
QWEN3_VL_IMAGE_MEAN = (0.5, 0.5, 0.5)
QWEN3_VL_IMAGE_STD = (0.5, 0.5, 0.5)

# Vision-token geometry for the dataloader's patcher.  Qwen3-VL uses
# patch_size=16 (Qwen2.5-VL used 14); temporal_patch_size and
# spatial_merge_size are unchanged.
QWEN3_VL_PATCH_SIZE = 16
QWEN3_VL_TEMPORAL_PATCH_SIZE = 2
QWEN3_VL_SPATIAL_MERGE_SIZE = 2


# ============================================================================
# Custom dataset registration
# ============================================================================
#
# torchtitan's MMDataLoader requires the dataset to be registered in the
# module-level MM_DATASETS dict.  Each entry pairs a `loader(path) ->
# IterableDataset` with a per-sample `sample_processor`.  We expose
# nuScenes planning under a per-recipe name (e.g. `nuscenes_planning_3cam_4f`)
# and use a passthrough processor (our dataset already emits torchtitan-format
# samples).
#
# Lazy AutoProcessor resolution: the HF processor is needed for chat-
# template / tokenization but is too heavy to instantiate at config-build
# time.  The loader resolves it on first call using QWEN3_VL_8B_HF_ID
# (configurable via the override dict's "hf_id" key).

_NUSCENES_DATASET_OVERRIDES: dict[str, dict] = {}


def _resolve_processor(hf_id: str):
    """Return an HF AutoProcessor for Qwen3-VL-8B.

    Uses `trust_remote_code=True` because Qwen3-VL ships its custom
    processor class in its model repo.  We deliberately do NOT cache the
    processor here — the dataloader caches it via the dataset instance.
    """
    from transformers import AutoProcessor  # local import

    return AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)


def _nuscenes_loader(path: str, **kwargs):
    """Construct the NuScenesPlanningDatasetTitan.

    Called by torchtitan's HuggingFaceMultiModalDataset.__init__ as
    `dataset_loader(path)`.  We pull the keyword overrides from the
    module-level registry, resolve the HF processor lazily, and instantiate
    the IterableDataset.
    """
    from .nuscenes_planning_dataset_titan import NuScenesPlanningDatasetTitan

    overrides = _NUSCENES_DATASET_OVERRIDES.get(path)
    if not overrides:
        raise RuntimeError(
            f"NuScenesPlanningDatasetTitan loader: no overrides registered "
            f"for dataset key {path!r}. Did you call the qwen3_vl_8b_* "
            f"config factory first?"
        )

    # Lazy processor resolution.
    hf_id = overrides.get("hf_id", QWEN3_VL_8B_HF_ID)
    processor = overrides.get("processor")
    if processor is None:
        processor = _resolve_processor(hf_id)

    # Build the dataset, dropping our own bookkeeping keys.
    ds_kwargs = {k: v for k, v in overrides.items() if k not in ("hf_id",)}
    ds_kwargs["processor"] = processor
    return NuScenesPlanningDatasetTitan(**ds_kwargs)


def _nuscenes_passthrough_processor(sample, **kwargs):
    """Identity processor — our dataset already emits torchtitan-format
    samples, so the per-sample processor just returns them as-is."""
    return sample


def _register_nuscenes_dataset(name: str, overrides: dict) -> None:
    """Register our dataset under `name` in torchtitan's MM_DATASETS."""
    _NUSCENES_DATASET_OVERRIDES[name] = overrides
    MM_DATASETS[name] = DatasetConfig(
        path=name,  # used as the lookup key in our loader
        loader=_nuscenes_loader,
        sample_processor=_nuscenes_passthrough_processor,
    )


def _qwen3_vl_dataloader(dataset: str, **kwargs) -> MMDataLoader.Config:
    """Dataloader config tuned for Qwen3-VL patch geometry.

    Mirrors torchtitan/models/qwen3_vl/config_registry::_qwen3_vl_dataloader
    but with a generous `max_images_per_batch` to fit 3-cam x 4-frame
    videos, and packing disabled (video samples are too large to pack).
    """
    return MMDataLoader.Config(
        dataset=dataset,
        max_images_per_batch=128,
        patch_size=QWEN3_VL_PATCH_SIZE,
        temporal_patch_size=QWEN3_VL_TEMPORAL_PATCH_SIZE,
        spatial_merge_size=QWEN3_VL_SPATIAL_MERGE_SIZE,
        # Qwen3-VL HF processor default min/max pixels (see Qwen3-VL repo).
        min_pixels=65536,
        max_pixels=16777216,
        image_mean=QWEN3_VL_IMAGE_MEAN,
        image_std=QWEN3_VL_IMAGE_STD,
        # No HF-style packing: nuScenes 3-cam videos are too large to pack.
        packing_buffer_size=0,
        **kwargs,
    )


# ============================================================================
# Recipe constants (paper-cited; see SATS_VLM_impl_plan_v2.md + MEMORY.md
# "no-lazy-shortcuts" / "warmup-by-tokens-not-steps").
# ============================================================================
#
# AutoVLA recipe at 4-8 frame video horizon: LR 2e-5, AdamW, step-decay
# x0.98 every lr_step_freq fraction of total steps, warmup 1.74% of total
# steps.  We retain the 3-epoch sweep across ~26k nuScenes planning
# samples; with global batch 8 over 8 ranks this gives ~26k * 3 / 8 ≈
# 9750 steps, rounded to 10000.

_TOTAL_STEPS = 10000
_WARMUP_RATIO = 0.0174        # AutoVLA: 1.74% of total
_LR_STEP_FREQ_RATIO = 0.0696  # AutoVLA: 6.96% of total (step-decay cadence)


def _warmup_steps(total: int = _TOTAL_STEPS) -> int:
    return max(1, int(round(total * _WARMUP_RATIO)))


def _lr_step_freq(total: int = _TOTAL_STEPS) -> int:
    return max(1, int(round(total * _LR_STEP_FREQ_RATIO)))


# ============================================================================
# Common Trainer.Config builder (FSDP baseline + TP variant)
# ============================================================================


def _common_overrides() -> dict:
    """Return the nuScenes dataset override dict (paths + recipe knobs)."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return {
        # HF assets — same canonical path as torchtitan's stock 8B config.
        "hf_id": QWEN3_VL_8B_HF_ID,
        "infos_path": os.path.join(
            repo_root, "data_processed", "nuscenes_infos_temporal_train.pkl"
        ),
        "nusc_root": os.path.join(repo_root, "data", "nuscenes"),
        # Processor is resolved lazily inside _nuscenes_loader().
        "processor": None,
        "max_length": 4096,
        "num_past_frames": 4,
        "num_future_waypoints": 6,
        "video_fps": 2.0,
        "vla_loss_mode": "answer_and_traj",
        "require_full_future": True,
        "require_all_cams": True,
        "infinite": True,
        # Use Qwen3-VL mean/std for the dataset normalization (matches the
        # dataloader's patcher and the vision tower's expected stats).
        "image_mean": QWEN3_VL_IMAGE_MEAN,
        "image_std": QWEN3_VL_IMAGE_STD,
    }


def _build_trainer_config(
    *,
    dataset_name: str,
    planning_cams: list[str],
    local_batch_size: int,
    parallelism: ParallelismConfig,
    seq_len: int = 4096,
    total_steps: int = _TOTAL_STEPS,
    max_length: int | None = None,
) -> Trainer.Config:
    """Shared Trainer.Config builder used by all variants.

    Per-variant differences are passed in via `parallelism`, `dataset_name`,
    `planning_cams`, `seq_len`, and `max_length`; the rest of the recipe
    is identical.  `max_length` controls per-sample token cap inside the
    dataset (used to right-size truncation); `seq_len` is torchtitan's
    batch sequence length (collator pads to this).  These should agree.
    """
    overrides = _common_overrides()
    overrides["planning_cams"] = planning_cams
    if max_length is not None:
        overrides["max_length"] = int(max_length)
    _register_nuscenes_dataset(dataset_name, overrides)

    return Trainer.Config(
        # HF assets path: torchtitan's CheckpointManager reads
        # `<hf_assets_path>/tokenizer/` for the tokenizer at startup, and
        # the Qwen3VLStateDictAdapter consumes safetensors from the same
        # tree to load the pretrained Qwen3-VL-8B weights.  Mirrors
        # torchtitan's stock qwen3_vl_8b config.
        hf_assets_path="./assets/hf/Qwen3-VL-8B-Instruct",
        tokenizer=MultiModalTokenizer.Config(**QWEN3_VL_SPECIAL_TOKENS),
        metrics=MetricsProcessor.Config(log_freq=10),
        # *** Native upstream Qwen3-VL-8B spec — no subclass, no lossy
        # adapter.  model_registry("8B") returns the fully-built ModelSpec
        # with parallelize_qwen3_vl + pipeline_qwen3_vl +
        # Qwen3VLStateDictAdapter wired up. ***
        model_spec=model_registry("8B"),
        dataloader=_qwen3_vl_dataloader(dataset_name),
        optimizer=OptimizersContainer.Config(
            # AutoVLA recipe for 4-8 frame video horizon.
            lr=2e-5,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            # Warmup is RATIO-scaled to total steps per MEMORY.md
            # "warmup-by-tokens-not-steps".
            warmup_steps=_warmup_steps(total_steps),
            decay_ratio=1.0,
            # AutoVLA uses step-decay x0.98 every _lr_step_freq() steps;
            # torchtitan's LRSchedulersContainer doesn't expose step-decay
            # directly, so we approximate with linear decay over the full
            # schedule.  If torchtitan adds step-decay support, swap to it
            # here (and pass _lr_step_freq() as the cadence).
            decay_type="linear",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=local_batch_size,
            seq_len=seq_len,
            steps=total_steps,
            mixed_precision_param="bfloat16",
            mixed_precision_reduce="float32",
        ),
        parallelism=parallelism,
        checkpoint=CheckpointManager.Config(
            enable=True,
            interval=1000,
            last_save_model_only=False,
            export_dtype="bfloat16",
            keep_latest_k=2,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            # 8B model + 3-cam x 4-frame video on 80GB H100/A100 -> full AC
            # is required to fit per-rank memory.  See
            # docs/2026-05-20_PLAN_long_video_vla_and_5D_parallelism.md.
            mode="full",
        ),
    )


# ============================================================================
# Factories registered with ConfigManager (`--config <name>`)
# ============================================================================


def qwen3_vl_8b_planning_fsdp() -> Trainer.Config:
    """FSDP-only 8-rank baseline, 3-cam x 4-frame nuScenes planning.

    8x H100/A100 FSDP shard (`data_parallel_shard_degree=-1` -> all
    ranks sharded), no TP / PP / CP.  Global batch = 8 (local 1 x 8 DP).

    CP stays at 1 — vision-attached CP is an open problem (see
    docs/2026-05-20_vision_cp_open_problem.md).
    PP stays at 1 — Agent B's PP wiring is functional but its scheduler
    interactions with the vision-scatter forward are still under
    pressure-testing.
    """
    return _build_trainer_config(
        dataset_name="nuscenes_planning_3cam_4f",
        planning_cams=["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"],
        local_batch_size=1,
        seq_len=4096,
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=-1,  # All ranks FSDP-shard.
            tensor_parallel_degree=1,
            context_parallel_degree=1,
            pipeline_parallel_degree=1,
        ),
    )


def qwen3_vl_8b_planning_fsdp_1cam_8f() -> Trainer.Config:
    """1-cam x 8-frame variant for direct comparison with the 3-cam x 4f
    baseline (same total visual tokens, different temporal/spatial split)."""
    return _build_trainer_config(
        dataset_name="nuscenes_planning_1cam_8f",
        planning_cams=["CAM_FRONT"],
        local_batch_size=1,
        seq_len=4096,
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=-1,
            tensor_parallel_degree=1,
            context_parallel_degree=1,
            pipeline_parallel_degree=1,
        ),
    )


def qwen3_vl_8b_planning_fsdp_tp() -> Trainer.Config:
    """FSDP=4 x TP=2 2D parallelism (8-rank mesh).

    Showcases torchtitan's 2D parallelism: shard params/grads/optim with
    FSDP across 4 groups while column/row-paralleling the LM attention
    and FFN across 2 ranks per group.  Vision encoder is also TP'd by
    parallelize_qwen3_vl (per README: 'TP applied to both vision encoder
    and decoder, without SequenceParallel due to vision scatter and
    DeepStack').

    Net per-rank memory drops because attention/FFN weights are split
    across 2 ranks, leaving more headroom for activations.  Trades
    1 extra all-reduce per attn/FFN block for that headroom.
    """
    return _build_trainer_config(
        dataset_name="nuscenes_planning_3cam_4f_tp",
        planning_cams=["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"],
        local_batch_size=1,
        seq_len=4096,
        parallelism=ParallelismConfig(
            # 8 ranks total = 4 FSDP-shard groups x 2 TP ranks/group.
            data_parallel_shard_degree=4,
            tensor_parallel_degree=2,
            context_parallel_degree=1,
            pipeline_parallel_degree=1,
        ),
    )


def qwen3_vl_8b_planning_fsdp_3cam() -> Trainer.Config:
    """FSDP-only 8-rank 3-cam x 4-frame nuScenes planning with generous
    seq_len headroom (max_length=8192).

    Visual context: CAM_FRONT + CAM_FRONT_LEFT + CAM_FRONT_RIGHT, 4 frames
    each @ 2 Hz.  AutoVLA-aligned forward-arc set.  ``require_all_cams=True``
    drops samples that are missing any of the three cam files on disk
    (the streaming-extract path for FL/FR is the gating factor; see
    scripts/stream_extract_nuscenes_3cam.sh).

    Token budget (per-sample):
      Each cam emits ~140 visual tokens / frame after Qwen3-VL's 2x2
      spatial merger + 2-frame temporal merger.  3 cams * 4 frames * 140 =
      ~1680 visual tokens; +chat-template / ego-speed preamble / multi-cam
      labels / 14-token trajectory block / assistant wrapper => ~2.0-2.3k
      tokens.  We pick max_length=seq_len=8192 to leave generous headroom
      for per-frame token-count drift, future longer prompts, and any
      vision-tower stride variation between Qwen3-VL builds.

    Hyperparams: identical to qwen3_vl_8b_planning_fsdp (LR 2e-5, AutoVLA
    warmup 1.74%, step-decay-via-linear @ 6.96% cadence, 10k steps).
    The only deltas vs that factory are dataset_name and seq_len/max_length.

    Parallelism: FSDP=8 mesh (no TP, no PP, no CP) — start simplest.  CP
    stays at 1 (vision-attached CP open problem); PP stays at 1 (Agent B
    pressure-test ongoing).
    """
    return _build_trainer_config(
        dataset_name="nuscenes_planning_3cam_4f_seqlen8k",
        planning_cams=["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"],
        local_batch_size=1,
        seq_len=8192,
        max_length=8192,
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=-1,  # All ranks FSDP-shard.
            tensor_parallel_degree=1,
            context_parallel_degree=1,
            pipeline_parallel_degree=1,
        ),
    )


# ============================================================================
# CPU-only smoke (config build only — no model materialization).
# ============================================================================


if __name__ == "__main__":
    for name, fn in [
        ("qwen3_vl_8b_planning_fsdp", qwen3_vl_8b_planning_fsdp),
        ("qwen3_vl_8b_planning_fsdp_1cam_8f", qwen3_vl_8b_planning_fsdp_1cam_8f),
        ("qwen3_vl_8b_planning_fsdp_3cam", qwen3_vl_8b_planning_fsdp_3cam),
        ("qwen3_vl_8b_planning_fsdp_tp", qwen3_vl_8b_planning_fsdp_tp),
    ]:
        cfg = fn()
        print(f"=== {name} ===")
        print("  Model spec:", cfg.model_spec.name, cfg.model_spec.flavor)
        print("  LR:", cfg.optimizer.lr)
        print("  Warmup steps:", cfg.lr_scheduler.warmup_steps)
        print("  Total steps:", cfg.training.steps)
        print("  Local BS:", cfg.training.local_batch_size)
        print("  Seq len:", cfg.training.seq_len)
        print("  FSDP shard degree:", cfg.parallelism.data_parallel_shard_degree)
        print("  TP degree:", cfg.parallelism.tensor_parallel_degree)
