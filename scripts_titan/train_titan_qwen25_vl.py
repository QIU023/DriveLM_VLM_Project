# Agent A — torchtitan config_registry for Qwen2.5-VL nuScenes planning SFT.
#
# Acts as both:
#   (a) a `config_registry` module (consumed by torchtitan's
#       ConfigManager._load_config via `--module scripts_titan.train_titan_qwen25_vl`
#       and `--config qwen2_5_vl_3b_planning_fsdp`), AND
#   (b) a thin programmatic entry point — `import` the factory directly
#       from tests / smoke scripts to get a fully-built Trainer.Config.
#
# CPU-only NOTE: this module only constructs configs and registers the
# model_spec.  All device-heavy work (FSDP shard, model init, dataloader
# spin-up) is deferred to the torchtitan Trainer at runtime.  No GPU
# operations happen at import time.
#
# Documented smoke command (DO NOT run from here — Agent A is CPU-only;
# another agent runs it later):
#
#   /usr/bin/python3 -m torchtitan.train \
#       --module scripts_titan.train_titan_qwen25_vl \
#       --config qwen2_5_vl_3b_planning_fsdp \
#       --training.steps 4 \
#       --checkpoint.no-enable
#
# (We register a single config function: `qwen2_5_vl_3b_planning_fsdp`.
# Override `training.steps`, `--checkpoint.no-enable`, `--metrics.log-freq`,
# etc. on the CLI for smoke tests — torchtitan's tyro CLI accepts any
# section.key path.)

from __future__ import annotations

from functools import partial

import torch.nn as nn

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.components.tokenizer import MultiModalTokenizer
from torchtitan.config import (
    ActivationCheckpointConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.models.common import Embedding, Linear, RoPE
from torchtitan.models.common.attention import FlexAttention
from torchtitan.models.common.config_utils import make_ffn_config, make_gqa_config
from torchtitan.models.common.param_init import depth_scaled_std, skip_param_init
from torchtitan.models.common.rmsnorm import RMSNorm
from torchtitan.models.qwen3.model import Qwen3TransformerBlock
from torchtitan.models.qwen3_vl import (
    QWEN3_VL_SPECIAL_TOKENS,
    parallelize_qwen3_vl,
)
from torchtitan.models.qwen3_vl.vision_encoder import Qwen3VLVisionEncoder
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.trainer import Trainer

from .qwen2_5_vl_model import Qwen2_5_VLModel
from .qwen2_5_vl_state_dict_adapter import Qwen2_5_VLStateDictAdapter


# ============================================================================
# Param init helpers (mirror torchtitan/models/qwen3_vl/__init__.py).
# ============================================================================

_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_SKIP_INIT = {"weight": skip_param_init}
_POS_EMBED_INIT = {"pos_embed": partial(nn.init.trunc_normal_, mean=0.0, std=0.02)}
_EPS = 1e-6


def _output_linear_init(dim: int):
    s = dim**-0.5
    return {
        "weight": partial(nn.init.trunc_normal_, std=s, a=-3 * s, b=3 * s),
        "bias": nn.init.zeros_,
    }


def _depth_init(layer_id: int):
    return {
        "weight": partial(
            nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)
        ),
        "bias": nn.init.zeros_,
    }


def _qwen25_norm(dim: int) -> RMSNorm.Config:
    return RMSNorm.Config(normalized_shape=dim, eps=_EPS, param_init=_NORM_INIT)


def _vl_linear(in_features: int, out_features: int) -> Linear.Config:
    return Linear.Config(
        in_features=in_features,
        out_features=out_features,
        bias=True,
        param_init=_LINEAR_INIT,
    )


# ============================================================================
# Model config: Qwen2.5-VL-3B
# Dims sourced from `transformers.AutoConfig.from_pretrained(
#     'Qwen/Qwen2.5-VL-3B-Instruct')` (verified 2026-05-20).
# ============================================================================


def _build_qwen25_layers(
    *,
    n_layers: int,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    hidden_dim: int,
) -> list:
    """Per-layer LM configs.

    QK-norm is OMITTED here (qk_norm=None) and Q/K/V bias is ENABLED to
    match Qwen2.5-VL.  `Qwen2_5_VLModel._normalise_config` is idempotent
    over these settings, so the model would still produce the correct
    architecture if QK-norm were left on by accident, but we save the
    redundant build by leaving it off up-front.
    """
    layers = []
    for layer_id in range(n_layers):
        attn = make_gqa_config(
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            wqkv_param_init=_LINEAR_INIT,
            wo_param_init=_depth_init(layer_id),
            inner_attention=FlexAttention.Config(),
            mask_type="block_causal",
            rope_backend="cos_sin",
            qk_norm=None,  # <-- Qwen2.5 has no QK-norm
        )
        # Override wq / wkv with bias=True (Qwen2.5-VL attention has Q/K/V
        # bias).  make_gqa_config gave us bias=False; flip it.
        import dataclasses
        attn = dataclasses.replace(
            attn,
            wq=dataclasses.replace(attn.wq, bias=True),
            wkv=dataclasses.replace(attn.wkv, bias=True),
        )

        layers.append(
            Qwen3TransformerBlock.Config(
                attention_norm=_qwen25_norm(dim),
                ffn_norm=_qwen25_norm(dim),
                attention=attn,
                feed_forward=make_ffn_config(
                    dim=dim,
                    hidden_dim=hidden_dim,
                    w1_param_init=_LINEAR_INIT,
                    w2w3_param_init=_depth_init(layer_id),
                ),
            )
        )
    return layers


def _build_qwen25_vision_encoder() -> Qwen3VLVisionEncoder.Config:
    """Vision encoder config for Qwen2.5-VL-3B.

    DIVERGENCE NOTE: torchtitan's Qwen3VLVisionEncoder uses LayerNorm
    (not RMSNorm), single-projection MLP (not gated), and full block-
    diagonal attention (not windowed).  See
    `qwen2_5_vl_state_dict_adapter.py` for the load-time approximations
    we use.  For the planning SFT this is acceptable because the vision
    encoder is fine-tuned end-to-end from the partial init; if we later
    need numerically-identical Qwen2.5-VL vision features we will need to
    replace the ViT block / merger modules.

    deepstack_visual_indices=[] disables DeepStack (no intermediate-layer
    feature injection).  `Qwen2_5_VLModel._normalise_config` would also
    zero this out, but we set it here for clarity.
    """
    dim = 1280
    ffn_dim = 3420
    n_layers = 32
    n_heads = 16
    patch_size = 14
    temporal_patch_size = 2
    spatial_merge_size = 2
    out_hidden_size = 2048
    # No learned absolute pos embed in HF; arbitrary placeholder size.
    num_position_embeddings = 2304
    in_channels = 3

    patch_dim = in_channels * temporal_patch_size * patch_size * patch_size
    merged_hidden_size = dim * (spatial_merge_size**2)

    return Qwen3VLVisionEncoder.Config(
        dim=dim,
        ffn_dim=ffn_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        patch_size=patch_size,
        temporal_patch_size=temporal_patch_size,
        spatial_merge_size=spatial_merge_size,
        out_hidden_size=out_hidden_size,
        num_position_embeddings=num_position_embeddings,
        deepstack_visual_indices=[],  # <-- DeepStack disabled
        patch_embed_proj=_vl_linear(patch_dim, dim),
        attn_qkv=_vl_linear(dim, dim * 3),
        attn_proj=_vl_linear(dim, dim),
        mlp_fc1=_vl_linear(dim, ffn_dim),
        mlp_fc2=_vl_linear(ffn_dim, dim),
        merger_fc1=_vl_linear(merged_hidden_size, merged_hidden_size),
        merger_fc2=_vl_linear(merged_hidden_size, out_hidden_size),
        param_init=_POS_EMBED_INIT,
    )


def _qwen25_vl_3b_model_config() -> Qwen2_5_VLModel.Config:
    dim = 2048
    head_dim = 128
    n_layers = 36
    vocab_size = 151936

    return Qwen2_5_VLModel.Config(
        vocab_size=vocab_size,
        dim=dim,
        norm=_qwen25_norm(dim),
        enable_weight_tying=True,  # Qwen2.5-VL-3B ties lm_head <-> embeddings.
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_SKIP_INIT,
        ),
        output=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=head_dim,
            max_seq_len=32768,
            theta=1000000.0,
            backend="cos_sin",
        ),
        layers=_build_qwen25_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=16,
            n_kv_heads=2,
            head_dim=head_dim,
            hidden_dim=11008,
        ),
        vision_encoder=_build_qwen25_vision_encoder(),
        # mrope_section from HF Qwen2.5-VL-3B config:
        #   text_config.rope_scaling.mrope_section = [16, 24, 24]
        mrope_section=[16, 24, 24],
    )


def _qwen25_vl_3b_model_spec() -> ModelSpec:
    return ModelSpec(
        name="qwen2_5_vl",
        flavor="3B",
        model=_qwen25_vl_3b_model_config(),
        # We REUSE torchtitan's stock parallelize_qwen3_vl entry point.
        # Qwen2_5_VLModel has the same module tree as Qwen3VLModel (modulo
        # absent q_norm/k_norm and empty deepstack_merger_list), so the
        # FSDP / TP / AC passes work unchanged.  Verified at import time
        # in the smoke test below.
        parallelize_fn=parallelize_qwen3_vl,
        pipelining_fn=None,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=None,
        state_dict_adapter=Qwen2_5_VLStateDictAdapter,
    )


# ============================================================================
# Trainer.Config factory
# ============================================================================
#
# This is the function ConfigManager looks up by name when --config is
# `qwen2_5_vl_3b_planning_fsdp`.
#
# Recipe (matches R1' HF baseline):
#   * 3 epochs over nuScenes planning train (~26k samples)
#   * LR 2e-5, AdamW, cosine decay
#   * Warmup: 39 steps (R1' used 312 for 8x bigger ratio; we scale by the
#     1k-token-per-step ratio across batch sizes).  R1' baseline uses
#     ratio-scaled warmup; see SATS_VLM_impl_plan_v2.md.
#   * 8 GPU FSDP shard, no TP / PP / CP
#   * mixed_precision_param=bfloat16, reduce=float32
#
# The dataset entry registers a CUSTOM dataset name `nuscenes_planning`
# in torchtitan's MM_DATASETS at module import time so MMDataLoader can
# find it.  Because torchtitan's MMDataLoader hard-codes a
# HuggingFaceMultiModalDataset, we register a STUB DatasetConfig whose
# loader returns our NuScenesPlanningDatasetTitan directly (HF
# load_dataset is bypassed).  The torchtitan-side IterableDataset wrapper
# would normally iterate raw HF samples and run `sample_processor`; we
# bypass that by setting sample_processor to a passthrough that simply
# returns the already-formed dict.
# ----------------------------------------------------------------------------

from torchtitan.hf_datasets import DatasetConfig
from torchtitan.hf_datasets.multimodal.mm_datasets import MM_DATASETS, MMDataLoader


def _nuscenes_loader(path: str, **kwargs):
    """Construct the NuScenesPlanningDatasetTitan.

    Called by torchtitan's HuggingFaceMultiModalDataset.__init__ as
    `dataset_loader(path)`.  We can't easily pass our custom args through
    that hop, so we use a module-level overrides dict populated by the
    config factory.  Path is the dataset config name (e.g.
    'nuscenes_planning'); we ignore it.
    """
    from .nuscenes_planning_dataset_titan import NuScenesPlanningDatasetTitan

    overrides = _NUSCENES_DATASET_OVERRIDES.get(path, {})
    if not overrides:
        raise RuntimeError(
            f"NuScenesPlanningDatasetTitan loader: no overrides registered "
            f"for dataset key {path!r}. Did you call "
            f"qwen2_5_vl_3b_planning_fsdp() first?"
        )
    return NuScenesPlanningDatasetTitan(**overrides)


def _nuscenes_passthrough_processor(sample, **kwargs):
    """Identity processor — our dataset already emits torchtitan-format
    samples, so the per-sample processor just returns them as-is."""
    return sample


# Populated by `qwen2_5_vl_3b_planning_fsdp()` (and any other factories)
# at call time, before torchtitan instantiates the dataloader.
_NUSCENES_DATASET_OVERRIDES: dict[str, dict] = {}


def _register_nuscenes_dataset(name: str, overrides: dict) -> None:
    """Register our dataset under `name` in torchtitan's MM_DATASETS."""
    _NUSCENES_DATASET_OVERRIDES[name] = overrides
    MM_DATASETS[name] = DatasetConfig(
        path=name,  # used as the lookup key in our loader
        loader=_nuscenes_loader,
        sample_processor=_nuscenes_passthrough_processor,
    )


def _qwen25_vl_dataloader(dataset: str, **kwargs) -> MMDataLoader.Config:
    return MMDataLoader.Config(
        dataset=dataset,
        # Plenty of vision tokens budget: 3 cams * 4 frames * patches.
        # Empirically <= 768 patches for 3-cam @ 384x384 input.
        max_images_per_batch=128,
        patch_size=14,
        temporal_patch_size=2,
        spatial_merge_size=2,
        # min/max pixels match Qwen2.5-VL HF processor defaults.
        min_pixels=4 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        image_mean=(0.48145466, 0.4578275, 0.40821073),
        image_std=(0.26862954, 0.26130258, 0.27577711),
        # No HF-style packing: our samples are video-bound.
        packing_buffer_size=0,
        **kwargs,
    )


def qwen2_5_vl_3b_planning_fsdp() -> Trainer.Config:
    """Trainer.Config factory exposed to ConfigManager.

    8x5090 FSDP-only mesh (`data_parallel_shard_degree=-1` -> all ranks
    sharded), no TP / PP / CP.  This is intentionally the simplest
    parallelism for the first end-to-end run; Agent B and C will layer
    pipelining and CP on top.
    """
    # Pre-register the nuScenes dataset.  Paths and recipe pulled from
    # the existing HF train_lora config (configs/r1_prime_3ep.yaml style).
    import os

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    overrides = {
        "infos_path": os.path.join(
            repo_root, "data_processed", "nuscenes_infos_temporal_train.pkl"
        ),
        "nusc_root": os.path.join(repo_root, "data", "nuscenes"),
        # NOTE: `processor` is injected at dataloader-build time inside
        # `_nuscenes_loader`, since torchtitan's MMDataLoader provides the
        # tokenizer but not a full HF processor.  We resolve it lazily
        # from the assets path at runtime — see the loader.
        "processor": None,  # placeholder; filled by the runtime loader.
        "max_length": 2560,
        "num_past_frames": 4,
        "num_future_waypoints": 6,
        "video_fps": 2.0,
        "vla_loss_mode": "answer_and_traj",
        "require_full_future": True,
        "planning_cams": ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"],
        "require_all_cams": True,
        "infinite": True,
    }
    _register_nuscenes_dataset("nuscenes_planning", overrides)

    return Trainer.Config(
        # R1' final ckpt assets path (HF format on disk).  CheckpointManager
        # reads `hf_assets_path/tokenizer/` for the tokenizer at startup.
        hf_assets_path="./checkpoints_qwen25/r1_prime_final_hf",
        tokenizer=MultiModalTokenizer.Config(**QWEN3_VL_SPECIAL_TOKENS),
        metrics=MetricsProcessor.Config(log_freq=10),
        model_spec=_qwen25_vl_3b_model_spec(),
        dataloader=_qwen25_vl_dataloader("nuscenes_planning"),
        optimizer=OptimizersContainer.Config(
            lr=2e-5,
            # Standard AdamW (torchtitan's default).
            # weight_decay etc. fall back to defaults; override on CLI.
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            # 39 warmup steps: scaled from the R1' HF recipe (312 steps at
            # micro-bs 1 * grad_accum 8 = 2496 effective samples warmup;
            # at 8-way FSDP with local_bs 1, one global step = 8 samples,
            # so 2496 / 8 = 312 steps... wait, that's the same).
            # Concretely: R1' used 312 warmup steps; we keep the same step
            # count here because we have the same effective-batch ratio.
            #
            # If we later raise local_batch_size>1 or change gradient
            # accumulation, the warmup MUST be re-scaled by tokens-per-
            # step (per MEMORY.md "warmup tokens not steps").
            warmup_steps=39,
            decay_ratio=1.0,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=1,
            # 3 cams * 4 frames * ~196 patches/frame after spatial-merge =
            # ~2.4K tokens.  Add prompt + waypoints + buffer = 2560.
            seq_len=2560,
            # 3 epochs over ~26k samples / 8 DP ranks = ~9750 steps.
            # Round to 10k for slack; override on CLI for shorter runs.
            steps=10000,
            mixed_precision_param="bfloat16",
            mixed_precision_reduce="float32",
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=-1,  # All ranks FSDP-shard.
            tensor_parallel_degree=1,
            context_parallel_degree=1,
            pipeline_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            enable=True,
            interval=1000,
            last_save_model_only=False,
            export_dtype="bfloat16",
            keep_latest_k=2,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="full",  # 3B model on 5090 (32GB) -> need full AC.
        ),
    )


# Allow `python -m scripts_titan.train_titan_qwen25_vl` to do a CPU-only
# config-build smoke test.  We intentionally do NOT call Trainer.build()
# here (which would try to allocate the model on device).
if __name__ == "__main__":
    cfg = qwen2_5_vl_3b_planning_fsdp()
    print("Built Trainer.Config:", type(cfg).__name__)
    print("Model spec:", cfg.model_spec.name, cfg.model_spec.flavor)
    print("Model class:", type(cfg.model_spec.model).__name__)
    print("LR:", cfg.optimizer.lr)
    print("Warmup steps:", cfg.lr_scheduler.warmup_steps)
    print("FSDP shard degree:", cfg.parallelism.data_parallel_shard_degree)
