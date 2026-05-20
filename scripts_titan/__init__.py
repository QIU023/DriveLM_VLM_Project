# Agent A — Qwen2.5-VL torchtitan adapter (CPU-only setup; GPU runs deferred).
#
# This package provides:
#   * Qwen2_5_VLModel — subclass of torchtitan Qwen3VLModel with QK-norm and
#     DeepStack disabled to match Qwen2.5-VL.
#   * Qwen2_5_VLStateDictAdapter — HF Qwen2.5-VL checkpoint <-> torchtitan
#     state-dict converter.
#   * NuScenesPlanningDatasetTitan — IterableDataset wrapper around the
#     existing HF-style PlanningDataset so torchtitan's MMDataLoader can
#     consume it.
#   * train_titan_qwen25_vl — config_registry module exposing the
#     `qwen2_5_vl_3b_planning_fsdp` Trainer.Config factory.
