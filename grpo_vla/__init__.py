"""grpo_vla: veRL-compatible GRPO components for the nuScenes planning VLA.

Modules:
    reward          -- 5-dim composite reward over (predicted_tokens, GT waypoints, bboxes, ego_state)
    dataset_adapter -- VeRLNuScenesDataset wrapping MultiModalPlanningDataset
    test_reward     -- CLI smoke-test: perfect > static > wrong-dir over 10 real val samples

NOTE: Do NOT modify upstream scripts/. All work is additive in this package.
"""
