# Branch Cleanup — Deferred Work

Tracks branches that should be cleaned up but are kept around for now per user preference (2026-05-21).

## Current branch inventory (as of 2026-05-21)

| Branch | Status | Notes |
|---|---|---|
| `main` | KEEP | upstream main |
| `qwen25_vl_video_vla` | KEEP (current trunk) | A.1-A.N integrated; all fixes here |
| `pixelshuffle_projector` | KEEP (torchtitan ref) | torchtitan-era working impl |
| `perceiver_resampler_projector` | KEEP (torchtitan ref) | torchtitan-era working impl |
| `pixelshuffle_hf_port` | KEEP for now | will cherry-pick `scripts/pixelshuffle_projector_hf.py` + `configs/nuscenes_planning_3cam_pixelshuffle.yaml` into trunk for A.2; rest of branch is stale (old train_lora.py / planning_eval.py) |
| `resampler_hf_port` | KEEP for now | same — cherry-pick `scripts/perceiver_resampler_projector_hf.py` + `configs/nuscenes_planning_3cam_resampler.yaml` for A.3 |
| `qformer_hf_projector_port` | DELETE later | already merged into trunk (commit `8899db0`) |
| `train_lora_l2_val` | DELETE later | already merged into trunk (commit `c2e2f91`) |
| `qformer_hf_port` | DELETE later | DEPRECATED — original design (qformer as cross_frame_compressor) was wrong; superseded by `qformer_hf_projector_port` |
| `video_vla` | DELETE later | early plan doc; no active work |

## Why deferred

User preference 2026-05-21: focus on shipping A.1/A.2/A.3 results before cleaning branches. Merge attempts have shown that naive `git merge pixelshuffle_hf_port → qwen25_vl_video_vla` would REVERT trunk-side fixes (qformer projector + save round-trip + DP L2 val + FSDP wrap) because the hf_port branches were forked from an earlier trunk state.

The correct integration is **cherry-pick the additive files only** (projector code + configs + docs), then wire dispatch fresh in trunk's already-updated `train_lora.py` / `planning_eval.py`. That's exactly what the A.2/A.3 prep step does.

## Cleanup sequence (when authorized)

After A.2 + A.3 ship and we're confident no rollback needed:

```bash
# Already-merged or DEPRECATED — safe to delete
git branch -D qformer_hf_projector_port  # in trunk @ 8899db0
git branch -D train_lora_l2_val          # in trunk @ c2e2f91
git branch -D qformer_hf_port             # DEPRECATED design
git branch -D video_vla                   # plan doc only

# After A.2 / A.3 cherry-picks land in trunk
git branch -D pixelshuffle_hf_port        # contents cherry-picked
git branch -D resampler_hf_port           # contents cherry-picked

# KEEP these (working refs):
# - main
# - qwen25_vl_video_vla (trunk)
# - pixelshuffle_projector (torchtitan ref)
# - perceiver_resampler_projector (torchtitan ref)
```

## Cherry-pick recipes (for A.2 / A.3 prep)

### A.2: pixelshuffle from `pixelshuffle_hf_port`
```bash
# On qwen25_vl_video_vla
git checkout pixelshuffle_hf_port -- scripts/pixelshuffle_projector_hf.py
git checkout pixelshuffle_hf_port -- configs/nuscenes_planning_3cam_pixelshuffle.yaml
git checkout pixelshuffle_hf_port -- docs/upstream_prs/009_hf_pixelshuffle_projector.md
# Then write dispatch in train_lora.py + planning_eval.py
# Then write configs/nuscenes_planning_1cam_pixelshuffle.yaml (already on trunk)
```

### A.3: resampler from `resampler_hf_port`
```bash
git checkout resampler_hf_port -- scripts/perceiver_resampler_projector_hf.py
git checkout resampler_hf_port -- configs/nuscenes_planning_3cam_resampler.yaml
git checkout resampler_hf_port -- docs/upstream_prs/010_hf_perceiver_resampler_projector.md
# Then dispatch + 1cam config (already on trunk)
```
