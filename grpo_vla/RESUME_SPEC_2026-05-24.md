# GRPO VLA Overnight 2026-05-24 — Handover (paused, not failed)

## What landed today (real value, all committed)

| Component | Status | Location | Notes |
|---|---|---|---|
| 4 agents built scaffolding | ✅ | `grpo_vla/` | reward + dataset adapter + config + launcher + e2e smoke |
| `reward.py` 5-dim composite | ✅ | sanity 10/10 PASS | perfect>static>wrong_dir |
| `dataset_adapter.py` | ✅ | wraps MultiModalPlanningDataset | emits veRL parquet rows |
| `build_parquet.py` v2 | ✅ | **188× faster** (37 rows/s) | JPEG q75 + 448px + 8-worker parallel |
| 12K train + 500 val parquet | ✅ | `grpo_vla/data/` 1.35G total | persists, idempotent |
| veRL v0.7.1 cloned + installed | ✅ | `/workspace/verl/` | remote pushed `qwen25_vla_grpo` |
| SGLang fork qwen25_vla_inference | ✅ | `/sgl-workspace/sglang/` | pushed 2 shim commits (trunk-drift + `_launch_subprocesses`) |
| vLLM 0.21.0 installed + many deps | ✅ partial | system venv | flash_attn missing — blocker |
| e2e save-load smoke | ✅ | `smoke/e2e_save_load_smoke.py` | delta=0 verified |
| sgl-kernel dist-info shim | ✅ | `/usr/local/lib/python3.12/dist-packages/sgl_kernel-0.4.2.dist-info/` | local only |
| Watchdog | ✅ active | `watchdog/watchdog.sh` | armed as Monitor `bbhvpcsmr` |

## What blocked (cascading dep hell, 5h, ~9 distinct fixes)

1. ✅ Hydra config search path (added to launcher)
2. ✅ veRL parquet schema (list-of-dicts not str; rewrote build)
3. ✅ SGLang `_launch_subprocesses` alias (committed to fork)
4. ✅ sgl-kernel dist-info missing (wrote fake METADATA)
5. ❌ **SGLang scheduler CUDA `release_block`** init crash (likely actor↔rollout GPU context conflict; option B layout not configured)
6. ✅ veRL `rollout.mode=sync` removed → `async`
7. ✅ vLLM many deps (cbor2, cpuinfo, ijson, watchfiles, depyf, model-hosting-container-standards, ...)
8. ❌ **vLLM Qwen2.5-VL needs flash_attn ≥2.6** for `flash_attn.ops.triton.rotary`; our system flash_attn is a broken stub

**Root cause of (5) and (8)**: system venv is tuned for our SFT training stack (torch 2.10 + custom AttnRes + sglang fork pinned to compat versions). Both rollout backends (sglang and vLLM) want different torch/flash_attn pins that conflict with the SFT stack.

## Fresh-eye path forward (recommended for next session)

**Dedicated venv approach** — don't fight system venv:

```bash
# 1. Build venv with veRL's official sglang/vllm extras
python -m venv /venv/grpo_vla
source /venv/grpo_vla/bin/activate
pip install -e /workspace/verl[sglang]   # OR [vllm]
# This installs sglang 0.5.6 + torch 2.9.1 (the veRL-tested combo)

# 2. Symlink data + reuse our parquets
ln -s /workspace/DriveLM_VLM_Project/grpo_vla/data /venv/grpo_vla/data

# 3. Run with --python /venv/grpo_vla/bin/python in launcher
```

Trade-off: separate venv means SFT stack and RL stack don't share env. Cleaner. ~2h to set up + likely the only fix that gets past dep hell.

## Files to revisit

- `grpo_vla/configs/grpo_b5prime_3cam.yaml` — 11-row audit table already done; keep
- `grpo_vla/launch_grpo_b5prime.sh` — has TOTAL_STEPS=3000 + KEEP_CKPTS=1 + save_contents=[model] overrides; keep
- `grpo_vla/build_parquet.py` — fast version, parquets are valid, **reuse** (no rebuild)
- `grpo_vla/reward.py` — `planning_reward` shim verified, sanity passes; keep
- `grpo_vla/dataset_adapter.py` — keep
- `grpo_vla/watchdog/watchdog.sh` — keep

## Quick commands

```bash
# Verify parquets still valid
/usr/bin/python3 -c "import pandas as pd; df=pd.read_parquet('/workspace/DriveLM_VLM_Project/grpo_vla/data/nusc_planning_train.parquet'); print(df.shape, df.columns.tolist())"

# Verify reward sanity (no model needed)
cd /workspace/DriveLM_VLM_Project && /usr/bin/python3 grpo_vla/test_reward.py

# Check sglang fork commits
cd /sgl-workspace/sglang && git log --oneline qwen25_vla_inference -5

# Check verl branch
cd /workspace/verl && git log --oneline qwen25_vla_grpo -3
```

## Estimated time to landing

| Phase | Hours |
|---|---|
| Dedicated venv setup + reinstall veRL+sglang clean | 2-3h |
| Smoke `smoke_rollout_step.py` end-to-end in new venv | 30min |
| Launch 3000 step GRPO (if smoke OK) | ~28h |
| Eval + commit + report | 2h |

**Total**: ~33h ≈ 1.5 days from fresh start.

## TODO for next session (in order)

1. Make dedicated `/venv/grpo_vla` with `pip install verl[sglang]` (NOT --no-deps)
2. Run `grpo_vla/test_reward.py` in new venv to verify
3. Run `grpo_vla/smoke_rollout_step.py` with new venv (this verifies veRL+sglang stack actually rolls 1 step)
4. If smoke OK: launch `bash grpo_vla/launch_grpo_b5prime.sh` overnight (PYTHONPATH update may be needed)
5. Mid-training: every 200 steps check reward trend (should be increasing if GRPO converging)
