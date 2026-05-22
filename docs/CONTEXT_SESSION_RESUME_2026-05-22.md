# Session Resume — 2026-05-22

> Snapshot of the conversation that finalized B.5/B.6 v2 eval, decided the
> fusion roadmap, set up TRT deployment scaffolding, and pinned the 185K
> benchmark identity. Continues from
> [docs/session_context.md](session_context.md) (2026-03-17 init) and
> [docs/full_modal_vla_research.md](full_modal_vla_research.md) (§11 v2 results).

## TL;DR

- v2 eval matrix complete. **B.5 multimodal wins 4/8 axes**; not "all washed out" as the pre-fix narrative claimed.
- Eval pipeline bug (`_build_batch_inputs` discarding HD map + bbox) **fixed**; six cells re-run; results published to `eval_results/track_v2/`.
- Fusion roadmap: **B.5 = AD-prod mature; B.7 Flamingo XATTN was downgraded** in favor of OmniDrive-style BEV-unified or LLaVA-style staged pretraining for the next ablation.
- TRT-LLM deployment scaffolding now in `deploy/` (env not yet installed; ~6-10h end-to-end on first run; vLLM baseline alternative documented).
- 185K = **CODA-LM 184,480** (DriveMM expansion of CODA-LM 36,896). Standard benchmark, NOT homebrew interpolation.

## v2 eval matrix — final outcomes

| Cell | tag | Notes |
|---|---|---|
| R1' / A.0 baseline | `R1prime_1cam` | 1-cam camera-only ref |
| R1'' baseline | `R1prime_3cam` | 3-cam camera-only — overall L2 winner |
| B.5 true L2 | `B5_multimodal` | HD map + bbox + ego at eval — wins collision / turning_L2 / lane_change_L2 / heading_error |
| B.5 robustness | `B5_cameraonly` | modality-degradation reference |
| B.6 true L2 | `B6_multimodal` | dropout p=0.1 — **+0.13 L2 vs B.5**, real negative finding |
| B.6 robustness | `B6_cameraonly` | dropout reg under masked modalities |

Detailed numbers in `eval_results/track_v2/*.json` + `logs/all_ckpts_eval_v2_summary.md`. Per-axis winners and the bug-fix narrative are documented in
[docs/full_modal_vla_research.md](full_modal_vla_research.md) §11 ("Phase B v2 results").

## What broke in v1 (and how it was fixed)

`scripts/planning_eval.py::_build_batch_inputs` rebuilt the user prompt from scratch
via `_build_user_content_multicam`, which **silently discarded** the HD map image and
bbox text — so `--multimodal` was byte-equivalent to camera-only. Symptom: B.5 and
R1' had identical L2 in the v1 sweep.

Fix:
1. Duck-type detect `MultiModalPlanningDataset` (via `hasattr(ds, 'hdmap_split_dir')`).
2. Call `_build_user_content_multimodal` + pass `images=[hdmap_pil]` to the processor.
3. CPU-only smoke (`scripts/_smoke_planning_eval_multimodal.py`) asserts the HD map
   key appears in the processor output and bbox text appears in the decoded prompt.

User feedback was sharp: *"你核心结果代码都是错的了 还问我要不要重跑？自觉修完重新跑一遍所有ckpt啊"*. The v2 sweep was a forced re-run of all 4 preserved
ckpts × the bug-fixed pipeline + new behavior metrics.

## New metrics added (v2)

`planning_eval.py` now emits in every result JSON:
- `scenario_counts` + `scenario_metrics`: sub-scenario L2 (straight / turning / lane_change / braking / cruising / stationary) — differentiates models when overall L2 saturates.
- `behavior`: `heading_error_rad`, `lateral_accel_rms`, `speed_error`, `hard_brake_rate`, `progress_ratio` — open-loop safety surrogates.

Motivation: open-loop L2 alone gives ~3% diff between baseline and multimodal at
this scale (per *"Is Ego Status All You Need?"*, arXiv 2312.03031). Sub-scenario +
behavior axes recover ~15% per-axis spread.

## Fusion landscape (8 prod approaches)

Full table + per-approach maturity/cost in [docs/full_modal_vla_research.md](full_modal_vla_research.md). Quick map:

| # | Approach | Prod adopters | Status here |
|---|---|---|---|
| 1 | **Image-as-modality + text serialize** | Waymo EMMA / AutoVLA / Tesla FSD / **猜测 XPeng VLA 2.0** | ✅ **B.5 / B.6** done |
| 2 | BEV-Unified Token Stream | NVIDIA OmniDrive, UniDriveVLM | candidate B.9 |
| 3 | Cross-modal Q-Former | OmniDrive variants, ImpromptuVLA | covered by A.1 (Q-Former projector) |
| 4 | Late fusion Flamingo XATTN | Wayve LINGO-2 only | **B.7 — DOWNGRADED** (research/Wayve-only, no major OEM) |
| 5 | Sparse query set | Sparse4D, StreamPETR | candidate B.10 |
| 6 | Staged pretraining (LLaVA-style) | LLaVA / MiniGPT | candidate B.11 |
| 7 | Dual-system S1/S2 | Li Auto DriveVLM-Dual | not pursued (XPeng JD targets single-stream) |
| 8 | World model pre-fusion | Wayve GAIA, XPeng X-World | infra-heavy, deferred |

**Decision**: B.7 not worth the effort given XPeng JD targets single-stream prod.
Next ablation is one of B.9 (BEV-unified) or B.11 (staged pretraining), pending
data-scale decision below.

## 185K = CODA-LM 184,480 ✓

Standard benchmark, NOT homebrew. The expansion was done by **DriveMM** (NeurIPS
2024) from the original CODA-LM 36,896 corner-case-AD-VQA subset of the CODA
dataset. References:
- [CODA-LM original](https://arxiv.org/abs/2404.10595)
- [DriveMM expansion](https://arxiv.org/abs/2412.07689) — 184,480 samples

**Implication for our scaling**: 185K is image+text VQA, NOT extra nuScenes
camera frames. Disk impact is ~5-10 GB additional (mostly text + low-res images
already inside CODA), not the ~80 GB 12Hz sweep scenario I originally costed.

**Revised disk budget for 185K scale-up** (vs. current 103G free):

| Item | Bytes | Notes |
|---|---|---|
| CODA-LM images | ~30 GB | ~160 KB/sample × 184K |
| CODA-LM annotations | ~500 MB | JSONL |
| New ckpts (1× full SFT) | 18 GB | one full-SFT run |
| **Total new draw** | **~48 GB** | |
| Free after | ~55 GB | safe per [[feedback_disk_panic_protocol]] (>15G) |

✅ Feasible WITHOUT pre-cleanup. But it's still good hygiene to push the 2
candidate-kill ckpts to HF Hub first (next steps below).

## TRT deployment scaffolding

Files added in `deploy/` (all from this session, 2026-05-22):

| File | Purpose |
|---|---|
| `requirements.core.txt` | curated subset of training stack (33 deps) |
| `requirements.full.txt` | full `pip freeze` (356 pkgs) |
| `ENV_FREEZE.md` | hardware + driver + python version pins |
| `trt_convert_qwen25vl.py` | end-to-end conversion stub: quantize → LM engine → vision engine |
| `vllm_baseline.sh` | vLLM baseline (faster setup, ~70-80% TRT perf) |
| `README.md` (existing, 2026-05-21) | hardware matrix / TRT-LLM mental model / Orin-Thor context |
| `quantize_fp4.sh`, `build_engine.sh`, `benchmark.py` (existing) | per-step helpers |

**TRT-LLM not yet installed.** End-to-end first-time cost: ~6-10h
- container bring-up + TRT-LLM install: 1-2h
- HF → NVFP4 quantize: 0.5h
- engine build (LM + vision): 0.5h
- parity vs HF: 1-2h
- latency benchmark + tuning: 1-2h
- buffer for version-flag drift: 0.5d

**Faster alternative path**: vLLM baseline runs on host (cu130 nightly), ~1h
setup, no container, no NVFP4. Use as a quick latency sanity table before
paying the TRT debug cost on every ckpt iteration.

## Memory rules invoked / created this session

Memory consulted (existing):
- [[feedback_no_churn_surface_problems]] — three idle "/loop stage 0 monitor" calls were short-circuited instead of auto-pivoting to new monitor targets.
- [[feedback_disk_panic_protocol]] — used in the 185K disk budget; informed the recommendation to back up + clean before any 12Hz scenario.
- [[reference_torchrun_uses_system_python]] — used in `ENV_FREEZE.md` install instructions.
- [[feedback_post_train_cleanup_intermediates]] — applied during 71G ckpt audit.

No new memory created this session — only existing rules applied.

## Next steps (priority-ordered)

**P0 (decision required from user before any GPU work)**:

1. Approve ckpt purge — keep R1' + B.5, drop R1'' + B.6 ckpts (results JSON stay).
   Releases ~36 GB.
2. Decide on next ablation: **B.9 BEV-unified** or **B.11 staged pretraining**.
3. Approve CODA-LM 184K download (~30 GB) for actual 185K scale-up.

**P1 (when GPU window opens)**:

4. TRT end-to-end (6-10h) — run `deploy/trt_convert_qwen25vl.py --hf-dir <merged_R1' or B.5> --out /tmp/trt_test_5090`. Capture latency table.
5. (Or skip TRT first pass) vLLM baseline benchmark — `bash deploy/vllm_baseline.sh` + plot tok/s vs context length.

**P2 (background research)**:

6. EMMA / AutoVLA / XPeng VLA 2.0 paper read for B.9 / B.11 details before commit.
7. Update resume bullets per `docs/interview_prep.md` with the v2 numbers.

## Key files for next session pickup

| Path | What's there |
|---|---|
| `eval_results/track_v2/*.json` | v2 sweep raw results (6 cells) |
| `logs/all_ckpts_eval_v2_summary.md` | tabulated v2 matrix |
| `scripts/planning_eval.py` | fixed eval pipeline (multimodal-aware) |
| `scripts/multimodal_planning_dataset.py` | per-modality dropout dataset |
| `scripts/run_all_ckpts_eval_v2.sh` | reproducer for the v2 sweep |
| `scripts/_smoke_planning_eval_multimodal.py` | CPU smoke for the multimodal path |
| `configs/nuscenes_planning_b5.yaml` / `b6.yaml` | training configs |
| `deploy/` | this session's deliverable — TRT scaffolding |
| `docs/full_modal_vla_research.md` §11 | v2 narrative + per-axis winners |
| `docs/CONTEXT_SESSION_RESUME_2026-05-22.md` | this file |

## Open questions for user

1. **B.9 or B.11 next?** BEV-unified is closer to OmniDrive (NVIDIA, prod) but
   needs a BEV encoder train; staged pretraining needs CODA-LM 184K download
   but reuses the existing fusion stack.
2. **TRT first or vLLM first?** TRT is the right end-state (Blackwell/Thor); vLLM gets a baseline number in ~1h. Recommend vLLM first if interview is close.
3. **Ckpt purge approval?** R1'' + B.6 ckpts safe to delete (results preserved in JSON, can re-train if needed but unlikely worth it).

---

## Late-session addendum (post 14:38 UTC) — B.5' full eval + R1''' retrain + FSDP save/load code overhaul

### B.5' (顶配 = 3-cam × multi-modal) full eval results

`eval_results/track_v2/B5prime_3cam_multimodal.json`:

| metric | B.5' | reference |
|---|---|---|
| L2_avg | **0.658 m** | R1' 0.642, B.5 0.678 |
| collision | 3.73% | B.5 best at 3.07% |
| turning_L2 | 0.996 m | B.5 best at 0.964 |
| lane_change_L2 | 0.812 m | B.5 best at 0.670 |

**Key finding**: 3-cam already provides enough spatial context that HD-map +
bbox become marginal/redundant on this 24K-sample scale. The "multi-modal at
1-cam" win (B.5 vs R1') doesn't extend additively to 3-cam (B.5' vs R1''/R1''').
Resume bullet's "multi-modal -18% turning" claim is still valid (B.5 vs R1'),
but顶配 framing was wrong — 3-cam + multi-modal didn't beat 3-cam alone.

### R1''' retrain — broken `_save_model_and_state` discovered

R1'' v1 (3-cam camera-only baseline) was found to have `max_length=8192`
silent truncation: 3-cam at 109760 px = ~10944 LM tokens, exceeds 8192,
prompt's tail (HD-map / bbox / suffix) gets clipped. R1''' = retrain with
max_length=12288 + AC=true to make a fair baseline.

**During R1''' resume attempt 2026-05-22, three pre-existing bugs surfaced**:

1. **`_save_model_and_state`** wrote optim via `torch.save(optimizer.state_dict())`
   on rank 0 only. Under FSDP this returns ONLY rank-0's local shard
   (1/N of full state). Confirmed: training_state.pt = 2.7 GB vs expected
   ~24 GB. ckpt-1100 was unrecoverable for proper FSDP resume.

2. **Resume load path** unconditionally loaded the broken file on every rank.
   Combined with FSDP reshard overhead, caused ~3 GB CPU+GPU spike per rank
   on first forward → OOM at step 1101.

3. **OOM handler** at `train_lora.py:3095` does `torch.cuda.empty_cache(); continue`.
   Under FSDP this leaves `_all_handles` corrupted → next forward dies with
   `AttributeError: ... no attribute '_all_handles'`. Skip-and-continue is a
   single-GPU / DDP pattern; FSDP must abort.

### Code fixes (committed to `scripts/train_lora.py`)

| Fix | Path | Mechanism |
|---|---|---|
| Save uses FSDP-aware state IO | `_save_model_and_state` | replaced hand-rolled `torch.save({"optimizer": ...})` with `accelerator.save_state(dir, save_model=False, safe_serialization=False)` + JSON `training_meta.json` for step/epoch |
| Strip duplicate FSDP-format model | `_save_model_and_state` | `accelerator.save_state` writes `pytorch_model_fsdp_0/*.distcp` even with `save_model=False` under FSDP — `shutil.rmtree` post-save (canonical model is `model.safetensors` from `save_pretrained`) |
| Load uses FSDP-aware state IO | resume block | `accelerator.load_state(dir, load_model=False)` + JSON meta read; legacy `training_state.pt` explicitly refused with WARN |
| Model weights load on resume | `from_pretrained` site | full_sft now points `from_pretrained` at the resume_path's `model.safetensors` directly (was a no-op NOTE) |
| OOM handler aborts under FSDP | OOM except branch | `is_fsdp = accelerator.state.fsdp_plugin is not None` → raise; non-FSDP keeps skip-and-continue |

### Ckpt size after fix

| section | size | note |
|---|---|---|
| `model.safetensors` | 16 GB | bf16 3B Qwen2.5-VL, canonical for eval |
| `accelerate_state/optimizer_0/` | ~22 GB | sharded fp32 Adam moments across 8 ranks |
| `accelerate_state/random_states_*.pkl` × 8 | < 1 MB | per-rank RNG for resume reproducibility |
| `training_meta.json` | 48 B | step / epoch / batch_idx |
| **TOTAL** | **~38 GB / ckpt** | (vs broken 2.7 GB or unstripped 54 GB) |

`keep_latest_k: 2 → 1` in config to keep 2-ckpt save peak (76 GB) safely under
the 12 GB disk-panic threshold given 140 GB free at launch.

### R1''' FIXED4 launch (~19:19 UTC)

Currently running. Step ~1700 / 2242 at 23:33 UTC. val L2 plateau ~0.28-0.30 m
(lowest mid-train 0.245 @ step 1550). Final standalone eval pending post-completion.

**Predicted full eval**: R1''' L2_avg ~0.45-0.55 m (better than R1'' v1 0.622 if
the truncation hypothesis was right). If R1''' beats B.5' on L2_avg, the
"3-cam ≥ multi-modal-at-1-cam" story holds; B.5 multi-modal advantage is
turning/collision-specific not L2-avg-specific.

### Memory rules created from this debug session

- `feedback_fsdp_resume_use_accelerate_state.md` — FSDP resume must use accelerator.save_state/load_state, never hand-rolled torch.save
- `feedback_fsdp_oom_handler_must_abort.md` — OOM under FSDP must abort; skip-and-continue is DDP-only

### Outstanding for next session

1. Run R1''' full eval (~10 min DP-8) once training completes (~01:00 UTC tomorrow)
2. TRT-LLM deployment of顶配 ckpt (B.5' or R1''' depending on full eval)
3. Reconsider whether 顶配 framing for resume bullet still holds after R1''' full result
4. Task #115: full E2E save→reload→parity smoke for the new resume code path (defer to dedicated session)

