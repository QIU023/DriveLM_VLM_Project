# Upstream PRs from DriveLM_VLM_Project

Tracks open-source contributions raised (or queued for raising) from this
project. Each PR has its own numbered file with status, files touched, test
result, reviewer hints, and local refs.

## Index

| # | Repo | Title | Status |
|---|---|---|---|
| 001 | pytorch/torchtitan | step-decay LR scheduler (lr *= f every N steps) | drafted + tested locally; pushed to fork; awaiting Track A smoke validation |
| 002 | pytorch/torchtitan | Qwen3-VL PP wiring (DeepStack-aware, vision-pinned stage 0) | drafted locally |
| 003 | pytorch/torchtitan | Qwen3-VL CP wiring (text-only path; vision raises NotImplementedError) | drafted locally |
| 004 | (research note, not PR) | Vision-CP open problem in VLM training | doc only |
| 008  | QIU023/DriveLM_VLM_Project | HF Q-Former wired as cross_frame_compressor (Track A.1) | **DEPRECATED** — branch `qformer_hf_port`; superseded by 008b (wrong ablation axis: conflated fusion mechanism with cross-frame compression) |
| 008b | QIU023/DriveLM_VLM_Project | HF Q-Former as PROJECTOR replacement (Track A.1 redo) | drafted locally + CPU smoke; pending GPU SFT; branch `qformer_hf_projector_port` |

## Conventions

- **Numbered** by drafting order, not by submission order.
- **Status** values: `in progress`, `drafted locally`, `pushed to fork`, `PR opened`, `merged`, `closed`, `doc only`.
- **One PR = one file**: keep diffs small + focused. If a feature splits into 2
  changes, raise 2 PRs.
- **Branch naming**: `<feature>` on QIU023 fork (e.g. `step_decay_lr_scheduler`,
  `qwen3_vl_pp_wiring`).
- **Local validation gate**: do not open a real PR until we have run the
  feature in our own Track A training successfully. Filing a PR for code we
  haven't validated end-to-end is bad form.
