# Eval Credibility Audit (2026-05-26)

## ⚠️ CORRECTION (2026-05-26, verified against AutoVLA repo) — read first
`min_pixels=max_pixels=109760` is **AutoVLA's OFFICIAL SFT setting** (`config/training/qwen2.5-vl-3B-mix-sft.yaml`,
github.com/ucla-mobility/AutoVLA), NOT a bug. min==max → deterministic ~140 merged tok/frame on Qwen2.5-VL,
a deliberate resolution-for-coverage trade to fit 3cam×4frame=12 images in context. The "thumbnail bug / 报废"
framing below (which treated NATIVE 64×114 as the gold standard) is **WRONG**. Corrected verdicts:
- **B.5' Qwen2.5-VL-3B 3-cam (240 tok/cam) = AutoVLA-FAITHFUL** (≥ paper's ~140). L2 0.658 is a legitimate
  paper-spec result — TRUST it, not "downscaled/broken".
- **Qwen3-VL 3-cam (36 tok/cam) = genuinely below AutoVLA's 140**, but the root cause is patch-size transfer
  (Qwen3 patch-16 vs Qwen2.5 patch-14 → same 109760 px gives fewer tokens), NOT a downscale bug. Fix = RAISE
  max_pixels so Qwen3 lands at ~140 tok/cam (not "use native"). Under-resolution, fixable, worth a retrain.
- **Qwen3-VL 1-cam B.5'' (2800 tok/cam) = ABOVE AutoVLA spec (~20×)** — the deviation-toward-native model; it is
  the right substrate for the spatial-compression story but is NOT AutoVLA-faithful resolution.
- Lesson: audit against the PAPER/recipe spec, not native; when porting a pixel budget across patch sizes,
  re-derive the value to preserve TOKEN count.

## TL;DR (original — superseded by the correction above re: "bug" framing)
A pre-deploy probe (triggered by "1-cam worse than 3-cam looks wrong") uncovered that **every 3-cam
multimodal model, and the temporal-compression line, was trained/evaluated on silently-downscaled
thumbnail camera video**. The HD-map image cap `max_pixels=109760` leaked onto the Qwen3-VL / Qwen2.5-VL
`video_processor.size.longest_edge`, crushing cameras to 36–240 tokens/cam vs native ~2800–3648.
The eval-`max_length` 4096/8192 hardcode (originally suspected) is NOT a problem — 0% of samples exceed it.

## Root cause
- Config sets `min_pixels=max_pixels=109760` for the HD-map BEV image (~331×331).
- Qwen3-VL **and** Qwen2.5-VL (transformers 5.6) `video_processor` HONOR this cap → it was applied to camera video.
- Result baked into each ckpt's `processor_config.json` as `video_processor.size.longest_edge=109760`.
- Code is now fixed: `train_lora.py:2704-2727` only mutates the video processor when cfg sets `video_max_pixels` (the 3-cam yaml does not). A fresh retrain → native video. The existing ckpts predate/escaped the fix.

## Evidence (native vs as-saved, post-merge tokens/cam)
| Model | native | as-saved (capped) | downscale |
|---|---|---|---|
| B.5'' Qwen3-VL-4B 1-cam | 2800 | **2800 (native ✓)** | none — CLEAN |
| B.5''' Qwen3-VL-4B 3-cam | ~2800 | **36** | ~100× 🔴 |
| B.5' Qwen2.5-VL-3B 3-cam | 3648 | **240** | ~15× 🔴 |

## Blast radius
| Result | cam | affected? |
|---|---|---|
| B.5'' 1-cam Qwen3 (L2 0.705–0.715) | 1 | ✅ CLEAN (native video) |
| B.5''' 3-cam Qwen3 (L2 0.549) | 3 | 🔴 thumbnail — L2 is HD-map+bbox+ego, cameras useless |
| B.5' 3-cam Qwen2.5 (L2 0.658) | 3 | 🔴 downscaled 15× |
| P3 spatial compression sweep (FasterVLM/PruMerge…) | 3 (B.5') | 🔴 ran on downscaled video → "16× lossless" is because little visual info existed to lose |
| Temporal compression 8f/16f (meanpool/VTM/LongVU, R3–R5) | 1 (R1') | 🔴 also downscaled (grid 16×30 vs native 64×114) |
| Projector A.1/A.2/A.3 (Q-Former/resampler/pixelshuffle) | 1 | ✅ unaffected by THIS bug (compress to 64 tok by design); ckpts deleted |
| Qwen2.5 1-cam B.5 (L2 0.66) | 1 | mostly clean (native-ish), ckpt exists |

## Eval-max_length (separately checked) — NOT a bug
Token audit: B.5 997 / B.5' 1508 / B.5'' 3537 / B.5''' 901 median; all **0%** exceed eval cap (4096/8192).
1-cam re-eval at 6144 (vs 4096) = 0.705 ≈ 0.715 → no truncation effect. Added `planning_eval.py --eval-max-length` override anyway for train/eval parity hygiene.

## Process failures (own them)
1. HD-map cap leaked to video processor — config comment "video UNCHANGED = native" was a false assumption for Qwen3/2.5-VL.
2. Documented on 2026-05-25 (`feedback_audit_must_match_training_processor`) yet B.5''' was never retrained; 0.549 kept being cited as a real vision result.
3. Found only at the quant/deploy stage. → Now enforced by P0 gate `feedback_p0_pretrain_recite_gate` (recite 7-item audit before any launch).

## Retrain priority
- **P0 — B.5''' Qwen3-VL-4B 3-cam multimodal (native video):** the headline 顶配 + deploy/quant target; also the substrate to redo the spatial-compression sweep honestly. ~2.5h. Config already omits `video_max_pixels` → native on retrain. MUST pass the P0 recite-gate first.
- **P1 — B.5' Qwen2.5-VL-3B 3-cam multimodal (native):** only if the cross-backbone (2.5 vs 3) comparison is needed for the story; else superseded by B.5'''.
- **P2 — Temporal-compression line (8f/16f) on native 1-cam:** re-setup only if that ablation is part of the deliverable; ckpts deleted, full re-run needed. Lowest priority.
- **No retrain needed:** B.5'' 1-cam Qwen3 (native, clean) — keep as the verified 1-cam vision baseline; projector A.x (unaffected by this bug, but ckpts gone).

## Fixes landed
- `scripts/train_lora.py:2704-2727` — video processor only mutated on explicit `video_max_pixels` (already in tree).
- `scripts/planning_eval.py` — added `--eval-max-length` override.
- Memory: `feedback_p0_pretrain_recite_gate` (P0 gate), `feedback_audit_must_match_training_processor` (status updated).
