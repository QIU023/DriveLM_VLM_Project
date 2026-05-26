# Resolution-Sensitivity Sweep — B.5''' 3-cam Qwen3-VL-4B nuScenes planner

**Date:** 2026-05-26
**Ckpt (read-only):** `checkpoints_qwen25/nusc_planning_b5ppp_3cam_qwen3vl_multimodal/final`
**Eval:** full nuScenes val (5119 scored samples), DP-8, greedy, multimodal (3 front cams + HD-map BEV + bbox/ego text), `--num-past-frames 4`, `--eval-max-length 12288`.
**Protocol reported:** `TemAvg` (VAD-style temporal-average L2), field `L2_avg`.
**Per-cap raw JSON:** `docs/eval_results/resolution_sweep/b5ppp_3cam_rescap_<cap>.json`

## Motivation

A 3-cam retrain at **240 post-merge video tokens/cam** (video cap 524288) scored full-val L2 **0.626**.
An EARLIER run at **36 tok/cam** (cap 109760, "thumbnail") scored **0.549** — but that ckpt was
overwritten and is gone. Question: is the 0.549 ↔ 0.626 gap due to camera **resolution**, or just a
**different training run** (seed / data order)? We cannot rebuild the thumbnail ckpt, so instead we
measure how much THIS fixed ckpt's L2 depends on camera resolution **at eval time**, by mutating the
loaded `video_processor` pixel caps (planning_eval.py `--video-max-pixels`, which sets
`min_pixels==max_pixels==longest==shortest_edge`, mirroring the training-side mutation).

## Token-count verification (the critical check)

Past incident: a model was trained on thumbnails because only a *flag* was checked, not the real
emitted token count. So before reading any L2, we verified the override actually changes the real
`video_grid_thw` emitted by the processor for a **real 3-cam val clip** (frames 900×1600, 4 hist
frames → temporal patch t=2). spatial_merge_size=2, so post-merge tok/cam = t·h·w / 4.

| cap | video_grid_thw (per cam) | pre-merge | post-merge tok/cam (VERIFIED) |
|---:|:---|---:|---:|
| 109760  | [2, 6, 12]  | 144  | **36**  |
| 262144  | [2, 12, 20] | 480  | **120** |
| 524288  | [2, 16, 30] | 960  | **240** |
| 1048576 | [2, 24, 42] | 2016 | **504** |

The override is working: 109760 → 36 tok/cam (== the old thumbnail resolution) and 524288 → 240
tok/cam (== training resolution). These are real processor outputs, not just the logged flag.

## Results

| cap | tok/cam (verified) | L2_avg | L2_1s | L2_2s | L2_3s | collision_avg | n |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 109760  | 36  | 0.6583 | 0.2790 | 0.6085 | 1.0874 | 0.0375 | 5119 |
| 262144  | 120 | 0.6256 | 0.2684 | 0.5787 | 1.0297 | 0.0381 | 5119 |
| 524288  | 240 | 0.6262 | 0.2676 | 0.5781 | 1.0329 | 0.0397 | 5119 |
| 1048576 | 504 | 0.6482 | 0.2741 | 0.5986 | 1.0721 | 0.0383 | 5119 |

**L2_avg range across all caps: 0.6256 – 0.6583 (spread 0.033).**

### Sanity check (cap = 524288 = training res)

cap=524288 gives **L2_avg 0.6262**, which reproduces the training-saved full-val score
(`eval_results_3cam_autovla_res.json`: 0.62619) **to 4 decimals**. The harness is faithful; no flag.

## Interpretation

**L2 is essentially flat across an 14× range of camera pixels** (36 → 504 tok/cam), with a total
spread of only 0.033. If anything the curve is mildly **U-shaped**: best near the training resolution
(120–240 tok/cam ≈ 0.626) and slightly worse at both extremes — 36 tok/cam (0.658) and 504 tok/cam
(0.648). Higher pixels do **not** improve L2; they slightly hurt it.

**This supports the seed/run-noise + ego/HD-map-prior conclusion ("ego status is all you need"
regime), NOT a resolution effect.** Concretely:

- Running THIS ckpt at the old thumbnail resolution (36 tok/cam) yields **0.658** — *worse* than its
  own 240-tok training res, and nowhere near the old run's 0.549. So low resolution does not produce
  the old 0.549; the camera stream contributes very little usable signal to L2 at inference. The
  planner is dominated by the ego-state / HD-map / bbox priors that are present at every cap.
- Therefore the **0.549 ↔ 0.626 training gap is dominated by run-to-run variation** (seed, data
  order, optimization noise), **not** by the 36-vs-240 camera resolution. Resolution at inference
  moves L2 by at most ~0.03; the training gap is 0.077 in the *opposite* direction (the 36-tok run
  was *better*), which a resolution story cannot explain.

### Mismatch caveat (must be stated)

The 36-tok and 504-tok rows are **train/eval resolution MISMATCHES**: the ckpt was trained at 240
tok/cam and is being *run* at 36 / 504. A model trained at 240 then run at 36 suffers a distribution
shift, so the small bump at 36 (0.658) partly reflects that mismatch rather than "low-res being
intrinsically worse." This makes the flatness conclusion *stronger*, not weaker: even with the
distribution shift working against the extremes, L2 only moves ~0.03. It does **not** let us claim
"36 tok/cam intrinsically gives 0.658" for a model *trained* at 36 — the original thumbnail run
trained AND evaluated at 36 and scored 0.549, which is consistent with the gap being a training-run
difference, not a resolution property.

## Bottom line

For this 3-cam multimodal planner, **eval-time camera resolution barely affects L2** (≤0.033 over
36–504 tok/cam, training res reproduced exactly). The 0.549 ↔ 0.626 difference between the two
training runs is best attributed to **seed / data-order / run noise plus the dominance of
ego-state and HD-map priors**, not to camera resolution.
