# B.5''' Qwen3-VL-4B × 3-cam × multimodal — AutoVLA-res retrain eval (2026-05-26)

Retrain of B.5''' at **AutoVLA-faithful 240 post-merge tok/cam** (`video_min/max_pixels=524288`
→ video_grid_thw `[2,16,30]` → 240 tok/cam), replacing the earlier 36-tok/cam thumbnail run.
Recipe: FSDP2, LBS=4 × GA=1 × world=8 = GBS 32, no AC, no offload, LR 2e-5 AutoVLA step-decay,
3 epochs (2241 steps, 2.55 s/it, ~95 min). Saved processor verified: video `longest_edge=524288`
(240 tok/cam), HD-map image cap 109760 — NOT the old thumbnail.

## Full 5119-val planning L2 (DP-8, batch 4, eval_max_length=12288, 0% truncation)

| protocol | L2_1s | L2_2s | L2_3s | **L2_avg** | collision_avg |
|---|---|---|---|---|---|
| TemAvg (VAD) | 0.268 | 0.578 | 1.033 | **0.626** | 0.040 |
| NoAvg | 0.370 | 1.099 | 2.265 | 1.244 | — |

Cameras: CAM_FRONT + CAM_FRONT_LEFT + CAM_FRONT_RIGHT. n_scored=5119 (full UniAD/VAD val, 0 leakage).
Output JSON: `checkpoints_qwen25/nusc_planning_b5ppp_3cam_qwen3vl_multimodal/final/eval_results_3cam_autovla_res.json`.

## Per-scenario (TemAvg L2_avg)
| scenario | n | L2_avg | collision |
|---|---|---|---|
| stationary | 939 | 0.258 | 0.030 |
| straight | 2674 | 0.648 | 0.040 |
| braking | 314 | 0.721 | 0.029 |
| cruising | 305 | 0.751 | 0.066 |
| lane_change | 346 | 0.847 | 0.040 |
| turning | 541 | 0.889 | 0.044 |

Non-degenerate: error scales sensibly with maneuver difficulty (stationary lowest, turning highest).

## Reading of the number (honest)
- **Old thumbnail (36 tok/cam) L2 0.549 → new AutoVLA-res (240 tok/cam) L2 0.626.** Higher camera
  resolution did **not** improve open-loop L2. This is expected and consistent with the project's
  broader finding: nuScenes open-loop L2 is dominated by the HD-map BEV + 3D-bbox + ego-state priors,
  not camera pixel detail. (It is also why training-free spatial compression on the camera tokens
  barely moves L2 — there is little camera-only planning signal to lose.)
- The value of this retrain is **methodological**, not a new SOTA: it makes the Qwen3-VL-4B 3-cam
  result **apples-to-apples** with **B.5' Qwen2.5-VL-3B 3-cam (L2 0.617)** at the *same* 240 tok/cam
  AutoVLA budget. Clean cross-backbone parity: **Qwen3-4B 0.626 ≈ Qwen2.5-3B 0.617**.
- Resolution-by-design split stands: 1-cam B.5'' is deliberately NATIVE 2800 tok/cam (the spatial-
  compression deploy showcase, L2 0.715); 3-cam is 240 tok/cam (AutoVLA-faithful planning). They are
  NOT a resolution-controlled 1-vs-3-cam comparison.

## Deliverable status
- B.5''' AutoVLA-res ckpt + eval: DONE (this doc).
- Pending (#176): the deploy e2e per-precision bench currently quantizes only the ViT; the TRT-LLM
  backbone ran bf16 in all rows. Real fp8/nvfp4 LM quant + re-bench is the next deploy item.
