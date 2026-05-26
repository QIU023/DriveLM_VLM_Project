# Measured FLOPs / BOPs / roofline — B.5'' 1-cam deploy (Qwen3-VL-4B, 2026-05-26)

Answers "are there FLOP stats per precision?" + "is bs=1 the right scenario?".
Measured with `torch.profiler(with_flops=True)` on a REAL 1-cam multimodal sample
(`deploy/trt_b5ppp/profile_flops.py`); roofline on RTX 5090 (BW 1.79 TB/s).

## Key conceptual point
**FLOPs (MAC count) are precision-INDEPENDENT** — bf16/fp8/fp4 execute the same number of
multiply-adds. What changes is **BOPs = FLOPs × bit-width** and achieved TOPS. For bs=1 onboard
decode the real bottleneck is **weight memory traffic (bytes), not FLOPs**.

## Measured FLOPs (one frame, raw HF model, prompt=3507 = 2800 video tok uncompressed)
| stage | FLOPs | note |
|---|---|---|
| full forward = ViT + 3507-tok prefill | **35.9 TFLOP** | ViT (11200 patches) dominates |
| 1 decode step | **8.05 GFLOP** | matches analytical 8.04 (validated) |
| 14-tok trajectory decode | 112.6 GFLOP | |

(Deploy compresses video 2800→700 via FasterVLM×4, so deploy prefill is ~1.4k tok < 3507; ViT FLOPs unchanged.)

## FLOP-vs-latency paradox
ViT is ~80% of the FLOPs but only ~51–75 ms of latency (big dense matmul, high arithmetic
intensity, runs near the compute roofline). LM decode is <1% of FLOPs but ~40–80 ms (tiny matmuls,
**memory-bandwidth bound**). **FLOPs ≠ latency** — this is why quantizing the LM (not the FLOP-heavy
ViT) is what cuts latency.

## BOPs per precision (FLOPs identical; bit-width differs)
| precision | full-fwd BOPs | 14-tok decode BOPs |
|---|---|---|
| bf16 | 574.6 TBOP | 1802 GBOP |
| fp8  | 287.3 TBOP | 901 GBOP |
| fp4  | 143.7 TBOP | 451 GBOP |

## bs=1 decode roofline (ridge bf16 = 117 FLOP/byte)
| prec | weights/tok | arith intensity | t_mem | t_compute | bound | 14-tok (theory) | measured |
|---|---|---|---|---|---|---|---|
| bf16 | 8.0 GB | 1.0 FLOP/B | 4.49 ms | 0.038 ms | **MEMORY** | 63 ms | 81 ms |
| fp8  | 4.0 GB | 2.0 FLOP/B | 2.24 ms | 0.019 ms | **MEMORY** | 31 ms | 41 ms |
| fp4  | 2.0 GB | 4.0 FLOP/B | 1.12 ms | 0.010 ms | **MEMORY** | 16 ms | 45 ms |

Arithmetic intensity (1–4) ≪ ridge (117) → **deeply memory-bound**. Compute time (0.01–0.04 ms)
is negligible. So decode latency ∝ weight BYTES: fp8 ≈ ½ bf16 (81→41 ms, matches). **fp4 theory 16 ms
but measured 45 ms** — TRT-LLM nvfp4 block-scale dequant overhead eats the bandwidth win at bs=1; fp4
buys footprint (−61%) not speed here. fp4's compute density (838 TFLOPS) only pays off at bs≫1 (cloud/
fleet replay), never reached onboard.

## bs=1 IS the correct onboard AD scenario
One ego vehicle, one sensor stream, one planning query per cycle — never batched across vehicles.
So bs=1 latency is the deployment metric, and the roofline above says quantization's onboard win is
**memory (footprint + decode bandwidth), not FLOPs/compute**. fp8 = best onboard pick.

## Real-time input recap (per frame)
camera CAM_FRONT 2 temporal frames @1600×900 → decode/resize/normalize/patchify →
`pixel_values_videos [11200,1536]` (~34 MB bf16) → ViT → 2800 tok → FasterVLM×4 → 700;
HD-map BEV raster → 121 tok; 3D bbox + ego → text; prompt ≈1k tok (deploy) → 14-tok trajectory.
