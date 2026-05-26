# Tier-2 cached frozen-ViT vision tokens — 3-cam NATIVE planning SFT

Wires pre-cached, int8-quantized **frozen-ViT + FasterVLM x4** vision tokens into
the REAL `scripts/train_lora.py` full_sft path so a 3-cam NATIVE VLA planning SFT
trains **without running the ViT every step**. Frozen-vision only
(`freeze_vision=True` is the full_sft default — valid). Setup: 3-cam NATIVE
2800 tok/cam → FasterVLM x4 → 700 tok/cam (2100 video tokens to the LM) + 121
HD-map image tokens, config `configs/nuscenes_planning_3cam_qwen3vl_NATIVE.yaml`,
backbone Qwen3-VL-4B-Instruct.

## Token counts (verified on REAL emitted tensors, not flags)

| quantity | value |
|---|---|
| video_grid_thw per cam (native) | `[2, 56, 100]` → post-merge **2800 tok/cam** |
| video_pad before trim | 3 × 2800 = **8400** |
| FasterVLM x4 kept per cam | 2800 // 4 = **700 tok/cam** |
| rebuilt video_grid_thw per cam | `[1, 50, 56]` → 25×28 = **700** |
| **video_pad after trim** | 3 × 700 = **2100** (verified `n_video_pad=2100`) |
| HD-map image_grid_thw | `[1, 22, 22]` → **121** image tokens (kept, un-compressed) |
| **image_pad** | **121** (verified `n_image_pad=121`) |
| cached video pooler | `[2100, 2560]` |
| cached video deepstack | `[3, 2100, 2560]` (3 deepstack layers) |
| cached image pooler | `[121, 2560]` |
| cached image deepstack | `[3, 121, 2560]` |

## Parity gate (HARD)

`data_infra/lance/parity_check.py`, 8 real 3-cam native samples, loss A (LIVE
`forward_with_video_compression_free`, ViT + FasterVLM x4 + deepstack) vs loss B
(`forward_with_cached_vision_tokens` reading the cache built from the SAME
samples). Tolerance: mean `|A−B|/A < 0.03`.

Run against the production n=450 cache:

| idx | lossA (live) | lossB (cache) | rel |
|----:|---:|---:|---:|
| 0 | 14.71241 | 14.65183 | 0.412% |
| 1 | 14.38228 | 14.33770 | 0.310% |
| 2 | 14.22828 | 14.20214 | 0.184% |
| 3 | 14.23622 | 14.27523 | 0.274% |
| 4 | 14.51774 | 14.53587 | 0.125% |
| 5 | 14.46121 | 14.43138 | 0.206% |
| 6 | 14.38685 | 14.37750 | 0.065% |
| 7 | 14.12571 | 14.05174 | 0.524% |

**mean lossA = 14.38134, mean lossB = 14.35792, mean rel = 0.2624% < 3.00% → PASS.**

(losses ~14.4 because the bench uses the base Qwen3-VL-4B weights, untrained on
this task; what the gate validates is that A and B agree, which they do to 0.26%.)

int8 round-trip: per-tensor symmetric int8 has a high *mean-relative* error on the
deepstack tensors (~17% — dominated by many small-magnitude entries), but because
deepstack features enter the LM as an **additive residual at a few visual
positions** the loss impact is negligible (0.26%). Per-tensor int8 is sufficient
for this throughput demo; a per-channel scheme would tighten it further if needed.

## A/B training-step throughput (real fwd+bwd+opt, frozen vision)

`data_infra/lance/bench_train_throughput.py`, 30 timed steps (3 warmup), bs=1,
single GPU, SGD + gradient checkpointing (so each arm fits on one 32G GPU;
the production run uses 8-GPU FSDP — see note below). Each arm runs in its own
subprocess for clean GPU memory.

| measurement | LIVE (ViT every step) | CACHED (no ViT) | speedup | peak mem |
|---|---:|---:|---:|---:|
| **NATIVE 700 tok/cam, full fwd+bwd+opt** | 0.8989 s/step (1.11 it/s) | **0.6032 s/step (1.66 it/s)** | **1.49x** | 20.8 → 20.0 GB |
| NATIVE 700 tok/cam, forward-only | 0.4646 s/step | 0.1581 s/step | **2.94x** | 16.3 → 15.6 GB |
| REDUCED 240 tok/cam @524288, full fwd+bwd+opt | 0.2622 s/step | 0.2158 s/step | 1.22x | 18.9 → 18.9 GB |

- **Live native does NOT OOM** on a single GPU once you drop AdamW state (SGD) and
  enable gradient checkpointing (peak 20.8 GB). The earlier OOM was the full 4B
  AdamW optimizer state (~32 GB) + cross-arm fragmentation, not the native vision
  config. So `live_native_train_OOM = False`; both arms fit and the native
  full-train A/B is a clean **1.49x**. (The reduced-res both-arms-fit arm is also
  reported per the task's fallback request: 1.22x.)

## Per-step speedup ceiling (honest)

The ceiling for "skip the ViT" is `1 + ViT_cost / (LM_fwd + LM_bwd)`, i.e. you can
only remove the vision-tower portion of the step.

- **Forward-only** isolates the ViT+FasterVLM cost the cache removes:
  live 0.4646 − cached 0.1581 = **0.3065 s** of ViT/FasterVLM per step (66% of the
  live forward). Realized forward speedup **2.94x**.
- **Full fwd+bwd+opt** native: under gradient checkpointing the frozen ViT is run
  in forward AND recomputed in backward, so the cache saves both. Saved =
  0.8989 − 0.6032 = **0.2957 s/step**; the remaining 0.6032 s is the LM fwd+bwd +
  opt + deepstack injection that the cache cannot remove. Realized **1.49x**, i.e.
  the cache eliminated **33%** of the per-step wall time at native resolution.
- The native full-train speedup (1.49x) is larger than the reduced-res one (1.22x)
  exactly as expected: at 2800 tok/cam the ViT is a bigger share of the step than
  at the 240-tok/cam reduced resolution, so caching it pays off more.

## Cache on disk

| | |
|---|---|
| rows (default build) | 450 (3-cam NATIVE train subset, first 450 in order) |
| on-disk | **5.06 GB** (~11.2 MB/row; Lance compresses the int8 below the 22 MB raw) |
| budget | < 12 GB ✓ |
| path | `data_infra/lance/nusc_3cam_native.lance` |

(Per-row raw int8 = video pooler 2100×2560 + video deepstack 3×2100×2560 + image
pooler 121×2560 + image deepstack 3×121×2560 ≈ 22 MB; Lance encoding lands at
~11 MB/row.)

## Files

Producer / cache / bench (new, under `data_infra/lance/`):
- `native_cache_common.py` — shared trim + FasterVLM-index + int8 + deepstack-compress helpers (single source of truth for layout; used by live forward, cached forward, producer, bench).
- `cache_3cam_native.py` — 8-GPU-sharded producer.
- `cached_native_dataset.py` — `CachedNativeDataset` + `collate_cached`.
- `parity_check.py` — HARD parity gate.
- `bench_train_throughput.py` — subprocess-isolated A/B bench.

Edits to the REAL training script `scripts/train_lora.py`:
- `forward_with_video_compression_free` — **fixed for Qwen3-VL**: now carries
  compressed **deepstack** features (same FasterVLM top-K indices as the pooler),
  trims `mm_token_type_ids` in lockstep, and passes the HD-map image branch
  (`pixel_values`/`image_grid_thw`) through under no_grad. Previously crashed on
  Qwen3-VL (`_FakeVisOut` had no `deepstack_features`) and silently dropped both
  the HD-map vision features and `mm_token_type_ids` (M-RoPE). Trim logic now
  delegates to `native_cache_common.trim_native_layout`.
- `forward_with_cached_vision_tokens` — **new**: scatters int8-dequantized cached
  video + image pooler tokens into `inputs_embeds` and injects cached deepstack via
  monkey-patched `get_video_features`/`get_image_features` (ViT never runs), then
  calls `model(...)` through the normal multimodal path so scatter + deepstack +
  M-RoPE are byte-identical to the live path.
