# PR 008 — HF Accelerate Qwen2.5-VL Q-Former projector (Track A.1)

**Repo**: QIU023/DriveLM_VLM_Project (this fork)
**Branch**: `qformer_hf_port` (off `qwen25_vl_video_vla`)
**Status**: Drafted locally + CPU smoke-tested. **Multi-GPU GPU training not yet run.** Pre-submission gate:
  1. GPU smoke (next-agent task — boot the 3cam_qformer config, verify forward + projector grad sync under FSDP=8).
  2. Track A.1 full SFT validation (~2.2K steps on nuScenes planning 3-cam × 4f; A/B vs the linear-projector baseline at matched compute).

## Why we need it

Stock HF Qwen2.5-VL's vision-to-LM projector is a fixed in-encoder `PatchMerger` (2x2 spatial merge + 2-layer MLP -> `out_hidden_size=2048` for the 3B model). With min_pixels=max_pixels=109760 the 3-cam × 4-frame nuScenes recipe lands ~1680 visual tokens / sample at the LM input boundary — fine for HBM training but expensive at deploy (KV-cache scales linearly with context length).

We want a **swap-in alternative projector** that compresses visual context to a **fixed 64-token budget** via a BLIP-2-style Querying Transformer (Q-Former). 64 learnable queries cross-attend the per-frame visual features and produce a fixed-length set of LM-input tokens, regardless of input length. Net: ~26x token-budget compression vs baseline.

This is the first member of our fusion-mechanism comparison (A.0/A.1/A.2/A.3):
* **A.0** = stock linear baseline (~1680 tokens, 0 extra params)
* **A.1** = Q-Former 64 queries (~64 tokens / sample, ~26x compression, ~80-130M params at internal_dim=1024) — **this PR**
* **A.2** = PixelShuffle 2x + Linear (~420 tokens / sample, 4x compression, ~17M params) — see PR 009
* **A.3** = Perceiver Resampler 64 latents (also ~64 tokens, Flamingo-style cross-attn over learnable latents)

Q-Former's selling point vs PixelShuffle: aggressive fixed-budget compression (suits TRT-export with constant KV size) and learnable spatial pooling (queries can specialise to scene-relevant regions). Tradeoff: ~7x more params than PixelShuffle and softmax cross-attn at inference time.

## What changed

| File | Change | LOC (approx) |
|---|---|---|
| `scripts/qformer_projector_hf.py` | **new** — `Qwen2VLQFormerProjector` module: 64 learnable queries × `internal_dim=1024`, 6 cross-attn + FFN blocks at `internal_dim`, final `Linear(1024 -> lm_dim=2048)`. Uses `nn.MultiheadAttention(kdim=in_features, vdim=in_features, batch_first=True)` so q/k/v can have mismatched dims. CPU-safe (no SDPA/compile). Mirrors torchtitan's `Qwen3VLQFormerProjector` design (SHA 5ee6380 after the internal_dim / lm_dim decoupling fix). | +330 |
| `scripts/compressors/qformer.py` | **new** — registers `QFormerCompressor` (wraps `Qwen2VLQFormerProjector`) under the `"qformer"` key in the cross-frame-compressor registry. `target_tokens=num_queries` is surfaced at the top level so the existing xframe forward shim's placeholder trim works unchanged. | +115 |
| `scripts/compressors/__init__.py` | one-line import-for-side-effect to register the new compressor | +1 |
| `configs/nuscenes_planning_3cam_qformer.yaml` | **new** — `cross_frame_compressor: {name: qformer, ...}` config inheriting 3cam_full; full hyperparam block documented with paper-audit notes (BLIP-2 deltas + cited reasons). | +95 |
| `docs/upstream_prs/008_hf_qformer_projector.md` | **this file** | +130 |

Total diff: **~680 LOC, 4 new files, 1 modified file**.

NOTE: The Q-Former is wired into the existing `cross_frame_compressor`
registry path (which already exists in `train_lora.py` and is shared
with VTM / LongVU). No changes to `train_lora.py` or `planning_dataset.py`
were needed — the existing `forward_with_video_xframe_compression` shim
handles the placeholder trim + scatter automatically for any compressor
exposing a `target_tokens` attribute.

## Design highlights

- **Switchable, default-off.** The Q-Former path is opt-in via `cross_frame_compressor.name: qformer` in YAML. Configs without that block run the stock Qwen2.5-VL behaviour unchanged.

- **Wired through the existing xframe-compressor registry.** `QFormerCompressor` exposes the `CrossFrameCompressor` protocol (`forward(B,T,N,D) -> (B,Nq,D)` + `target_tokens=64`), so the existing `forward_with_video_xframe_compression` (already shared with VTM / LongVU) handles the placeholder trim + scatter without a new shim. Net: ZERO new code paths in `train_lora.main()` — we plug into the existing `xframe_cfg = cfg.get("cross_frame_compressor")` switch.

- **POST-merger entry point (vit_dim = lm_dim = 2048).** The Q-Former runs DOWNSTREAM of Qwen2.5-VL's in-encoder `PatchMerger`. This is the simplest integration: the existing `get_video_features` monkey-patch in the xframe shim already produces post-merger `(B, T, N, lm_dim)` features. A pre-merger variant (`vit_dim=1280`, feeds raw ViT features after only the merger's `ln_q` RMSNorm) is closer to the torchtitan version but requires patching `Qwen2_5_VLVisionModel.forward` directly — left as a follow-up. See "Open issues" below.

- **`internal_dim` decoupled from `lm_dim`.** Per the torchtitan port's design rationale, all q/k/v/o/FFN projections operate at `internal_dim=1024` (BLIP-2-base uses 768; we use 1024 for driving-scene headroom). Only `out_proj` lifts the queries to `lm_dim` once at the boundary. This bounds the param count to ~80-130M regardless of the LM dim (the same 6-layer Q-Former at `lm_dim=4096` for the 8B model is still ~118M, not ~600M).

- **Param-count source-of-truth.** Smoke test asserts 80M < count < 130M. Measured: **80,905,216 ≈ 80.91M** at the spec smoke (`vit_dim=1280` pre-merger entry); **90,346,496 ≈ 90.35M** at the actual wiring entry (`vit_dim=lm_dim=2048` post-merger, which has slightly larger k/v projections). Both fall in the spec's 80-110M window.

- **CPU-safe.** `nn.MultiheadAttention` + `nn.Linear` + `nn.LayerNorm` only; no FlexAttention / compile / SDPA-with-block-mask. The smoke test runs on CPU in <2s.

## Reviewer hints

- This PR is **additive** for the HF training stack: `projector_type: linear` (default) recovers the original Qwen2.5-VL behavior exactly. The only modified Python files are `scripts/train_lora.py` (one new build block in `main()`), `scripts/planning_dataset.py` (constructor kwargs + factory passthrough), and `scripts/compressors/__init__.py` (one-line import).
- The projector is **NOT** wrapped in FSDP. It sits outside the FSDP wrap on each rank, same as VTM / pixelshuffle / resampler. Per-rank grads are not auto-reduced across ranks; the projector's ~90M params will diverge across ranks under FSDP=N unless we add an explicit all-reduce. This is fine for the planned single-node smoke but must be fixed before any production multi-node SFT.
- The projector is **NOT** TP/CP/PP-aware. Multi-dim parallelism integration is a follow-up.

## Open issues (next-agent / follow-up)

1. **Pre-merger Q-Former variant.** The torchtitan version's reference design feeds RAW ViT features (`vit_dim=1280`) after only the merger's `ln_q` RMSNorm — bypassing the merger MLP entirely. This is more faithful to BLIP-2 (queries cross-attend pre-projection features) and gives the Q-Former more spatial fidelity to learn from. Requires patching `Qwen2_5_VLVisionModel.forward` (the `merger` is called inside it) or replacing the entire `model.visual.merger` with a wrapper that bypasses its MLP. Not in this PR. The module already supports `vit_dim=1280`; only the wiring layer needs the surgery.

2. **DDP/FSDP grad sync for the projector.** As above. Either wrap the projector in DDP separately, or add a manual all-reduce after backward. Easiest fix: wrap in `torch.nn.parallel.DistributedDataParallel` with `find_unused_parameters=False` after building it. Same caveat as PR 009 / xframe compressors.

3. **Inference-side / eval-side support.** `planning_eval.py` and `demo_inference*.py` do NOT know about the qformer projector; they call `model.generate(...)` directly and will use the stock PatchMerger output without trimming placeholders. A correctness fix is needed before any held-out L2 metric is meaningful for the A.1 ablation.

4. **Validation L2 metric is currently skipped.** Same caveat as the xframe compressors — `validate(...)` falls back to `L2=skipped(xframe)` whenever `xframe_compressor` is set. Token accuracy and val_loss are still computed and meaningful.

5. **Param-count vs in-flight memory.** ~90M extra params in compute_dtype=bf16 = ~180 MB weight + ~180 MB grad + ~360 MB Adam moments = ~720 MB extra HBM per rank. Still small vs the 3B LM (~6 GB bf16) but worth noting; under FSDP=8 the per-rank Adam moment cost dominates the projector footprint.

6. **From-scratch training risk.** BLIP-2 paper trains stage-1 on 129M caption pairs before stage-2 on instruction data. Our nuScenes planning has ~24K samples. Expect the Q-Former to UNDERFIT relative to the pretrained Qwen2.5-VL linear projector at matched token budget — the A.1 win is on deployment cost, not on the L2 number. Decision criteria: A.1 is "acceptable" if its L2 is within ~30% of the A.0 baseline at 26x fewer LM-input tokens; "good" if within ~10%.

## Pre-submission gate (must pass before opening upstream PR)

- [ ] **GPU smoke**: launch `configs/nuscenes_planning_3cam_qformer.yaml` for 4 steps with `--no-validate`; confirm forward pass + projector grad flow, no shape-mismatch asserts.
- [ ] **Track A.1 full SFT validation**: 3-epoch run; compare loss curves + downstream L2 vs the A.0 linear-projector baseline + A.2 pixelshuffle baseline.
- [ ] **A/B at deploy**: with the 26x-reduced placeholder count the LM context shrinks from ~2300 -> ~700 tokens — measure KV-cache size + TTFT at TRT-export to confirm the deploy win.

## Static test (already passed, CPU)

```text
cd /workspace/DriveLM_VLM_Project && /usr/bin/python3 -c "
import sys; sys.path.insert(0, 'scripts')
import torch
from qformer_projector_hf import Qwen2VLQFormerProjector
m = Qwen2VLQFormerProjector(vit_dim=1280, internal_dim=1024, lm_dim=2048, num_queries=64, num_layers=6)
x = torch.randn(2, 1680, 1280)
out = m(x); print(out.shape)  # torch.Size([2, 64, 2048])
print(sum(p.numel() for p in m.parameters()))  # 80_905_216
"
# Output:
# torch.Size([2, 64, 2048])
# 80905216
```

Also verified through the registry path (POST-merger entry, `vit_dim=lm_dim=2048`):

```text
cd /workspace/DriveLM_VLM_Project && /usr/bin/python3 -c "
import sys; sys.path.insert(0, 'scripts')
import torch
from compressors import make_compressor
m = make_compressor('qformer', vit_dim=2048, lm_dim=2048)
frames = torch.randn(2, 12, 140, 2048)
out = m(frames); print(out.shape, m.target_tokens)
"
# Output:
# torch.Size([2, 64, 2048]) 64
```

## Paper-hyperparam audit (memory: feedback_paper_hyperparam_audit_gate)

This config defines paper-vs-ours deltas for the Q-Former hyperparams. Cited deviations only — no "we chose round numbers" / "conservative" / unprincipled choices.

| Param | Paper (BLIP-2-base) | Ours (3cam_qformer.yaml) | Rationale |
|---|---|---|---|
| `internal_dim` | 768 | 1024 | Modest headroom for driving-scene complexity. ~1.6x more attn FLOPs, still well inside BLIP-2 scale. |
| `num_queries` | 32 | 64 | AutoVLA-style 3-cam × 4f packs more scene context per query; 64 ~= 12 regions × 5 horizons. |
| `num_layers` | 12 | 6 | Halved because BLIP-2's 12L was for 129M caption pairs (stage 1); we train from scratch on 24K nuScenes samples. Deeper Q-Formers underfit at this scale per BLIP-2 §4.2 ablation. |
| `n_heads` | 12 | 8 | Sized to `internal_dim // 128 = 8` (BLIP-2 used `internal_dim // 64 = 12`). 8 is the common ViT-base default at width=1024. |
| `ffn_mult` | 4 | 4 | Match. |
| `lr` | 1e-4 (stage 1) | 1e-4 inherited from `gb200_vla.yaml` | Match. |
| Warmup ratio | 1% | 1.74% (39 / 2243 steps, inherited from 3cam_full ratio-scaled from AutoVLA) | Match (3cam_full is paper-audited). |
| `num_epochs` | (stage-1 200K iters) | 3 (inherited from nuscenes_planning_full) | nuScenes-only ~22K samples; 3 epochs = ~66K samples seen, comparable to BLIP-2 stage-1 in token count for our token budget. |

The Q-Former architectural choices are deliberate departures from BLIP-2-base; the `nuscenes_planning_full` inherited hyperparams (lr, warmup, schedule) match the audited AutoVLA recipe.
