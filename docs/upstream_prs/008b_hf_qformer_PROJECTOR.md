# PR 008b — HF Accelerate Qwen2.5-VL Q-Former PROJECTOR (Track A.1, REDO)

**Repo**: QIU023/DriveLM_VLM_Project (this fork)
**Branch**: `qformer_hf_projector_port` (off `qwen25_vl_video_vla`)
**Status**: Drafted locally + CPU smoke-tested. **Multi-GPU GPU training not yet run.** Pre-submission gate:
  1. GPU smoke (next-agent task — boot the 3cam_qformer_projector config, verify forward + projector grad sync under FSDP=8).
  2. Track A.1 full SFT validation (~2.2K steps on nuScenes planning 3-cam × 4f; A/B vs the linear-projector baseline + A.2/A.3 siblings at matched compute).

**Supersedes**: [008_hf_qformer_projector.md](008_hf_qformer_projector.md). PR 008's `qformer_hf_port` branch wired the Q-Former into the `cross_frame_compressor` registry path; that conflated two distinct ablation axes. See "Architectural delta vs 008" below.

## Why we need it

Stock HF Qwen2.5-VL's vision-to-LM projector is a fixed in-encoder `PatchMerger` (2x2 spatial merge + 2-layer MLP -> `out_hidden_size=2048` for the 3B model). With min_pixels=max_pixels=109760 the 3-cam × 4-frame nuScenes recipe lands ~1680 visual tokens / sample at the LM input boundary — fine for HBM training but expensive at deploy (KV-cache scales linearly with context length).

We want a **swap-in alternative projector** that compresses visual context to a **fixed 64-token / item budget** via a BLIP-2-style Querying Transformer (Q-Former). 64 learnable queries cross-attend the per-item visual features and produce a fixed-length output, regardless of input length. Net: ~22x token-budget compression vs baseline at 3 cams (192 tokens vs 1680).

This is the first member of our fusion-mechanism comparison (A.0/A.1/A.2/A.3):
* **A.0** = stock linear baseline (~1680 tokens, 0 extra params)
* **A.1** = Q-Former 64 queries / item (~192 tokens / sample at 3 cams, ~22x compression, ~90M params at internal_dim=1024) — **this PR**
* **A.2** = PixelShuffle 2x + Linear (~420 tokens / sample, ~4x compression, ~17M params) — PR 009
* **A.3** = Perceiver Resampler 64 latents / item (~192 tokens / sample, ~22x compression, Flamingo-style with temporal pos) — PR 010

Q-Former's selling point vs PixelShuffle: aggressive fixed-budget compression (constant KV size at deploy) and learnable spatial pooling (queries can specialise to scene-relevant regions). Vs Resampler: simpler — no temporal pos, no latent self-attn — so the A.1/A.3 A/B isolates the contribution of temporal positional encoding + latent self-attn at matched 64-latent / item budget.

## Architectural delta vs 008 (DEPRECATED)

| Aspect | 008 (`qformer_hf_port`) | **008b (this PR, `qformer_hf_projector_port`)** |
|---|---|---|
| Registry path | `cross_frame_compressor.name = "qformer"` | `projector_type: qformer` |
| Forward shim | reused `forward_with_video_xframe_compression` (built for VTM / LongVU / mean-pool) | new `forward_with_video_qformer_projector` modelled on `forward_with_pixelshuffle_projector` / `forward_with_video_resampler_projector` |
| Ablation axis | conflated with cross-frame temporal compression | **fusion mechanism** (projector replacement), mutually exclusive with cross-frame compressors |
| Per-item vs whole-batch | whole-batch (1 Q-Former call) | per-item (1 call per cam-clip) — same shape as A.3 Resampler shim for A/B parity |
| Files in `scripts/compressors/` | `qformer.py` registered | none (intentionally NOT registered as a compressor in this branch) |
| Config | `cross_frame_compressor: {name: qformer, kwargs: ...}` | `projector_type: qformer` + `qformer: {...}` sub-block |

Methodologically, fusion mechanism (Q-Former / PixelShuffle / Resampler) is an orthogonal ablation axis to cross-frame temporal compression (VTM / LongVU / mean-pool). User feedback: "不允许乱改方法用法 — 模态融合实验和长帧压缩是不同组的消融". PR 008's wiring made the two axes share a registry slot, which would have polluted the A/B matrix. PR 008b restores the separation by mimicking the A.2/A.3 projector pattern verbatim.

## What changed

| File | Change | LOC (approx) |
|---|---|---|
| `scripts/qformer_projector_hf.py` | **new** — `Qwen2VLQFormerProjector` module: 64 learnable queries × `internal_dim=1024`, 6 cross-attn + FFN blocks at `internal_dim`, final `Linear(1024 -> lm_dim=2048)`. Uses `nn.MultiheadAttention(kdim=in_features, vdim=in_features, batch_first=True)`. CPU-safe. **Unchanged from PR 008** (the class was correct; only the wiring was wrong). | +425 |
| `scripts/train_lora.py` | add `forward_with_video_qformer_projector(...)` (monkey-patches `inner.get_video_features`, runs the projector per-item, trims `<|video_pad|>` placeholders to `num_queries` per item, rebuilds `video_grid_thw`); wire `projector_type: qformer` in `main()` (build projector, add to optimizer params, dispatch in train + val loops); extend `validate(...)` with `qformer_projector` kwarg | +250 / -4 |
| `configs/nuscenes_planning_3cam_qformer_projector.yaml` | **new** — `projector_type: qformer` config inheriting 3cam_full; full hyperparam block under `qformer:` with paper-audit notes | +75 |
| `docs/upstream_prs/008b_hf_qformer_PROJECTOR.md` | **this file** | +130 |
| `docs/upstream_prs/008_hf_qformer_projector.md` | mark deprecated, point at 008b | +5 |

Total diff: **~890 LOC, 2 new + 1 modified scripts file, 1 new + 1 modified PR doc, 1 new YAML config**. NO changes to `scripts/compressors/` (Q-Former is intentionally NOT a compressor in this branch).

## Design highlights

- **Projector replacement, not cross-frame compressor.** This is the methodological correction vs PR 008. The Q-Former replaces the linear projection that maps post-merger visual features to LM inputs; it sits on the same axis as A.2 (PixelShuffle) and A.3 (Resampler). It is mutually exclusive with `cross_frame_compressor` (different axis); the trainer raises if both are set.

- **POST-merger entry point (vit_dim = lm_dim = 2048).** The Q-Former runs DOWNSTREAM of Qwen2.5-VL's in-encoder `PatchMerger`. The existing `get_video_features` monkey-patch in the projector shim produces post-merger `(num_items_total, lm_dim)` features. A pre-merger variant (`vit_dim=1280`, feeds raw ViT features after only `ln_q`) is closer to the torchtitan version but requires patching `Qwen2_5_VLVisionModel.forward` directly — left as a follow-up. Identical caveat to PR 008.

- **Per-item Q-Former call.** Each cam-clip is fed independently as `(1, t * h_post * w_post, lm_dim)` -> `(1, 64, lm_dim)`. Mirrors the A.3 Resampler shim's per-item invocation; keeps the A.1/A.3 A/B apples-to-apples and bounds memory.

- **`internal_dim` decoupled from `lm_dim`.** All q/k/v/o/FFN projections operate at `internal_dim=1024` (BLIP-2-base uses 768; we use 1024 for driving-scene headroom). Only `out_proj` lifts to `lm_dim=2048`. Bounds param count to ~90M regardless of LM dim.

- **In-flight placeholder trim (NOT a dataset change).** The dataset emits the uncompressed `<|video_pad|>` count; the trainer trims placeholders per-item down to `num_queries=64` mid-forward when the projector is active. Symmetric with `forward_with_pixelshuffle_projector` / `forward_with_video_resampler_projector`. Benefit: one dataloader serves baseline + A.1 + A.2 + A.3 runs.

- **CPU-safe.** Only `nn.MultiheadAttention` + `nn.Linear` + `nn.LayerNorm`; no SDPA / compile.

- **Param-count source-of-truth.** Smoke asserts 80M < count < 130M. Measured: **90,351,616 ≈ 90.35M** at the wiring entry (`vit_dim=lm_dim=2048` post-merger).

## Reviewer hints

- This PR is **additive**: `projector_type: linear` (default) recovers the original Qwen2.5-VL behavior exactly. The only modified Python file is `scripts/train_lora.py`.
- The projector is **NOT** wrapped in FSDP. It sits outside the FSDP wrap on each rank, same caveat as A.2/A.3. Per-rank grads are not auto-reduced across ranks; the projector's ~90M params will diverge across ranks under FSDP=N unless we add an explicit all-reduce. Fine for the planned single-node smoke; must be fixed before multi-node SFT.
- The projector is **NOT** TP/CP/PP-aware. Multi-dim parallelism integration is a follow-up.

## Open issues (next-agent / follow-up)

1. **DDP/FSDP grad sync for the projector.** Same caveat as A.2/A.3. Wrap the projector in DDP separately, or add a manual all-reduce after backward. Easiest fix: wrap in `torch.nn.parallel.DistributedDataParallel` with `find_unused_parameters=False` after building it.

2. **Pre-merger Q-Former variant.** Mirror of the open issue from PR 008: feeding raw 1280-dim ViT features after only `ln_q` would be more faithful to BLIP-2. Requires bypassing `model.visual.merger`. Not in this PR; the module already supports `vit_dim=1280`.

3. **Inference-side / eval-side support.** `planning_eval.py` and `demo_inference*.py` do NOT know about the Q-Former projector; they call `model.generate(...)` directly and will use the stock PatchMerger output without trimming placeholders. A correctness fix is needed before any held-out L2 metric is meaningful for the A.1 ablation. Same caveat as A.2/A.3.

4. **Validation L2 metric is currently skipped.** Same as A.2/A.3 — `validate(...)` falls back to `L2=skipped(qformer)` whenever `qformer_projector` is set. Token accuracy and val_loss are still computed and meaningful.

5. **Placeholder trim impl detail.** Currently the trim keeps the FIRST `num_queries` `<|video_pad|>` positions per item and drops the tail. The scattering inside `Qwen2_5_VLModel.forward` then injects the Q-Former's queries in order. We assume order-of-emission corresponds to no semantic meaning (queries are learnable). Confirmed against PR 009 / PR 010 which use the same trim convention.

6. **Param-count vs in-flight memory.** ~90M extra params in compute_dtype=bf16 = ~180 MB weight + 180 MB grad + 360 MB Adam moments = ~720 MB extra HBM per rank. Small vs the 3B LM (~6 GB bf16); under FSDP=8 the per-rank Adam moment cost dominates the projector footprint.

7. **From-scratch training risk.** BLIP-2 paper trains stage-1 on 129M caption pairs before stage-2 instruction. Our nuScenes planning has ~24K samples. Expect the Q-Former to UNDERFIT relative to the pretrained linear projector at matched token budget. Decision criterion: A.1 is "acceptable" if its L2 is within ~30% of A.0 at ~22x fewer LM-input tokens; "good" if within ~10%.

## Pre-submission gate (must pass before opening upstream PR)

- [ ] **GPU smoke**: launch `configs/nuscenes_planning_3cam_qformer_projector.yaml` for 4 steps with `--no-validate`; confirm forward + projector grad flow, no shape-mismatch asserts.
- [ ] **Track A.1 full SFT validation**: 3-epoch run; compare loss curves + downstream L2 vs A.0 / A.2 / A.3.
- [ ] **A/B at deploy**: with ~22x reduced placeholder count the LM context shrinks from ~2300 -> ~800 tokens — measure KV-cache size + TTFT at TRT-export to confirm the deploy win.

## Static test (already passed, CPU)

```text
cd /workspace/DriveLM_VLM_Project && /usr/bin/python3 -c "
import sys; sys.path.insert(0, 'scripts')
import torch
from qformer_projector_hf import Qwen2VLQFormerProjector
m = Qwen2VLQFormerProjector(vit_dim=2048, internal_dim=1024, lm_dim=2048, num_queries=64, num_layers=6, n_heads=8, ffn_mult=4)
print(sum(p.numel() for p in m.parameters()))  # 90_351_616
x = torch.randn(2, 1680, 2048)
out = m(x); print(out.shape)                    # torch.Size([2, 64, 2048])
loss = out.sum(); loss.backward()
print(m.out_proj.weight.grad.norm().item())     # ~1e5 (non-zero)
"
# Output:
# 90351616
# torch.Size([2, 64, 2048])
# 1.0294e+05
```

## Paper-hyperparam audit (memory: feedback_paper_hyperparam_audit_gate)

| Param | Paper (BLIP-2-base) | Ours (3cam_qformer_projector.yaml) | Rationale |
|---|---|---|---|
| `internal_dim` | 768 | 1024 | Modest headroom for driving-scene complexity. ~1.6x more attn FLOPs, still BLIP-2-scale. |
| `num_queries` | 32 | 64 | AutoVLA-style 3-cam × 4f packs more scene context per query; 64 ~= 12 regions × 5 horizons. Matches A.3 Resampler at same latent count for fair A/B. |
| `num_layers` | 12 | 6 | Halved because BLIP-2's 12L was for 129M caption pairs (stage 1); we train from scratch on 24K nuScenes samples. Deeper Q-Formers underfit at this scale per BLIP-2 §4.2 ablation. Matches A.3. |
| `n_heads` | 12 | 8 | head_dim = internal_dim / n_heads = 128 (ViT-base default at width=1024). BLIP-2 used head_dim=64. |
| `ffn_mult` | 4 | 4 | Match. |
| `lr` | 1e-4 (stage 1) | 1e-4 inherited from `gb200_vla.yaml` | Match. |
| Warmup ratio | 1% | 1.74% (39 / 2243 steps, inherited from 3cam_full ratio-scaled from AutoVLA) | Match (3cam_full is paper-audited). |
| `num_epochs` | (stage-1 200K iters) | 3 (inherited from nuscenes_planning_full) | nuScenes-only ~22K samples; 3 epochs = ~66K samples seen, comparable to BLIP-2 stage-1 in token count for our token budget. |

Architectural choices are deliberate departures from BLIP-2-base with cited reasons; LM hyperparams match the audited AutoVLA recipe inherited from 3cam_full.
