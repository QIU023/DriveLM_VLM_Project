# PR 009 — HF Accelerate Qwen2.5-VL PixelShuffle 2x + Linear projector (Track A.2)

**Repo**: QIU023/DriveLM_VLM_Project (this fork)
**Branch**: `pixelshuffle_hf_port` (off `qwen25_vl_video_vla`)
**Status**: Drafted locally + CPU smoke-tested. **Multi-GPU GPU smoke not yet run.** Pre-submission gate:
  1. GPU smoke (next-agent task — boot the 3cam_pixelshuffle config, verify forward + projector grad sync under FSDP=8).
  2. Track A.2 full SFT validation (~2.2K steps on nuScenes planning 3-cam × 4f; A/B vs the linear-projector baseline at matched compute).

## Why we need it

Stock HF Qwen2.5-VL's vision-to-LM projector is a fixed in-encoder `PatchMerger` (2x2 spatial merge -> 4x token reduction). With min_pixels=max_pixels=109760 the 3-cam × 4-frame nuScenes recipe lands ~1680 visual tokens / sample at the LM input boundary — fine for HBM training but expensive at deploy (KV-cache scales linearly with context length).

We want a **swap-in alternative projector** that compresses visual context **deterministically by a further 4x** by stacking a PixelShuffle 2x on top of the in-encoder PatchMerger. Output: ~420 visual tokens / sample (~105 / cam at 3 cams). Net stack = 16x raw-ViT-patch compression.

This is the third member of our fusion-mechanism comparison (A.0/A.1/A.2/A.3):
* **A.0** = stock linear baseline (~1680 tokens, 0 extra params)
* **A.1** = Q-Former 64 queries (~64 tokens / item, ~26x compression, ~117M params at internal_dim=1024)
* **A.2** = PixelShuffle 2x + Linear (~420 tokens / sample, 4x compression, ~16.78M params) — **this PR**
* **A.3** = Perceiver Resampler 64 latents (also ~64 tokens, Flamingo-style)

PixelShuffle's selling point vs Q-Former: deterministic compression, zero learnable queries, ~7x fewer params, no cross-attention softmax — i.e. a much cheaper "honest" baseline for the A.0-vs-{A.1,A.3} learnable-pool comparison.

## What changed

| File | Change | LOC (approx) |
|---|---|---|
| `scripts/pixelshuffle_projector_hf.py` | **new** — `Qwen2VLPixelShufflePlusLinearProjector` module: PixelUnshuffle 2x (channel-last) + Linear projection. CPU smoke `__main__` block. Mirrors torchtitan's `Qwen3VLPixelShufflePlusLinearProjector` design (SHA b627122 on the `pixelshuffle_projector` branch). | +280 |
| `scripts/train_lora.py` | add `forward_with_pixelshuffle_projector(...)` (monkey-patches `inner.get_video_features` to inject post-merger features through the projector, trims `<|video_pad|>` placeholders by `1 - 1/shuffle_unit`, halves `h_pre`/`w_pre` in `video_grid_thw`); wire `projector_type: pixelshuffle` in `main()` (build projector, add to optimizer params, dispatch in train + val loops); extend `validate(...)` with `pixelshuffle_projector` kwarg | +185 / -4 |
| `scripts/planning_dataset.py` | docstring-only — document the in-flight placeholder-trim contract for the pixelshuffle path | +20 |
| `configs/nuscenes_planning_3cam_pixelshuffle.yaml` | **new** — `projector_type: pixelshuffle` config inheriting 3cam_full; documents the stacked-projector design + token budget | +55 |
| `docs/upstream_prs/009_hf_pixelshuffle_projector.md` | **this file** | +90 |

Total diff: **~625 LOC, 3 new files, 2 modified files**.

## Design highlights

- **Switchable, default-off.** `projector_type` defaults to `"linear"`, so every existing config / checkpoint / smoke launcher works unchanged. The pixelshuffle path is opt-in via `projector_type: pixelshuffle` in YAML.

- **Stacked projector (= 16x raw-patch compression).** The in-encoder `PatchMerger` (2x2 spatial merge) runs first under `no_grad` (vision frozen, per `freeze_vision: true`), then the PixelShuffle 2x + Linear runs ON TOP of it. This mirrors the torchtitan path-(a) design. A path-(b) variant (bypass the in-encoder merger and consume raw 1152-dim ViT features) would be a follow-up — currently the model.visual API doesn't expose a `return_premerge` flag and bypassing PatchMerger requires module surgery.

- **In-flight placeholder trim (NOT a dataset change).** The dataset emits the uncompressed `<|video_pad|>` count produced by Qwen2.5-VL's processor; the trainer trims placeholders mid-forward when the projector is active. This is symmetric with `forward_with_video_xframe_compression` which already does the same trick for the cross-frame compressors. Benefit: one dataloader serves baseline + pixelshuffle + Q-Former + perceiver runs.

- **Per-item batching.** The forward function asserts identical `(t, h_post, w_post)` across all video items in the batch (which holds by construction in the 3-cam config since planning_cams iterate over fixed-resolution clips). The projector then operates on a stacked `(num_items, t*h*w, lm_dim)` tensor for one big matmul. Mixed-resolution batches would need grouping — flagged as a follow-up.

- **CPU-safe.** Only `nn.Linear` + reshape/permute; no SDPA, no attention, no compile. The smoke test in `pixelshuffle_projector_hf.__main__` runs on CPU in <1s.

- **Param-count source-of-truth.** Smoke test asserts 10M < count < 30M. At Qwen2.5-VL-3B's `lm_dim=2048` the measured count is **16,779,264 ≈ 16.78M** = `2048*4 * 2048 + 2048` — matches the agent spec's ~17M target.

## Reviewer hints

- This PR is **additive** for the HF training stack: `projector_type: linear` (default) recovers the original Qwen2.5-VL behavior exactly. The only modified Python files are `scripts/train_lora.py` (new function + 3 dispatch sites) and a docstring in `scripts/planning_dataset.py`.
- The projector is **NOT** wrapped in FSDP. It sits outside the FSDP wrap on each rank, same as the xframe compressor. Per-rank grads are not auto-reduced across ranks; the projector's ~16.78M params will diverge across ranks under FSDP=N unless we add an explicit all-reduce. This is fine for the planned single-node smoke but must be fixed before any production multi-node SFT.
- The projector is **NOT** TP/CP/PP-aware. Multi-dim parallelism integration is a follow-up.

## Open issues (next-agent / follow-up)

1. **DDP/FSDP grad sync for the projector.** As above. Either wrap the projector in DDP separately, or add a manual all-reduce after backward. Easiest fix: wrap in `torch.nn.parallel.DistributedDataParallel` with `find_unused_parameters=False` after building it.

2. **Pre-merger PixelShuffle variant.** Mirroring torchtitan's path-(b), feeding raw 1152-dim ViT features into PixelShuffle would give a much higher resolution post-shuffle map (8x raw-patch compression vs current 16x stacked). Requires bypassing `model.visual.merger`. Not in this PR.

3. **Inference-side / eval-side support.** `planning_eval.py` and `demo_inference*.py` do NOT know about the pixelshuffle projector; they call `model.generate(...)` directly and will use the stock PatchMerger output without trimming placeholders. A correctness fix is needed before any held-out L2 metric is meaningful for the A.2 ablation.

4. **Validation L2 metric is currently skipped.** Same as the xframe compressors — the L2 decode path in `validate(...)` assumes logits seq_len matches `batch["labels"]` seq_len. Under pixelshuffle the trainer trims placeholders, so the lengths diverge; we currently print `L2=skipped(pixelshuffle)`. Token accuracy and val_loss are still computed and meaningful.

5. **Param-count vs in-flight memory.** 16.78M extra params in compute_dtype=bf16 = ~33.6 MB weight + 33.6 MB grad + 67.2 MB Adam moments = ~134 MB extra HBM per rank. Negligible vs the 3B LM but worth noting.

## Pre-submission gate (must pass before opening upstream PR)

- [ ] **GPU smoke**: launch `configs/nuscenes_planning_3cam_pixelshuffle.yaml` for 4 steps with `--no-validate`; confirm forward pass + projector grad flow, no shape-mismatch asserts.
- [ ] **Track A.2 full SFT validation**: 3-epoch run; compare loss curves + downstream L2 vs the linear-projector baseline.
- [ ] **A/B at deploy**: with the trimmed placeholder count the LM context shrinks from ~2300 -> ~720 tokens — measure KV-cache size + TTFT at TRT-export to confirm the deploy win.

## Static test (already passed, CPU)

```text
cd /workspace/DriveLM_VLM_Project && /usr/bin/python3 scripts/pixelshuffle_projector_hf.py
# [smoke] params: 16,779,264 (16.78M)
# [smoke] forward OK: in=(2, 560, 2048) -> out=(2, 140, 2048) (4x compression)
# [smoke] backward OK: proj.weight grad_norm=1.400e-01
# [smoke] 3-cam-stack shape OK: 1680 -> 420 tokens (matches spec)
# [smoke] odd-H rejection OK: PixelShuffle ratio 2 requires post-merger h (5) and w (5) to be divisible by 2. ...
# [smoke] PASS
```
