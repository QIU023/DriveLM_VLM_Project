# PR 010 — HF Accelerate Qwen2.5-VL Perceiver Resampler projector (Track A.3)

**Repo**: QIU023/DriveLM_VLM_Project (this fork)
**Branch**: `resampler_hf_port` (off `qwen25_vl_video_vla`)
**Status**: Drafted locally + CPU smoke-tested. **Multi-GPU GPU smoke not yet run.** Pre-submission gate:
  1. GPU smoke: boot `configs/nuscenes_planning_3cam_resampler.yaml` under accelerate-launch FSDP=8, verify forward + backward complete without OOM at LBS=4.
  2. Param-count line `[resampler] Built projector ... 90.44M` should appear in the FSDP boot log.
  3. Track A.3 full SFT validation (~2.2K steps on 3-cam x 4f nuScenes planning); A/B vs A.0 (linear) and A.1 (Q-Former-64, no temporal pos) at matched compute. Hypothesis: A.3 closes >=0.05 m L2 vs A.1 due to the temporal-pos inductive bias.

## Why we need it

Stock HF Qwen2.5-VL's vision-to-LM projector is a fixed in-encoder `PatchMerger` (2x2 spatial merge -> 4x token reduction). With min_pixels=max_pixels=109760 the 3-cam x 4-frame nuScenes recipe lands ~1680 visual tokens / sample at the LM input boundary — a bag-of-patches view with no notion of which frame each token came from.

We want a **swap-in alternative projector** that:
  1. Compresses visual context to a **fixed budget of 64 latents per cam** (~192 LM-input tokens / sample at 3 cams; 8.75x compression).
  2. Makes the latents **frame-aware** via an explicit learnable per-frame positional encoding added to the cross-attention KV.

This is Track A.3 in our fusion-mechanism school comparison (A.0/A.1/A.2/A.3):
* **A.0** = stock linear baseline (~1680 tokens, 0 extra params)
* **A.1** = Q-Former-64 (BLIP-2-style; 64 tokens / cam, no temporal pos, ~117M params)
* **A.2** = PixelShuffle 2x + Linear (~420 tokens / sample, 4x compression, ~17M params)
* **A.3** = Perceiver Resampler-64 (Flamingo-style; 64 tokens / cam, **WITH temporal pos**, ~90M params) — **this PR**

The Perceiver Resampler's selling point vs the Q-Former (A.1): **per-frame temporal positional encoding** added to the cross-attention KV BEFORE each block. The latents see frame-distinguishable features rather than a bag of patches. For 3-cam x 4f autonomous-driving inputs where the 6-waypoint trajectory output is strongly correlated with motion direction / speed, this inductive bias should help disambiguate scenes that look spatially similar but temporally distinct (e.g. stationary at red-light vs slowly accelerating).

## What changed

| File | Change | LOC (approx) |
|---|---|---|
| `scripts/perceiver_resampler_projector_hf.py` | **new** — `Qwen2VLPerceiverResamplerProjector` module. 64 learnable latents at internal_dim=1024, 6 (latent_self_attn -> cross_attn(latents <- visual + temporal_pos) -> FFN) blocks, final `Linear(1024 -> lm_dim=2048)` output proj. Mirrors torchtitan commit `0026ea5` but with internal-dim decoupling (1.21B -> 90M). | +330 |
| `scripts/train_lora.py` | add `forward_with_video_resampler_projector(...)`; wire `projector_type: resampler` in `main()`; extend `validate(...)` with `resampler_projector` kwarg; route train + val dispatch | +260 / -6 |
| `scripts/planning_dataset.py` | extend `projector_type` valid values to include `"resampler"` and `"pixelshuffle"`; new `resampler_num_latents` field | +25 / -10 |
| `configs/nuscenes_planning_3cam_resampler.yaml` | **new** — `projector_type: resampler` config inheriting 3cam_full | +50 |
| `docs/upstream_prs/010_hf_perceiver_resampler_projector.md` | **this file** | +100 |

Total diff: **~775 LOC, 3 new files, 2 modified files**.

## Design highlights

- **Switchable, default-off.** `projector_type` defaults to `"linear"`, so every existing config / checkpoint / smoke launcher works unchanged. Opt-in via `projector_type: resampler` in YAML. Mutually exclusive with `pixelshuffle` and `cross_frame_compressor`.

- **Internal-dim decoupling.** The torchtitan reference runs the resampler at `lm_dim` (4096 for Qwen3-VL-8B), giving ~1.21B params (FFN-dominated). For Qwen2.5-VL-3B (lm_dim=2048) we DECOUPLE the internal width: latents, self-attn, cross-attn-Q, FFN at `internal_dim=1024`; cross-attn-KV input dim = `in_features=2048` (projected to 1024 by k_proj/v_proj); final `Linear(1024 -> 2048)` lifts back to lm_dim. **Measured param count: 90,441,728 ~= 90.44M** — squarely in the 80-110M spec target.

- **Per-cam temporal-pos with shared T policy.** `nn.Embedding(t_max=32, in_features)` added to KV BEFORE every cross-attn (added once up-front; per-block `norm_cross_kv` handles post-add normalization). For each visual item with grid `(t, h, w)`, the first `h*w` input tokens get `temporal_pos[0]`, next `h*w` get `temporal_pos[1]`, etc. The forward shim calls the projector PER CAM with its own grid_thw, so cam-A frame-0 and cam-B frame-0 BOTH get `temporal_pos[0]` (multi-cam temporal index resets per cam — matches torchtitan policy).

- **In-flight placeholder trim (NOT a dataset change).** The dataset emits the uncompressed `<|video_pad|>` count from Qwen2.5-VL's processor; the trainer trims to 64 per cam mid-forward. Symmetric with `forward_with_pixelshuffle_projector` and `forward_with_video_xframe_compression`. Benefit: one dataloader serves baseline + pixelshuffle + Q-Former + resampler runs.

- **Per-item batching.** The forward function asserts identical `(t, h_post, w_post)` across all video items (true by construction in the 3-cam config). The projector is called PER ITEM (not per LM-batch element) so each cam gets its own temporal-pos lookup with T starting at 0. Simplest way to enforce the "cam-A f0 == cam-B f0" shared-T policy.

- **CPU-safe.** Pure `nn.Linear` + `F.scaled_dot_product_attention` (CPU falls back to math-impl). Smoke test runs in <2 s.

- **Param-count source-of-truth.** Smoke test asserts 80M < count < 130M. At lm_dim=2048, internal_dim=1024 the measured count is **90,441,728**. Breakdown:
  - Latents: 64 * 1024 = 65,536
  - Temporal pos: 32 * 2048 = 65,536
  - Per block (6 blocks): self-attn ~4.2M + cross-attn ~6.3M + FFN ~4.2M + LNs ~12K ~= 14.7M
  - 6 blocks: ~88.2M
  - Final LN + out_proj: ~2.1M
  - Sum: ~90.4M

## CPU smoke test

```
python3 -c "
import sys; sys.path.insert(0, 'scripts')
import torch
from perceiver_resampler_projector_hf import Qwen2VLPerceiverResamplerProjector
torch.manual_seed(0)
proj = Qwen2VLPerceiverResamplerProjector(
    in_features=2048, lm_dim=2048, internal_dim=1024,
    num_latents=64, num_layers=6, n_heads=8, ffn_mult=2, t_max=32,
).eval()
n = sum(p.numel() for p in proj.parameters())
assert 80e6 < n < 130e6, n
print(f'param_count = {n/1e6:.2f}M')
vision = torch.randn(2, 1680, 2048)
grid_thw = torch.tensor([[4,10,14],[4,10,14],[4,10,14]], dtype=torch.long)
out = proj(vision, grid_thw=grid_thw)
assert out.shape == (2, 64, 2048), out.shape
print('SMOKE OK')
"
# Output: param_count = 90.44M ; SMOKE OK
```

All 7 sub-tests pass: param-count, production-shape forward, temporal-pos read-back, grid-thw sensitivity, forward_xframe adapter, target_tokens API, masked-forward.

## Reviewer hints

- **Additive PR.** `projector_type: linear` (default) recovers the original Qwen2.5-VL behavior exactly. Modified Python: `scripts/train_lora.py` (new function + 3 dispatch sites + 1 validate kwarg) and `scripts/planning_dataset.py` (projector_type enum + new arg).
- **Not FSDP-wrapped.** Projector sits outside FSDP, same as xframe + pixelshuffle siblings. ~90M params will diverge across ranks under FSDP=N unless we add explicit all-reduce. OK for single-node smoke; must be fixed before multi-node SFT.
- **Not TP/CP/PP-aware.** Multi-dim parallelism integration is a follow-up.
- The 192-placeholder budget (3 cams x 64 latents) is dynamically derived from `projector.num_latents`, so changing `resampler.num_latents` in YAML doesn't require any other code changes.

## Open issues (next-agent / follow-up)

1. **DDP/FSDP grad sync for the projector.** Same as siblings. Fix: wrap in `torch.nn.parallel.DistributedDataParallel` with `find_unused_parameters=False`, or manual all-reduce after backward.

2. **Cross-cam temporal-pos policy.** Currently per-cam-resets-to-0. Alternative: per-cam offset (cam-B f0 == `temporal_pos[4]` for 4-frame clips), letting the model distinguish CAMERA per frame. Defer to ablation.

3. **Pre-merger variant.** Currently consumes post-merger features (in_features=lm_dim=2048). Pre-merger (in_features=1152 raw ViT hidden) would expose 4x more spatial tokens but requires bypassing `model.visual.merger`. Not in this PR.

4. **Latent init scheme.** Currently `N(0, 0.02)` on both `self.latents` and `self.temporal_pos.weight`. Flamingo uses orthogonal init for latents. Ablatable.

5. **Loss-mask interaction.** Trim removes placeholder tokens from labels too (alignment preserved). But teacher-forced L2 is skipped in val when resampler is active (`l2_skipped=True`), because logits seq_len after trim doesn't align with raw labels. Plan: re-enable L2 by re-running trim on labels' GT-bin positions. Non-blocking; val_loss + val_acc still meaningful.

6. **Param-count sensitivity to in_features.** Current 90M target assumes lm_dim=2048 (3B). For 7B (lm_dim=3584) the count jumps to ~190M; YAML allows overriding internal_dim per-experiment.

## Pre-submission gate

1. [ ] GPU smoke: boot `configs/nuscenes_planning_3cam_resampler.yaml` under accelerate-launch FSDP=8.
2. [ ] Param-count sanity: `[resampler] Built projector ... 90.44M` line in boot log.
3. [ ] Track A.3 full SFT (~2.2K steps); A/B vs A.0 + A.1 at matched compute.
4. [ ] DDP wrap or all-reduce shim for the projector.
