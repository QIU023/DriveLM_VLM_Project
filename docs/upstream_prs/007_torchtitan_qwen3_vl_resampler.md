# PR 007 -- torchtitan Qwen3-VL Perceiver Resampler projector (Flamingo-style, temporal-aware)

**Repo**: pytorch/torchtitan (potential upstream PR)
**Branch**: `perceiver_resampler_projector` (off `qwen25_vl_video_vla` on QIU023/DriveLM_VLM_Project fork; the torchtitan worktree branches off the upstream submodule's `step_decay_lr_scheduler` tip with no Q-Former / PixelShuffle agent edits applied)
**Status**: Drafted locally + unit-tested on CPU. **PR not yet opened.** Pre-submission gate:
  1. GPU smoke (next-agent task -- boot the 8B-perceiver-resampler factory, verify forward pass + FSDP shard).
  2. Track A.3 full SFT validation (~10K steps on nuScenes planning; A/B vs the linear-projector baseline AND the Q-Former sibling at matched 64-token visual budget).

## Why we need it

This PR adds a **third projector option** to the Qwen3-VL projector school comparison:

  * **A.0 = Linear** (Qwen native; stock PatchMerger; pretrained alignment baseline).
  * **A.1 = Q-Former-64** (BLIP-2-style; cross-attn pool over flat token set; PR 005).
  * **A.2 = PixelShuffle 2x + Linear** (more aggressive spatial compression; sibling PR).
  * **A.3 = Perceiver Resampler-64** (THIS PR; Flamingo-style, **temporal-aware**).

The architectural distinction vs Q-Former is concrete:

| Aspect                  | Q-Former (BLIP-2)                       | Perceiver Resampler (Flamingo)                |
|---                      |---                                       |---                                             |
| Per-block layer order   | self-attn -> cross-attn -> FFN          | (latent self-attn -> cross-attn -> FFN) x N    |
| Self-attn on queries    | yes                                      | yes (between latents)                         |
| Pos encoding on queries | learnable                                | learnable on latents + **temporal pos on KV** |
| Origin                  | BLIP-2                                   | Flamingo                                       |
| Typical query count     | 32-128                                   | 64                                             |
| Temporal aware          | no (single image set)                    | **yes** (per-frame temporal-pos embedding)    |

Our implementation matches the Flamingo paper's Sec. 3.1 design: per-block pre-LN
`(latent_self_attn -> cross_attn(latents <- kv + temporal_pos) -> FFN)`, with a
learnable `nn.Embedding(T_max=32, in_features)` table added to KV-side inputs
*once up-front* (so every block sees the same temporally-augmented KV; the
per-block `norm_cross_kv` then normalizes after the add).

For our 3-cam x 4-frame nuScenes planning task the hypothesis is that
**explicit temporal positional encoding gives Resampler an inductive-bias edge
over Q-Former** on multi-frame inputs. The hypothesis is testable end-to-end via
Track A.3 SFT validation; this PR is only the module + factory + TOML + tests.

Token budget for 3-cam x 4f, matching the Q-Former sibling:

  * Linear (A.0):                ~1680 tokens at LM input.
  * Q-Former-64 (A.1):              64 tokens, no temporal pos.
  * PixelShuffle (A.2):           ~420 tokens.
  * Perceiver Resampler-64 (A.3):   64 tokens, **WITH temporal pos**.

## What changed

| File | Change | LOC (approx) |
|---|---|---|
| `torchtitan_qwen25/torchtitan/models/qwen3_vl/perceiver_resampler_projector.py` | **new** -- `Qwen3VLPerceiverResamplerProjector` module + supporting `_ResamplerSelfAttention` / `_ResamplerCrossAttention` / `_ResamplerFFN` / `_ResamplerBlock` building blocks + `_compute_temporal_indices` helper; full Configurable wiring; CPU-safe SDPA attention path with optional KV padding mask; learnable temporal-pos embedding | +470 |
| `torchtitan_qwen25/torchtitan/models/qwen3_vl/__init__.py` | add `Qwen3VLPerceiverResamplerProjector` import + export; new `_vl_resampler_config(...)` helper with self/cross/FFN Linear configs; new `_8b_perceiver_resampler()` config factory; registered as `"8B-perceiver-resampler"` flavor; `dataclasses` import | +85 |
| `torchtitan_qwen25/torchtitan/models/qwen3_vl/model.py` | add `projector_type: str = "linear"` and `perceiver_resampler: Qwen3VLPerceiverResamplerProjector.Config | None = None` to `Qwen3VLModel.Config`; switch in `__init__` to instantiate `self.perceiver_resampler_projector` when `projector_type == "perceiver_resampler"`; raises on bad value | +40 |
| `torchtitan_qwen25/tests/unit_tests/test_perceiver_resampler_projector.py` | **new** -- 8-case CPU unit test: production-shape (B=2, 1680, 1152) -> (B=2, 64, 4096); multi-cam split-grid; T=4 vs T=8 forward acceptance; temporal-pos read-back assertion (same KV different grid -> different output); padding-mask shape + finiteness; gradient flow on tiny config; num_latents invariance vs input length; param-count ballpark guard | +230 |
| `scripts_titan/train_titan_qwen3_vl.py` | add `model_flavor: str = "8B"` parameter to `_build_trainer_config`; new `qwen3_vl_8b_planning_fsdp_3cam_resampler()` factory using `model_flavor="8B-perceiver-resampler"`; smoke loop covers the new factory + prints projector_type | +75 |
| `scripts_titan/configs/qwen3_vl_8b_planning_fsdp_3cam_resampler.toml` | **new** -- full hyperparam manifest for the Resampler variant with rationale, `[model.perceiver_resampler]` section, and dataset key rename | +145 |
| `docs/upstream_prs/007_torchtitan_qwen3_vl_resampler.md` | **this file** | +145 |

Total diff: **~1190 LOC, 3 new files, 4 modified files.**

## Design highlights

- **Switchable, default-off.** `projector_type` defaults to `"linear"`, so every existing factory / TOML / checkpoint keeps working unchanged. The Resampler is opt-in via either the `"8B-perceiver-resampler"` flavor or by setting `projector_type="perceiver_resampler"` + populating `perceiver_resampler=...` on a custom config.

- **Direct LM-dim output.** Resampler emits `(B, num_latents, lm_dim)` -- no extra linear projector needed downstream. Final output `LayerNorm` keeps activations bounded; tested for finiteness on bf16-friendly random inputs.

- **Temporal-pos is a first-class input.** `nn.Embedding(T_max, in_features)` is added to KV inputs based on a per-token frame index derived from `grid_thw`. The unit test `test_temporal_pos_is_read` asserts that the same KV tokens with different `grid_thw` give DIFFERENT outputs (i.e. the temporal-pos embedding is actually being read, not silently bypassed).

- **CPU-safe path.** Self- and cross-attention both use `torch.nn.functional.scaled_dot_product_attention` (NOT FlexAttention), so the same code runs on CPU + CUDA without a compile pass. `num_latents=64` is small enough that the SDPA fast-path wins.

- **Configurable KV input dim.** `Qwen3VLPerceiverResamplerProjector.Config.in_features` toggles between consuming **pre-merger ViT features** (e.g. 1152 for Qwen3-VL-8B) and **post-merger features** (4096). The `_8b_perceiver_resampler()` factory currently uses post-merger (4096) -- same default as the Q-Former sibling -- see Open Issues for why this might not be ideal.

- **Param-count source-of-truth in the unit test.** `test_param_count_ballpark` measures and asserts the real count rather than relying on docstring estimates. At the production config (`in_features=4096`, `lm_dim=4096`, `num_latents=64`, `num_layers=6`, `n_heads=16`, `ffn_mult=2`, `t_max=32`) the measured count is **~1.21B**. At `in_features=1152` (pre-merger) it drops to ~1.06B. Both numbers are FFN-dominated even at `ffn_mult=2`; see Open Issues for the path to the original 80-110M target.

- **Two attention sub-layers per block.** Unlike the Q-Former sibling (which omits self-attn between queries), the Resampler's `_ResamplerBlock` carries BOTH a `_ResamplerSelfAttention` over latents AND a `_ResamplerCrossAttention` from latents to KV, in that order. This is the Flamingo paper's exact block layout and is the load-bearing structural delta vs Q-Former.

## Reviewer hints

- The Resampler in this PR is **standalone and additive**: deleting `projector_type="perceiver_resampler"` / `perceiver_resampler=None` from any config recovers the original Qwen3-VL behavior exactly. The risk surface for existing users is therefore the new optional field on `Qwen3VLModel.Config` (Python-side default = `"linear"`).
- The Resampler does NOT replace the in-encoder `PatchMerger` -- it consumes the merger's output. Open issue below (same as Q-Former PR 005's #1).
- The Resampler is NOT yet plumbed into `parallelize_qwen3_vl`. The Q-Former sibling PR added a sibling FSDP wrap; we'll mirror that pattern in a follow-up commit on this branch (kept minimal here per "don't gold-plate", since the parallelize wiring is also where the Q-Former agent's parallel branch touches the same file).
- The Resampler is NOT tensor-parallelized in this PR.

## Open issues (next-agent / follow-up PRs)

1. **Pre- vs post-merger Resampler input.** Currently the Resampler consumes the post-merger encoder output (`in_features=4096`). This means the vision signal is compressed twice -- once by `PatchMerger` (2x2 spatial merge -> 4x reduction) and once by the Resampler (~26x reduction). For nuScenes front-cam data the 4x spatial merge may already discard useful detail before Resampler pooling. Pre-merger Resampler (`in_features=1152`, encoder bypasses the merger) is the more aggressive but potentially higher-quality option. Same wiring requirement as the Q-Former sibling: skip `self.vision_encoder.merger` in the model's forward or expose a `return_premerge=True` flag on the encoder.

2. **Collator/tokenizer placeholder count.** `Qwen3VLModel._scatter_vision_embeds` asserts `num_placeholder_tokens == num_vision_tokens`. With the Resampler path every visual item produces exactly `num_latents=64` tokens at the LM input, so the prompt / chat-template must emit exactly 64 `<|image_pad|>` placeholders per image and 64 `<|video_pad|>` placeholders per video (regardless of T x H x W). This wiring lives outside the model in `MMDataLoader` / `NuScenesPlanningDatasetTitan` and is a **next-agent task**. Without it the GPU smoke will trip the placeholder-count assert immediately. **Shared with PR 005 (Q-Former).**

3. **MRoPE 3D positions for compressed tokens.** `_compute_mrope_freqs` builds 3D `(T, H, W)` positions per visual placeholder. After Resampler compression the 64 latent tokens no longer correspond to a spatial grid -- they are temporally-pooled features. We need to decide: (a) treat each latent as a "text-like" 1D position (simplest), or (b) keep a synthetic 8x8 grid (matches `num_latents=64`). Option (a) is the standard Flamingo convention. **Shared with PR 005.**

4. **Temporal-pos encoding init / scale choice.** We use `trunc_normal_(std=0.02)` on the temporal-pos table for sibling-projector parity with the latents init. Open question: should temporal-pos use a different (larger?) scale because it carries SEMANTIC information (frame index), not just a free parameter? The Flamingo paper does not specify; sinusoidal would also be defensible here. To swap: change `_RESAMPLER_PARAM_INIT["temporal_pos.weight"]` in `__init__.py` or replace `nn.Embedding(T_max, in_features)` with a fixed sinusoidal buffer.

5. **Multi-cam temporal-index assignment.** Currently every cam shares the same temporal-pos slots: cam-A frame-0, cam-B frame-0, cam-C frame-0 ALL get `temporal_pos[0]`. The alternative is to offset each cam by a fixed amount (e.g. cam-A in [0..3], cam-B in [4..7], cam-C in [8..11]), implicitly encoding camera identity through the temporal-pos slot. This is a small `_compute_temporal_indices` edit + a `t_max` bump from 32 to 32 * num_cams. We left the shared-T policy as the default because it is the simpler invariant and matches a single-cam future-frame extrapolation cleanly; the offset version is a natural A/B once Track A.3 lands.

6. **TP / CP for Resampler.** Not done in this PR. Same considerations as the Q-Former sibling -- the Resampler is small (~1.21B at 4096 dim) so TP within Resampler is unlikely to be a memory win unless the FFN scales. CP on the latent side is straightforward (latents are short); on the KV side it requires re-thinking. Punt to a separate PR.

7. **Param-count vs spec target.** The original spec targeted ~80-110M params for the Resampler. At `lm_dim=4096` and `num_layers=6` with the self-attn block + FFN (even at `ffn_mult=2`) the measured count is **~1.21B**. To hit the 80-110M target, either: (a) shrink `num_layers` to 1, (b) add an `internal_dim` field decoupling Resampler width from `lm_dim` (most principled; project lm_dim 4096 -> 1024 internally, then back up for output), or (c) drop self-attn entirely (but then it's just a Q-Former with a temporal-pos add, defeating the architectural-distinction premise of this PR). Recommended path for v2: (b). Out of scope here. **Shared concern with PR 005.**

## Pre-submission gate (must pass before opening upstream PR)

- [ ] **GPU smoke**: torchrun the `qwen3_vl_8b_planning_fsdp_3cam_resampler` factory at `--training.steps 4 --checkpoint.no-enable`; confirm forward pass completes, FSDP shards the Resampler, no shape-mismatch asserts. (Requires Open Issue #2 resolved first.)
- [ ] **Track A.3 full SFT validation**: 10K-step run; compare loss curves vs **both** the linear-projector baseline AND the Q-Former sibling at matched 64-token visual budget. The Resampler's temporal-pos inductive bias is the entire point of this PR; if it doesn't outperform Q-Former on multi-frame planning the architectural delta is not earning its keep.
- [ ] **Parallelize wiring**: add the sibling FSDP wrap inside `parallelize_qwen3_vl` (mirror the `qformer_projector` block in PR 005's parallelize.py edit). This is the only piece intentionally deferred from this PR to keep it small + focused; it must land before the upstream PR opens.

## Static test (already passed, CPU)

```text
cd /workspace/DriveLM_VLM_Project && \
  PYTHONPATH=torchtitan_qwen25:. /usr/bin/python3 \
    -m unittest torchtitan_qwen25.tests.unit_tests.test_perceiver_resampler_projector -v
# Ran 8 tests in 24.289s
# OK
```

And factory build:

```text
PYTHONPATH=torchtitan_qwen25:. /usr/bin/python3 -m scripts_titan.train_titan_qwen3_vl
# === qwen3_vl_8b_planning_fsdp_3cam_resampler ===
#   Model spec: qwen3_vl 8B-perceiver-resampler
#   Projector type: perceiver_resampler
#   LR: 2e-05
#   Warmup steps: 174
#   Total steps: 10000
#   Local BS: 1
#   Seq len: 8192
#   FSDP shard degree: -1
#   TP degree: 1
```
