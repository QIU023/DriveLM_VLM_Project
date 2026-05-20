# PR 006 — Qwen3-VL PixelShuffle + Linear projector

**Repo**: pytorch/torchtitan
**Branch**: `pixelshuffle_projector` (on QIU023/torchtitan fork, off
`qwen25_vl_video_vla`)
**Status**: drafted locally on `pixelshuffle_projector` branch, awaiting
Track A.2 SFT validation. Sibling to PR 005 (Q-Former; in-flight on
`qformer_projector` branch — see open issues there for the same
"in-encoder PatchMerger double-compresses" trade we make here).

## Why

For the fusion-mechanism school comparison (A.0 Linear vs A.1 Q-Former vs
A.2 PixelShuffle vs A.3 Perceiver) we need a *deterministic* vision-to-LM
compressor with **zero learnable query parameters** — the antithesis of
Q-Former's 64-query cross-attention pool. LLaVA-NeXT and InternVL both
use PixelShuffle 2x (space-to-depth) + a single Linear at the projector
slot for exactly this reason: deterministic compression, no extra
learnable pool, cheap forward (pure reshape + matmul).

In the 3-cam x 4-frame nuScenes planning token-budget plot, PixelShuffle
2x is intended to sit **between A.0 linear (~1680 tokens) and A.1
Q-Former-64 (64 tokens)** at ~420 LM-input tokens (4x compression vs A.0).
The current wiring uses post-merger features and so compounds to 16x
total compression (~105 tokens) — see Open Issues below for the
follow-up plumbing to reach the canonical 4x target.

## What changed

| File | Change | LOC (approx) |
|---|---|---|
| `torchtitan/models/qwen3_vl/pixelshuffle_projector.py` | new module: `Qwen3VLPixelShufflePlusLinearProjector` (Config + forward, manual space-to-depth + Linear) | +280 |
| `torchtitan/models/qwen3_vl/__init__.py` | new `_vl_pixelshuffle_config()` helper + `_8b_pixelshuffle()` factory + `"8B-pixelshuffle"` flavor registration + `Qwen3VLPixelShufflePlusLinearProjector` import | +60 |
| `torchtitan/models/qwen3_vl/model.py` | `Qwen3VLModel.Config`: `projector_type` + `pixelshuffle` fields; `__init__`: build the projector when `projector_type=='pixelshuffle'`; `_get_vision_embeds`: post-encoder PixelShuffle compression + post-merger grid_thw passthrough + DeepStack skip (open issue) | +65 / -2 |
| `tests/unit_tests/test_pixelshuffle_projector.py` | 9 CPU tests: shape contract at vit_dim=1152 (20x20 grid, 3-cam-4f 1680->420), post-merger shape contract, pre-merger param-count (~19M), post-merger param-count (~67M), grad flow, even-grid invariant, padding preservation, mixed-grid raises | +200 |
| `scripts_titan/configs/qwen3_vl_8b_planning_fsdp_3cam_pixelshuffle.toml` | (DriveLM_VLM_Project) hyperparam manifest for the A.2 Track A SFT run | +175 |
| `scripts_titan/train_titan_qwen3_vl.py` | (DriveLM_VLM_Project) plumb `model_flavor` through `_build_trainer_config`; new factory `qwen3_vl_8b_planning_fsdp_3cam_pixelshuffle()` | +75 / -5 |

## Design highlights

- **No FlexAttention machinery**: PixelShuffle has no attention. The
  forward path is a single `view + permute + view` (manual
  `PixelUnshuffle`) followed by one `Linear`. CPU-safe; no torch.compile
  detour for this projector.
- **Channel-last layout** kept end-to-end. `torch.nn.PixelUnshuffle`
  expects `(B, C, H, W)`; we keep `(B, H, W, C)` to match the rest of the
  vision pipeline (zero extra permutes).
- **Padded-batch friendly**: forward accepts the encoder's padded
  `(num_items, max_num_patch, dim)` tensor as-is, rearranges only the
  valid `t*h*w` prefix, and zero-pads the compressed-grid tail so the
  downstream `valid_mask = arange(max_tokens) < num_tokens_per_item //
  shuffle_unit` lines up cleanly.
- **Temporal as batch**: PixelShuffle is spatial-only. For video items
  (t > 1) the temporal axis is folded into the batch so each frame is
  shuffled independently.

## Open issues (to address before merging upstream)

### 1. In-encoder PatchMerger double-compresses

The wiring as committed feeds **post-merger** features into the
PixelShuffle projector. This compounds the in-encoder 2x2 merger with
the 2x2 PixelShuffle for a total **16x** compression vs raw ViT
patches (~105 LM-input tokens at the 3-cam x 4-frame nuScenes setting).
The canonical LLaVA-NeXT compression is **4x** — i.e. bypass the
in-encoder merger and feed *raw ViT features* (`in_features=1152`)
through the PixelShuffle projector. The projector class already
supports this (it's a pure `in_features` config knob); the plumbing
change is to make the in-encoder `PatchMerger` no-op when
`projector_type=='pixelshuffle'`, route raw ViT features out of the
encoder, and adjust DeepStack accordingly. This is the same "pre- vs
post-merger" open question Q-Former (PR 005) faces; resolving it
consistently across both projector heads is a follow-up agent task.

### 2. DeepStack interaction

DeepStack injects intermediate ViT features into early LLM hidden
states at the **post-merger grid**. The PixelShuffle projector then
compresses by another 4x; the DeepStack mask no longer lines up with
the compressed vision token positions. As committed, the model.py
wiring **drops DeepStack** when `projector_type=='pixelshuffle'` (i.e.
`deepstack_embeds = []`). This is a soft regression for the
pixelshuffle variant. The two fix options are:

  (a) **Compress DeepStack identically.** Apply the same PixelShuffle
      rearrange (no Linear; just the space-to-depth) to the DeepStack
      features, then a per-DeepStack-index Linear to lm_dim. This
      preserves the architectural intent (high-frequency residual at
      compressed grid).
  (b) **Drop DeepStack entirely** for the pixelshuffle variant and
      document the quality trade.

Track A.2 SFT will run with option (b) (the trivial wiring) so we can
measure the DeepStack-free pixelshuffle baseline first. Option (a) is
queued as a follow-up if (b) leaves significant quality on the table.

### 3. Dataset placeholder count

`Qwen3VLModel._scatter_vision_embeds` strictly requires the number of
`<|image_pad|>` placeholders in `tokens` to match the number of vision
tokens. After PixelShuffle the per-item visual-token count is
`(t*h*w / merger_unit) // shuffle_unit` — typically ~105 / item for
3-cam x 4-frame nuScenes planning. The collator currently emits the
post-merger count (~420 / item). The next agent must update the
collator to emit the compressed count when `projector_type` is
`pixelshuffle`. This is the same "scatter alignment" task tracked in
PR 005 for Q-Former (which emits a fixed 64 / item).

### 4. Even-grid invariant

PixelShuffle 2x requires both `h` and `w` to be even at the grid where
it operates. The Qwen3-VL collator pads images to multiples of
`patch_size * spatial_merge_size = 32`, so the **post-merger** grid is
always at least 1×1 with both sides positive integers; for the most
common nuScenes input resolutions the post-merger grid sides are even.
We assert this at forward time and raise `ValueError` with an actionable
message. Mixed-resolution batches (different `(t, h, w)` per item) are
rejected with `ValueError` and the caller is told to group by shape and
call per group.

## Test

```bash
cd torchtitan_qwen25
PYTHONPATH=. /usr/bin/python3 -m unittest tests.unit_tests.test_pixelshuffle_projector -v
# 9 passed
```

Static factory build:

```bash
cd DriveLM_VLM_Project
PYTHONPATH=torchtitan_qwen25:. /usr/bin/python3 -m scripts_titan.train_titan_qwen3_vl
# All 5 factories build cleanly:
#   qwen3_vl_8b_planning_fsdp                      (linear)
#   qwen3_vl_8b_planning_fsdp_1cam_8f              (linear)
#   qwen3_vl_8b_planning_fsdp_3cam                 (linear)
#   qwen3_vl_8b_planning_fsdp_3cam_pixelshuffle    (pixelshuffle)
#   qwen3_vl_8b_planning_fsdp_tp                   (linear)
```

## Reviewer hints

- Tag: torchtitan maintainers (HDCharles, awgu, tianyu-l) — same set as
  PR 002/003/005.
- Highlight: this is the SECOND switchable post-encoder projector for
  qwen3_vl (sibling to PR 005 Q-Former); both share the
  `projector_type` Config field. When the two PRs merge, the enum will
  read `{"linear", "qformer", "pixelshuffle"}`. The branches are
  isolated to avoid stepping on each other.
- Highlight: ~67M params at the post-merger wiring, ~19M at the
  pre-merger wiring (when the follow-up plumbing lands). Significantly
  smaller than Q-Former (~1.2B) — comparable to the existing
  PatchMerger.
- Note: the DeepStack interaction (open issue #2) is the largest
  outstanding architectural question; happy to fold whichever direction
  upstream prefers.

## Local refs

- Module: `/workspace/DriveLM_VLM_Project_pixelshuffle/torchtitan_qwen25/torchtitan/models/qwen3_vl/pixelshuffle_projector.py`
- Test: `/workspace/DriveLM_VLM_Project_pixelshuffle/torchtitan_qwen25/tests/unit_tests/test_pixelshuffle_projector.py`
- Wiring: `/workspace/DriveLM_VLM_Project_pixelshuffle/torchtitan_qwen25/torchtitan/models/qwen3_vl/{model.py,__init__.py}`
- TOML + factory: `/workspace/DriveLM_VLM_Project_pixelshuffle/scripts_titan/{configs/qwen3_vl_8b_planning_fsdp_3cam_pixelshuffle.toml,train_titan_qwen3_vl.py}`
- Branch: `pixelshuffle_projector` on both QIU023/torchtitan (submodule
  files) and QIU023/DriveLM_VLM_Project (factory + TOML + this doc).

## Pre-submission gate

Will run Track A.2 8B SFT (3-cam x 4-frame, 10k steps) with the
pixelshuffle projector first to validate the deterministic-compression
quality vs A.0 linear baseline. Open issues #2 (DeepStack) and #3
(collator placeholder count) must be resolved before the SFT run.
