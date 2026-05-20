# PR 005 — torchtitan Qwen3-VL Q-Former projector (token-budget compressor)

**Repo**: pytorch/torchtitan (potential upstream PR)
**Branch**: `qwen25_vl_video_vla` (Track A branch on QIU023/DriveLM_VLM_Project fork; the torchtitan submodule sits on `attention_residual_dev`)
**Status**: Drafted locally + unit-tested on CPU. **PR not yet opened.** Pre-submission gate:
  1. GPU smoke (next-agent task — boot the 8B-qformer factory, verify forward pass + FSDP shard).
  2. Track A.1 full SFT validation (~10K steps on nuScenes planning; A/B vs the linear-projector baseline at matched compute).

## Why we need it

Stock Qwen3-VL's vision-to-LM projector is a fixed two-layer `PatchMerger` MLP applied INSIDE the vision encoder. With a `spatial_merge_size=2` it shrinks the patch grid by 4x. For our 3-cam x 4-frame nuScenes planning recipe this leaves **~1680 visual tokens per sample** at the LM input boundary — fine for HBM training (FSDP-8 has the bandwidth) but expensive for TensorRT deploy where KV-cache scales linearly with context length.

We want a **swap-in alternative projector** that compresses the visual context to a **fixed budget** (here: 64 tokens, ~26x compression). The selected mechanism is a BLIP-2-style Q-Former: 64 learnable query tokens crossed with the vision encoder's output through a stack of cross-attention + FFN blocks. The output is always `(B, 64, lm_dim)`, regardless of how many patches the input video had.

This is **explicitly a token-budget vs L2 tradeoff for deployment**, not a fusion-mechanism research comparison. From-scratch Q-Former on 24K SFT samples will lose to the pretrained-aligned linear baseline on quality metrics — that is the expected, accepted cost; the win is at TRT inference time.

## What changed

| File | Change | LOC (approx) |
|---|---|---|
| `torchtitan_qwen25/torchtitan/models/qwen3_vl/qformer_projector.py` | **new** — `Qwen3VLQFormerProjector` module + supporting `_QFormerBlock` / `_QFormerCrossAttention` / `_QFormerFFN` building blocks; full Configurable wiring; CPU-safe SDPA attention path with optional KV padding mask. Internal width decoupled from `lm_dim` via `internal_dim` (default 1024); final `out_proj: internal_dim -> lm_dim` lifts the output. | +355 |
| `torchtitan_qwen25/torchtitan/models/qwen3_vl/__init__.py` | add `Qwen3VLQFormerProjector` import + export; new `_vl_qformer_config(...)` helper with `internal_dim=1024`, `n_heads=8` defaults; new `_8b_qformer()` config factory; registered as `"8B-qformer"` flavor; `dataclasses` import | +65 / -1 |
| `torchtitan_qwen25/torchtitan/models/qwen3_vl/model.py` | add `projector_type: str = "linear"` and `qformer: Qwen3VLQFormerProjector.Config \| None = None` to `Qwen3VLModel.Config`; switch in `__init__` to instantiate `self.qformer_projector` when `projector_type == "qformer"`; wire Q-Former into `_get_vision_embeds` so it consumes the encoder's `merged_embeds` + valid mask and emits `(num_items * num_queries, lm_dim)`; raises on bad value | +55 |
| `torchtitan_qwen25/torchtitan/models/qwen3_vl/parallelize.py` | extend `parallelize_qwen3_vl` to FSDP-wrap `model.qformer_projector` as its own unit on the same `dp_mesh` as the vision encoder | +25 |
| `torchtitan_qwen25/tests/unit_tests/test_qformer_projector.py` | **new** — 8-case CPU unit test: production shape, post-merger shape, padding-mask shape + finiteness, gradient flow on tiny config (with internal_dim != lm_dim), num_queries invariance vs input length, param-count guard (80M-120M for post-merger; ~80M for pre-merger), internal_dim-vs-lm_dim decoupling check | +220 |
| `scripts_titan/train_titan_qwen3_vl.py` | add `model_flavor: str = "8B"` parameter to `_build_trainer_config`; new `qwen3_vl_8b_planning_fsdp_3cam_qformer()` factory using `model_flavor="8B-qformer"`; smoke loop covers the new factory | +50 / -1 |
| `scripts_titan/configs/qwen3_vl_8b_planning_fsdp_3cam.toml` | annotate baseline with `projector_type = "linear"` for diff symmetry vs the Q-Former TOML | +5 |
| `scripts_titan/configs/qwen3_vl_8b_planning_fsdp_3cam_qformer.toml` | **new** — full hyperparam manifest for the Q-Former variant with rationale, `[model.qformer]` section, and dataset key rename | +130 |
| `docs/upstream_prs/005_torchtitan_qwen3_vl_qformer.md` | **this file** | +130 |

Total diff: **~940 LOC, 3 new files, 6 modified files**.

## Design highlights

- **Switchable, default-off.** `projector_type` defaults to `"linear"`, so every existing factory / TOML / checkpoint keeps working unchanged. The Q-Former is opt-in via either the `"8B-qformer"` flavor or by setting `projector_type="qformer"` + populating `qformer=...` on a custom config.

- **Direct LM-dim output.** Q-Former emits `(B, num_queries, lm_dim)` — no extra linear projector needed downstream. The pre-LN cross-attention design and final output `LayerNorm` keep activations bounded; tested on bf16-friendly random inputs.

- **CPU-safe path.** Cross-attention uses `torch.nn.functional.scaled_dot_product_attention` (NOT FlexAttention), so the same code runs on CPU + CUDA without a compile pass. `num_queries=64` is small enough that the SDPA fast-path wins.

- **Configurable KV input dim.** `Qwen3VLQFormerProjector.Config.in_features` toggles between consuming **pre-merger ViT features** (e.g. 1152 for Qwen3-VL-8B) and **post-merger features** (4096). The `_8b_qformer()` factory currently uses post-merger (4096) — see Open Issues for why this might not be ideal.

- **FSDP-wrapped as its own unit.** The projector is a sibling FSDP unit to the vision encoder (same `dp_mesh`). It is NOT bundled into the vision encoder's single `fully_shard` call because (a) the vision encoder still emits DeepStack features that go to the LM independently, and (b) keeping it as a separate unit simplifies future PP cuts.

- **Param-count source-of-truth in the unit test.** `test_param_count_guard` measures and asserts the real count rather than relying on docstring estimates. At the production config (`internal_dim=1024`, `lm_dim=4096`, `in_features=4096`, `num_queries=64`, `num_layers=6`, `n_heads=8`, `ffn_mult=4`) the measured count is **~117.6M** — matching BLIP-2 scale. At `in_features=1152` (pre-merger), it drops to ~81.4M. The `internal_dim` decoupling from `lm_dim` (introduced as the fix for the original 1.21B blow-up) is what bounds the count to this BLIP-2-shaped envelope; without it, FFN at `lm_dim=4096` dominates (2 * 4096 * 16384 ≈ 134M per layer × 6 = 805M).

## Reviewer hints

- The Q-Former in this PR is **standalone and additive**: deleting `projector_type="qformer"` / `qformer=None` from any config recovers the original Qwen3-VL behavior exactly. The risk surface for existing users is therefore the new optional field on `Qwen3VLModel.Config` (Python-side default = `"linear"`).
- The Q-Former does NOT replace the in-encoder `PatchMerger` — it consumes the merger's output. Open issue below.
- The Q-Former is NOT tensor-parallelized in this PR. `parallelize_qwen3_vl` only adds the FSDP wrap; TP / CP / PP integration is a follow-up (kept minimal here per "don't gold-plate").

## Open issues (next-agent / follow-up PRs)

1. **Pre- vs post-merger Q-Former input.** Currently the Q-Former consumes the post-merger encoder output (`in_features=4096`). This means the vision signal is compressed twice — once by `PatchMerger` (2x2 spatial merge -> 4x reduction) and once by the Q-Former (~26x reduction). For nuScenes front-cam data the 4x spatial merge may already discard useful detail before Q-Former pooling. Pre-merger Q-Former (`in_features=1152`, encoder bypasses the merger) is the more aggressive but potentially higher-quality option. To switch: feed pre-merger features into `qformer_projector` and update `_8b_qformer()` to set `in_features=1152`. Requires either skipping `self.vision_encoder.merger` in the model's forward or exposing a `return_premerge=True` flag on the encoder.

2. **Collator/tokenizer placeholder count.** `Qwen3VLModel._scatter_vision_embeds` asserts `num_placeholder_tokens == num_vision_tokens`. With the Q-Former path every visual item produces exactly `num_queries=64` tokens at the LM input (wired in `Qwen3VLModel._get_vision_embeds` — when `qformer_projector` is set the merged ViT features are routed through it and the output is flattened to `num_items * 64`), so the prompt / chat-template must emit exactly 64 `<|image_pad|>` placeholders per image and 64 `<|video_pad|>` placeholders per video (regardless of T x H x W). This wiring lives outside the model in `MMDataLoader` / `NuScenesPlanningDatasetTitan` and is a **next-agent task**. Without it the GPU smoke will trip the placeholder-count assert immediately. *Model-side status*: Q-Former is now invoked in `_get_vision_embeds` and emits the contracted `(num_items * num_queries, dim)` shape; the remaining work is on the dataset side. DeepStack features are intentionally NOT routed through the Q-Former (they encode spatial detail that the 64-query pooling would destroy) — this means the LM still expects the *original* (post-merger) DeepStack token count at vision positions for the DeepStack add-paths; reconciling that with the 64-placeholder change is part of the dataset task.

3. **MRoPE 3D positions for compressed tokens.** `_compute_mrope_freqs` currently builds 3D `(T, H, W)` positions per visual placeholder. After Q-Former compression the 64 query tokens no longer correspond to a spatial grid — they are global pooled features. We need to decide: (a) treat each query as a "text-like" 1D position (simplest), or (b) keep a synthetic 8x8 grid (matches `num_queries=64`). Option (a) is the BLIP-2 convention.

4. **TP / CP for Q-Former.** Not done in this PR. Q-Former is small (~1.2B at 4096 dim) so TP within Q-Former is unlikely to be a memory win unless the FFN scales. CP on the query side is straightforward (queries are short) but on the KV side requires re-thinking — punt to a separate PR.

5. **Param-count vs spec target. [RESOLVED — torchtitan commit `5ee6380`]** The original spec targeted ~80-120M params for the Q-Former. The first cut implicitly used `internal_dim = lm_dim = 4096`, which gave ~1.21B params (FFN-dominated: 2 * 4096 * 16384 ≈ 134M per layer × 6 = 805M). **Resolved by decoupling `internal_dim` from `lm_dim`** — Q-Former now operates at `internal_dim=1024` (BLIP-2-base uses 768) with `n_heads=8`, `ffn_mult=4`, `num_layers=6`, and a final `out_proj: 1024 -> 4096` lifts the output to LM width. Measured param counts: **117.6M at the production post-merger config** (`in_features=4096`) and **81.4M at the pre-merger variant** (`in_features=1152`) — both within the BLIP-2 envelope. The `Qwen3VLQFormerProjector.Config.internal_dim` field carries the new dimension; `_vl_qformer_config()` defaults it to 1024. Guard is in `tests/unit_tests/test_qformer_projector.py::test_param_count_guard` (80M < count < 120M).

## Pre-submission gate (must pass before opening upstream PR)

- [ ] **GPU smoke**: torchrun the `qwen3_vl_8b_planning_fsdp_3cam_qformer` factory at `--training.steps 4 --checkpoint.no-enable`; confirm forward pass completes, FSDP shards the Q-Former, no shape-mismatch asserts. (Requires Open Issue #2 resolved first.)
- [ ] **Track A.1 full SFT validation**: 10K-step run; compare loss curves vs linear-projector baseline. Q-Former is expected to underperform; the bar is "loss decreases monotonically + no NaN".
- [ ] **A/B at deploy**: TRT-export both variants, measure KV-cache size + TTFT. The Q-Former wins here is the entire point of the PR; if it doesn't show ~26x KV-cache reduction the whole exercise is invalid.

## Static test (already passed, CPU)

```text
cd /workspace/DriveLM_VLM_Project && \
  PYTHONPATH=torchtitan_qwen25:. /usr/bin/python3 -m pytest \
    torchtitan_qwen25/tests/unit_tests/test_qformer_projector.py -v
# 8 passed in 5.93s   (post param-count fix; was 6 tests / 22.961s pre-fix)
```

And factory build:

```text
PYTHONPATH=torchtitan_qwen25:. /usr/bin/python3 -m scripts_titan.train_titan_qwen3_vl
# === qwen3_vl_8b_planning_fsdp_3cam_qformer ===
#   Model spec: qwen3_vl 8B-qformer
#   Projector type: qformer
#   ...
```
