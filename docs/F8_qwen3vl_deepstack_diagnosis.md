# F8: planning_eval_compress.py — Qwen3-VL deepstack incompatibility

**Status**: DIAGNOSED, NOT FIXED. Patch sketch only — do not commit.
**Author**: prep agent (CPU-only, training in progress).
**Date**: 2026-05-25.
**Scope**: applies when `--ckpt` is any Qwen3-VL checkpoint
(B.5'' v2 1-cam `nusc_planning_b5pp_1cam_qwen3vl_multimodal/final` or
B.5''' 3-cam `nusc_planning_b5ppp_3cam_qwen3vl_multimodal/final`).
Qwen2.5-VL ckpts (B.5, B.5', R1*) are unaffected.

## TL;DR

`install_compression_hook()` monkey-patches `inner.get_video_features` and
returns `_FakeVisOut(pooler_output=<list of compressed tensors>)`.
That mocks only ONE of the two return fields Qwen3-VL's LM forward
consumes. The deepstack feature list is silently dropped → either
runtime crash (`AttributeError: 'NoneType' object has no attribute
'masked_scatter'` or similar) or silent visual-grounding loss (zero
deepstack residuals injected into early decoder layers).

## Trigger lines (cross-referenced)

### 1. The monkey-patch is too narrow

`scripts/planning_eval_compress.py` — `install_compression_hook` line 332:

```python
class _FakeVisOut:
    """Mimics the vision-feature return object the LM forward expects."""
    def __init__(self, t):
        self.pooler_output = t       # <-- ONLY pooler_output is set
```

…and at line 408 it returns `return _FakeVisOut(compressed_items)`.
There is no `.deepstack_features` attribute.

### 2. What Qwen3-VL ACTUALLY returns and consumes

`transformers==5.5.3` `modeling_qwen3_vl.py`:

- Vision tower returns `BaseModelOutputWithDeepstackFeatures` (declared
  L51-L55) which holds both `pooler_output` AND `deepstack_features:
  list[torch.FloatTensor] | None` (the per-layer feature taps).
- In the LM forward path the model unpacks BOTH:
  - L1290: `image_embeds = image_outputs.pooler_output`
  - L1291: `deepstack_image_embeds = image_outputs.deepstack_features`
  - L1302: `video_embeds = video_outputs.pooler_output`
  - L1303: `deepstack_video_embeds = video_outputs.deepstack_features`
- The deepstack list is then injected through `_deepstack_process()`
  inside each decoder layer (L926-930): per-layer residual that
  augments visual tokens with multi-scale features.

### 3. What happens with the current hook

When `get_video_features` is patched to return `_FakeVisOut`:
- L1303 `deepstack_video_embeds = video_outputs.deepstack_features`
  → `AttributeError: '_FakeVisOut' object has no attribute
  'deepstack_features'`.

Reproduced (CPU dry-run, no model load):

```python
class _FakeVisOut:
    def __init__(self, t): self.pooler_output = t
o = _FakeVisOut([1,2,3])
o.deepstack_features   # AttributeError
```

So the crash is deterministic and happens BEFORE any compute reaches a
deepstack merger — a clean error path, not silent wrongness.

### 4. Secondary failure mode (if hook is hardened wrong)

If we naively set `_FakeVisOut(t, deepstack_features=[])`, the LM passes
an empty list through `_deepstack_process` and the layer-index range
check (`layer_idx in range(len(deepstack_visual_embeds))`, L926) is
falsy for ALL layers → silent loss of multi-scale visual grounding.
That is the "wrong but silent" variant — must avoid.

## Compress-method-specific risk

The spatial methods (FasterVLM, PyramidDrop, PruMerge) prune based on
attention scores from the pooler. The deepstack tap is taken from
MIDDLE encoder layers (config.deepstack_visual_indexes, typically [4,8,12]
for Qwen3-VL-4B vision config), BEFORE the final merger. So the "kept
token indices" derived from the pooler-side attention need to be
mirrored on the per-layer deepstack tensors too — you can't compress
the pooler and leave deepstack uncompressed (token-count mismatch in
`_deepstack_process` masked_scatter).

## Patch sketch (DO NOT COMMIT — flagged for review)

Two-pronged change to `install_compression_hook`:

```python
class _FakeVisOut:
    """Now mocks BOTH return fields of Qwen3-VL's get_video_features."""
    def __init__(self, pooler, deepstack=None):
        self.pooler_output = pooler
        # Qwen3-VL: list of (sum_visual_tokens, D) per deepstack tap
        self.deepstack_features = deepstack
```

…and inside `_patched()`:

```python
with torch.no_grad():
    real = orig(pv, grid)
    embeds = real.pooler_output
    deepstack = getattr(real, "deepstack_features", None)  # Qwen3-VL only

# ... existing compression on `embeds` ...

# Mirror the SAME keep-mask onto every deepstack layer:
new_deepstack = None
if deepstack is not None:
    new_deepstack = []
    for layer_feat in deepstack:                    # layer_feat: (sum_N_post, D)
        # split layer_feat by per_item_orig (same row layout as embeds)
        chunks = torch.split(layer_feat, per_item_orig, dim=0)
        compressed_chunks = []
        for i, chunk in enumerate(chunks):
            # CRITICAL: must use the SAME compression schedule applied to
            # `embeds[i]` so token counts agree. The training-free methods
            # are deterministic given (method, ratio, seed) so re-running
            # _compress_per_item with the same args is correct.
            comp = _compress_per_item(
                chunk, n_per_item=per_item_orig[i],
                spatial_method=spatial_method, spatial_ratio=spatial_ratio,
                temporal_compressor=temporal, temporal_ratio=temporal_ratio, T=T,
            )
            # Pad/truncate to per_item_comp[i] (same as the pooler-side branch)
            target = per_item_comp[i]
            if comp.shape[0] != target:
                if comp.shape[0] > target: comp = comp[:target]
                else:
                    pad = comp.new_zeros((target - comp.shape[0], comp.shape[-1]))
                    comp = torch.cat([comp, pad], dim=0)
            compressed_chunks.append(comp)
        new_deepstack.append(torch.cat(compressed_chunks, dim=0))

return _FakeVisOut(compressed_items, new_deepstack)
```

### Why "same compression schedule" is non-trivial

FasterVLM and PyramidDrop pick keep-indices based on the *pooler*
attention map — they're not arbitrary index lists you can reuse. To
mirror correctly you either:

  (a) Refactor the compressors to return `(compressed_tokens,
       keep_indices)` and apply `keep_indices` to the deepstack
       tensors directly. This is the **right** fix.
  (b) Cache the random/seeded state and re-run the compressor on each
       deepstack layer — works for deterministic methods only.

Option (a) needs a `compress_visual_tokens` signature change. Option
(b) is hacky but no API churn. The patch sketch above implicitly takes
(b) but is INCORRECT for FasterVLM (which re-derives the attention map
from a per-call pooler) — those methods need (a).

**Recommendation**: do (a) — return `(compressed, keep_indices)` from
`compress_visual_tokens` and apply once to pooler + each deepstack tap.

## CPU-side dry-run (confirms diagnosis without GPU)

Run from the repo root with the training process untouched (no torch
imports go near CUDA — module-level only):

```python
# scripts/_smoke_deepstack_compat.py — does NOT touch CUDA
import sys, types
sys.path.insert(0, "scripts")
from planning_eval_compress import _FakeVisOut

# Simulate Qwen3-VL's return contract
class RealOut:
    pooler_output = [1, 2, 3]
    deepstack_features = [[10, 20], [30, 40]]   # list of per-layer features

# Simulate the LM's unpacking pattern (modeling_qwen3_vl.py L1302-1303)
def lm_consume(video_outputs):
    video_embeds = video_outputs.pooler_output
    deepstack_video_embeds = video_outputs.deepstack_features  # <- breaks
    return video_embeds, deepstack_video_embeds

# Reference path (real) — passes
print("REAL:", lm_consume(RealOut))

# Hook path (current) — fails
fake = _FakeVisOut([1, 2, 3])
try:
    print("HOOK:", lm_consume(fake))
except AttributeError as e:
    print("HOOK CRASH (as predicted):", e)
```

Expected output:
```
REAL: ([1, 2, 3], [[10, 20], [30, 40]])
HOOK CRASH (as predicted): '_FakeVisOut' object has no attribute 'deepstack_features'
```

## What to do next (operator action)

1. Review this diagnosis.
2. Decide between patch-option (a) [API change, clean] or (b) [closure
   re-run, hacky but local]. Recommendation: (a).
3. Once decided, schedule a 10-sample smoke on the actual 1-cam Qwen3-VL
   ckpt with `--spatial-method fastervlm --spatial-ratio 4` BEFORE
   trusting any compression-vs-precision matrix that uses Qwen3-VL.

## Files referenced (absolute)

- `/workspace/DriveLM_VLM_Project/scripts/planning_eval_compress.py` (lines 100-110, 332-415)
- `/venv/trt_llm/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py` (lines 51-55, 802-821, 1290-1332, 926-947)
- `/workspace/DriveLM_VLM_Project/scripts/visual_compress.py` (compressor library)
