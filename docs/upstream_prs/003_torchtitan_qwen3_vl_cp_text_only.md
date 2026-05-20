# PR 003 — Qwen3-VL Context Parallel wiring (text-only)

**Repo**: pytorch/torchtitan
**Branch**: `qwen25_vl_video_vla` (on QIU023/torchtitan fork)
**Status**: drafted locally, not yet pushed as PR

## Why

torchtitan's `qwen3_vl::parallelize.py` had `NotImplementedError` at the CP
wiring site. We added text-only CP support (vision inputs raise explicitly)
matching Megatron-LM's posture on multimodal CP.

## What changed

| File | Change | LOC |
|---|---|---|
| `torchtitan/models/qwen3_vl/parallelize.py` | replace `NotImplementedError` (line ~259) with `apply_cp_to_attention_module(...)` call; preserve hard-raise for vision-attached path | +38 / -4 |
| `tests/test_cp_qwen3_vl.py` | 5 tests: text-only CP forward / loss-parity / 2-rank / 4-rank / explicit vision-CP raise | +247 |

## Design highlights

- **Text-only CP works fully**: instruction-tuning / pure-text data flows
  through CP without modification. Attention sharding along seq dim is via
  PyTorch upstream `torch.distributed.tensor.experimental._attention.apply_cp`.
- **Vision-attached CP raises explicitly**: `if vision_embeds is not None:
  raise NotImplementedError(...)`. Mirrors Megatron's assert. Reasoning is
  documented in this PR's body + research note 004.
- **PP + CP combined**: also raises (overlap with PR 002 caveat).

## Caveats

- **No silent degradation**: explicit raise > silent wrong output. Maintainers
  may prefer warning + skip-CP-for-this-step, but for ML correctness we chose
  raise.
- **No partial vision-CP fallback**: even with very small vision token blocks
  (~64 tokens via Q-Former), CP would shard text path while AllGather'ing the
  small vision block. We do NOT implement this — gap 2 (MRoPE per-shard index
  advancement) is still wrong. Open problem; see doc 004.

## Test

```bash
pytest tests/test_cp_qwen3_vl.py -v
# 5 passed
```

## Reviewer hints

- Tag: torchtitan maintainers
- Cross-reference: PR 002 (PP wiring) for the parallel sibling
- Highlight: aligns torchtitan's stance with Megatron's on VLM-CP (explicit
  no-multimodal-CP today)

## Local refs

- File: `/workspace/DriveLM_VLM_Project/torchtitan_qwen25/torchtitan/models/qwen3_vl/parallelize.py`
- Commit: 0c68660

## Pre-submission gate

Will run a small text-only SFT (e.g. ShareGPT instruction data) on Qwen3-VL-8B
with CP=2 in `noop_vision_input` mode to validate before opening PR. **This is
NOT part of our Track A — it's a separate showcase / validation step done in
spare GPU windows.**
