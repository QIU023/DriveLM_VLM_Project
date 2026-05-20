# PR 002 — Qwen3-VL Pipeline Parallel wiring

**Repo**: pytorch/torchtitan
**Branch**: `qwen25_vl_video_vla` (on QIU023/torchtitan fork)
**Status**: drafted locally, not yet pushed as PR

## Why

torchtitan's `qwen3_vl` model had no PP support — only FSDP/TP/CP were wired.
For 8B-scale VLM training across multiple nodes, PP is the natural way to cut
the model. We added PP wiring for the dense Qwen3-VL family (2B/8B), with
DeepStack-aware layer cut and vision-encoder pinned to stage 0.

## What changed

| File | Change | LOC |
|---|---|---|
| `torchtitan/distributed/pipeline_parallel.py` | new `generate_vlm_fqn_per_model_part(num_stages, num_layers, last_vision_consumer_layer, …)` next to existing LLM helper | +142 |
| `torchtitan/models/qwen3_vl/parallelize_pp.py` | new `pipeline_qwen3_vl()`; DeepStack-aware cut; vision pinned stage 0 | +260 |
| `torchtitan/models/qwen3_vl/model.py` | `forward()` tolerates `vision_encoder=None` / `tok_embeddings=None` on non-first stages | +68 / -24 |
| `torchtitan/models/qwen3_vl/__init__.py` | `pipelining_fn=pipeline_qwen3_vl` | +1 |
| `tests/unit_tests/test_pipeline_qwen3_vl.py` | 14 tests, all pass | +317 |

## Design highlights

- **DeepStack-aware**: Qwen3-VL injects intermediate ViT features into early
  decoder layers (`vision_intermediate_indices`). The cut function ensures all
  consumer layers of a given DeepStack injection stay on the same stage as
  the corresponding vision-encoder output — no cross-stage tensor-passing for
  these auxiliary connections.
- **Vision pinned to stage 0**: ViT runs end-to-end on rank-0; only LM
  decoder is split. Avoids fragmenting a forgivingly small (~1B param) ViT.
- **PP+CP combined**: hard-raises `NotImplementedError`. Mirrors Megatron-LM's
  assert. See PR 003 / open-problem doc 004 for rationale.

## Caveats

- **Weight tying** (`tok_embeddings.weight` shared with `output.weight`): with
  PP=2 these end up on different stages. We disable weight tying when PP>1
  (force untied LM head); document this in `init_states`. Upstream maintainers
  may prefer an alternative (cross-stage AllReduce in optimizer step).
- **VLM cut helper API**: `generate_vlm_fqn_per_model_part` signature
  intentionally diverges from `generate_llm_fqn_per_model_part`. Open to
  refactor as a single helper with a `vision_consumer` kwarg.

## Test

```bash
pytest tests/unit_tests/test_pipeline_qwen3_vl.py -v
# 14 passed
```

## Reviewer hints

- Tag: torchtitan maintainers (HDCharles, awgu, tianyu-l)
- Highlight: this is the *first* VLM PP impl in torchtitan; LLM PP path
  untouched
- Note: PP+CP combined is a separate problem; see doc 004

## Local refs

- Files under `/workspace/DriveLM_VLM_Project/torchtitan_qwen25/torchtitan/models/qwen3_vl/`
- Commits: f86fe8d (parallelize_pp.py), 2a18b82 (model.py tolerant forward),
  88f6107 (__init__.py registration)

## Pre-submission gate

Will run our own Track A 8B SFT with PP=2 first to validate end-to-end before
opening upstream PR.
