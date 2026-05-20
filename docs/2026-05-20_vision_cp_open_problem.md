# Vision Context-Parallel for VLM Training: An Open Problem

*2026-05-20 — torchtitan / Qwen3-VL CP wiring notes*

## 1. Executive summary

Context Parallel (CP) shards the sequence dimension across ranks so that
attention can scale to sequences far longer than any single device can hold.
For text-only LMs the recipe is well established (Megatron-LM, DeepSpeed-Ulysses,
PyTorch's `torch.distributed.tensor.experimental.context_parallel`). For
**vision-attached VLMs** it is still an open industry problem. Megatron-LM hard
asserts CP off when images are present, NeMo's tracker carries open VLM+CP
instability bugs, DeepSpeed makes no VLM-CP guarantee, and HuggingFace
Accelerate has no CP at all. In porting CP to torchtitan's `qwen3_vl` we
confirmed two specific, well-defined gaps in the vision-attached forward path.
We document them here as a reference for future work rather than attempting a
solve.

## 2. Gap 1 — `masked_scatter` shard / index frame mismatch

**Location.**
`torchtitan_qwen25/torchtitan/models/qwen3_vl/model.py`, `_scatter_vision_embeds`
around line 416 (`inputs_embeds = inputs_embeds.masked_scatter(...)`).

**Problem.** The vision encoder produces a dense tensor of shape
`(num_vision_tokens, hidden)` that lives on the *vision-mesh* ranks. The text
side, by the time we hit the LM forward under CP, has already been sharded
along the sequence dimension. The scatter call

```python
inputs_embeds.masked_scatter_(image_mask, vision_embeds)
```

assumes `image_mask` indexes into a **full** sequence and that `vision_embeds`
was laid out in that same full frame. Under CP, `inputs_embeds` is the local
shard, `image_mask` was built against the unsharded token stream, and the two
no longer agree. Index frame and destination frame have decoupled.

**Why it is hard.** Vision tokens are not uniformly distributed along the text
sequence. A single image inserted near the start of the prompt lands entirely
on one CP rank; a long interleaved video could pile every vision token onto
one shard while leaving others empty. Any naive fix that AllGathers the full
`inputs_embeds` to do the scatter on every rank materialises the unsharded
sequence and destroys the memory benefit CP was bought to deliver.

**Industry workaround.** Compress vision tokens before they enter the LM
(Q-Former, spatial-merge, cross-frame pruning) so the merged LM-sequence is
short enough that CP is not needed in the first place, or so that the
AllGather is cheap. This is what every shipping open VLA stack does today.

## 3. Gap 2 — `_compute_mrope_freqs` per-shard index advancement

**Location.**
`torchtitan_qwen25/torchtitan/models/qwen3_vl/model.py::_compute_mrope_freqs`
(line 122).

**Problem.** Qwen3-VL's Multimodal RoPE (MRoPE) interleaves four position-id
streams (text, T, H, W) with **stateful advancement**: when the iterator
encounters an `<image>` marker it advances the 3D positions by
`(T_image, H_image, W_image)` before resuming the text counter. Each token's
absolute position therefore depends on every image marker that appeared before
it. Under CP, each rank holds only a contiguous slice of the token stream and
has no view of prior shards' image markers, so it cannot independently
reconstruct its own absolute positions.

**Why it is hard.** The advancement rule is sequential by construction. Three
ways out, none free:

1. AllGather the position-ID tensor before sharding. Cheap on small batches,
   defeats CP for the multi-hundred-K context regime where CP earns its keep.
2. Precompute MRoPE position IDs out of band on rank 0, then scatter them
   alongside the token shards. Correct, but requires rewriting the data path
   end-to-end (dataset, collator, trainer's `prepare_context_parallel_input`).
3. Restructure MRoPE to be stateless — for example, encode image extents as
   side-channel offsets the encoder can sum locally. This changes the model's
   numeric definition and would require Qwen to bless a new revision. Open
   research.

## 4. Industry status

- **Megatron-LM.** `examples/multimodal/` and `megatron/training/arguments.py`
  carry an explicit guard against `context_parallel_size > 1` when images are
  present. Multimodal CP is listed as future work in commit history.
- **NeMo (NVIDIA).** Open issues on GitHub report training instability for
  CP + VLM; the recommended configuration is CP=1 with vision attached.
- **DeepSpeed.** ZeRO-3 plus Ulysses sequence parallel is documented for text.
  No VLM mode is officially documented for SP or CP.
- **HuggingFace Accelerate.** No CP. Sequence parallelism is limited to what
  FSDP provides for free.
- **OpenAI / Anthropic / Google.** Closed source. Public papers (Gemini,
  GPT-4V, Claude 3.5 Sonnet) do not describe CP across the vision path; the
  internal parallelism is presumably TPU- or cluster-specific and distinct
  from the open-source CP recipe.

## 5. What SOTA VLAs actually do (the workaround)

Every shipping open VLA paper sidesteps vision-CP by aggressive pre-LM
compression:

| Paper        | Compression                        | LM seq target |
|--------------|------------------------------------|---------------|
| AutoVLA (3B) | 2x2 spatial merge in ViT           | ~1.7 K tokens |
| DriveVLM (7B)| Q-Former, 64 query tokens          | ~1 K tokens   |
| EMMA (Gemini)| Single keyframe                    | small         |
| Apollo (7B)  | Per-frame downsample               | ~3–8 K        |
| LongVU (7B)  | DINOv2 cross-frame prune           | <8 K          |
| Wayve GAIA-2 | Latent diffusion (vision-only)     | small         |

Common pattern: **compress before the LM**, then run FSDP + TP + (optionally
Ulysses-style SP on the text side). Nobody ships vision-CP.

## 6. Path forward

For this project:

- **Short term.** Do not solve vision-CP. Use pre-LM compression
  (mean-pool / VTM / LongVU-style pruning) plus FSDP + TP + PP. Keep text-only
  CP available for the text-rich runs where it actually pays.
- **Long term research.** Stateless MRoPE, pre-scattered position-ID compute,
  or vision-vs-text mesh splits. Each is a publishable ML-systems contribution
  in its own right and is out of scope for the driving-VLA work.
- **Upstreamable to torchtitan.** The text-only CP wiring works and the test
  suite passes. The clean PR is: land the text-only CP wiring, and raise an
  explicit `NotImplementedError` when vision inputs are present. That matches
  Megatron's defensive stance and prevents silent miscompute.

## 7. Pointers

- CP wiring on the LM attention modules:
  `torchtitan_qwen25/torchtitan/models/qwen3_vl/parallelize.py`,
  `apply_cp_to_attention_module` call around line 319, with the docstring at
  line 59 explicitly noting that DTensor-incompatible boolean indexing
  (`masked_scatter`, `tensor[bool_mask]`) blocks the vision path.
- Test suite, five tests, text-only path passes:
  `torchtitan_qwen25/tests/test_cp_qwen3_vl.py`
  (`test_pp_plus_cp_raises_not_implemented`, `test_cp_only_path_calls_apply_cp`,
  `test_cp_disabled_no_apply_cp`, `test_apply_cp_to_attention_module_is_importable`,
  `test_cp_call_site_mirrors_qwen3_text_only`).
- Megatron-LM's VLM CP assert, for reference / parity:
  `Megatron-LM/megatron/training/arguments.py` (search `context_parallel_size`).
- Vision scatter site that breaks under CP:
  `torchtitan_qwen25/torchtitan/models/qwen3_vl/model.py:416`.
- MRoPE site that breaks under CP:
  `torchtitan_qwen25/torchtitan/models/qwen3_vl/model.py:122`.
