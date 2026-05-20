# PR 011 — train_lora.py: real planning metrics in mid-training validate()

**Repo**: QIU023/DriveLM_VLM_Project (internal; not a third-party upstream)
**Branch**: `train_lora_l2_val` (off `qwen25_vl_video_vla`)
**Status**: Implemented + CPU sanity-tested. Slated for Track A.1 / A.2 / A.3
launches (NOT applied to the in-flight A.0 run which is on the older
train_lora.py).

## Why we need it

Today `scripts/train_lora.py::validate()` runs every `val_every` steps on
`val_batches=20` micro-batches and reports only **teacher-forced cross-entropy
+ token accuracy** plus a teacher-forced L2 over trajectory bins. Teacher-
forced L2 is **highly optimistic**: the next bin's logit is computed against
the *true* previous bin, so a model with a 30 % single-bin error still scores
near-paper L2 because every step is reset. The paper-comparable metrics
require **autoregressive greedy decode**: the model must roll out its own
trajectory from the prompt with no oracle feedback.

The full greedy-decode + dequantize + UniAD-port collision pipeline already
exists in `scripts/planning_eval.py` (the offline eval), but it is run
post-training only. The mid-training trainer is blind to the real numbers —
we discover collapse / divergence hours later.

This PR ports the planning_eval.py logic into the trainer's `validate()` so:

* Mid-training validation prints L2_1s / L2_2s / L2_3s / L2_avg (TemAvg
  protocol matches AutoVLA / VAD; NoAvg protocol matches UniAD).
* Collision rate at 1s / 2s / 3s using the UniAD-port BEV overlap check
  (ego footprint 4.084 m × 1.85 m, +0.5 m forward shift, agents reframed to
  the current ego frame — the three bug-fixes we already validated for
  planning_eval.py).
* DP aggregation across all FSDP ranks via `accelerator.gather_object` (per-
  sample lists so the mean is exact across ranks).
* Light pass on val_batches=20 by default (~25 s overhead at FSDP=8); opt-in
  `val_full_eval: true` walks all 5119 val samples (~3 min).

## What changed

| File | Change | LOC delta |
|---|---|---|
| `scripts/train_lora.py` | Added `_strip_meta`, `_greedy_decode_l2_collision`, extended `validate()` + collate_fn passthroughs + CLI flags `--val-full-eval` / `--no-val-planning-l2`. | +~210 / -~5 |
| `scripts/planning_dataset.py` | Emit `_meta_prompt_len` + `_meta_action_len` from `__getitem__` (the index of `<traj_start>` in input_ids, needed to slice prompt-only sequences for greedy generate). | +6 / -0 |
| `configs/nuscenes_planning_full.yaml` | Added `val_planning_l2: true`, `val_full_eval: false`, `val_greedy_max_new_tokens: 20` defaults. | +6 / -0 |
| `configs/nuscenes_planning_3cam_full.yaml` | Same defaults, with comment about the +25 s cost. | +7 / -0 |
| `docs/upstream_prs/011_train_lora_l2_val.md` | This doc. | +n/a |

## Implementation approach

### 1. Dataset meta passthrough
`PlanningDataset.__getitem__` already emits `_meta_waypoints` (GT future xy
in ego frame), `_meta_valid_mask`, and `_meta_token`. We add
`_meta_prompt_len = action_insert_start` so the validate loop can recover the
prompt-only token slice (`input_ids[:_meta_prompt_len]`) at greedy-decode
time. `collate_fn` now forwards these as per-sample Python lists / tensor
lists; `_strip_meta` removes them before the model forward call (the legacy
`forward_with_compression` and `forward_with_video_xframe_compression` both
do `model(**batch)` and would fail on Python-list inputs).

### 2. Two-pass validate
The val loader is consumed once and each batch's CPU copy is cached. After
the standard teacher-forced loss/acc pass completes:

```python
gd_temavg, gd_noavg, gd_coll = _greedy_decode_l2_collision(
    model, cached_batches, val_dataset, processor, device,
    traj_tok, traj_cfg, max_new_tokens=20,
)
```

For each cached batch we:
1. Slice `input_ids[:_meta_prompt_lens[j]]` per sample.
2. Left-pad to a common length (so newly-generated tokens align on the
   right column for every row — matches planning_eval.py's batched-generate
   logic).
3. `model.generate(do_sample=False, num_beams=1, ...)`.
4. Slice off the prompt, find `[<traj_start> ... <traj_end>]`, dequantize via
   `TrajectoryTokenizer.decode` to (Δx, Δy) metres.
5. Compute TemAvg + NoAvg L2 against GT (from `_meta_waypoints`).
6. Resolve current sample + future infos via `val_dataset.tok2idx[token]` →
   `val_dataset.infos[base_idx]` + `val_dataset._walk_future(...)`, run the
   UniAD-port collision check.

### 3. Cross-rank aggregation
Each rank only sees its 1/world_size shard of val (Accelerator shards the
dataloader). We collect **per-sample lists** locally, then
`accelerator.gather_for_metrics(use_gather_object=True)` (or `dist.gather_object`
fallback) merges them on rank 0. Mean is computed across the merged list so
there's no numerical drift from per-rank averaging.

### 4. CLI / config knobs
* `val_planning_l2: true` (yaml) — enables the greedy-decode pass for
  planning configs.
* `val_full_eval: false` (yaml) or `--val-full-eval` (cli) — walk entire val
  set vs `val_batches=20`.
* `val_greedy_max_new_tokens: 20` — generate budget. Enough for 14
  trajectory tokens (1 start + 12 bins + 1 end) with 6-token slack.
* `--no-val-planning-l2` — runtime kill switch if greedy decode breaks
  somehow (e.g. transformers version regression in `model.generate`).

### 5. Output / logging
The existing tqdm.write line at val time now appends collision metrics + the
`n_greedy` sample count + the TF L2 (under `tf_L2_avg`) so we can directly
compare TF-vs-rollout L2 every val. wandb log keys also gain
`val_collision_1s/2s/3s/avg`, `val_noavg_L2_avg`, `val_tf_L2_avg`.

## Val cost analysis

* **Light pass (val_batches=20, default)**:
  per-rank `20 × batch_size=4 = 80 samples`. 8 ranks = 640 samples seen, ~12.5
  % of the 5119 val set. Greedy decode of 14 tokens at 2-cam vision context
  ≈ 25 s (from the planning_eval.py 8-GPU benchmark of 3 min for 5119
  samples). Acceptable on a 50-step val cadence.
* **Full pass (val_full_eval=true)**:
  ~3 min on 8 GPUs (planning_eval.py 8-GPU DP benchmark). Run at end-of-
  training only via `--val-full-eval`.

## Open issues / known limitations

1. **FSDP `summon_full_params`**: We rely on `model.eval()` + `model.generate()`
   to handle FSDP unsharding internally. Accelerate ≥ 1.x + transformers
   ≥ 5.x handle this through the FSDP wrap's `forward` path with autoshard
   summoning. If we see `AttributeError: 'NoneType' object has no attribute ...`
   from generate(), we'd need to wrap the call in
   `FSDP.summon_full_params(model, recurse=True, writeback=False)`. NOT
   needed at the transformer / accelerate versions pinned in our env, but
   worth re-verifying on the first 8 GPU run.
2. **3-cam vs 1-cam memory**: with 3 cams × 4 frames = ~1680 visual tokens
   + prompt + 14 traj tokens, the per-sample greedy decode peak is ~14×
   forward of a ~2400-token context. Per-device val_batch_size=4 is the
   same as training; if greedy decode blows VRAM we may need
   `val_batch_size_override: 2` (not added in this PR — surface as needed).
3. **Light pass bias**: L2 on the first 20 batches × 4 batch × 8 ranks =
   640 samples is sample 0..640 in val (PlanningDataset is shuffle=False).
   This is a fixed deterministic slice, so trend tracking is meaningful but
   the absolute number is slightly biased vs the full 5119. Final eval via
   `val_full_eval` removes the bias.
4. **A.0 isolation**: A.0 launched on the unchanged train_lora.py and will
   NOT see these metrics until the next restart. A.1 / A.2 / A.3 (next
   launches) must use the `train_lora_l2_val` branch — document this in the
   launcher scripts.
5. **No automatic end-of-training full eval**: We do NOT auto-trigger
   `--val-full-eval` at train end. Launchers should add an explicit
   `scripts/launch_planning_eval_dp.sh <ckpt>` step (which is what we
   already do today). Keeping this out of train_lora.py preserves the
   1-process / 1-job principle.

## Testing

CPU-only sanity (no model.generate, no GPU):

```
$ python3 -c "from trajectory_tokenizer import ...; from planning_eval import ..."
PASS: tokenizer round-trip residual ≤ 0.043 m (bin quantization)
PASS: L2 TemAvg / NoAvg finite for synthetic ±0.1 m offset prediction
PASS: greedy-decode-style self-decode of GT tokens yields L2 ~= 0.04 m
```

GPU smoke (TODO before A.1 launch): run `train_lora.py` 1 step on the smoke
config with val_every=1, val_batches=2 and verify the printout contains
collision_avg + n_greedy.
