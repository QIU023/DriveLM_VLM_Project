# PR 001 — torchtitan step-decay LR scheduler

**Repo**: pytorch/torchtitan
**Branch**: `step_decay_lr_scheduler` (on QIU023/torchtitan fork)
**Commit**: `b521799d2d0fd0a3df3bc73d339ac3076990573d`
**Status**: Implemented + tested locally + pushed to `QIU023/torchtitan/step_decay_lr_scheduler`. **PR not yet opened** (waiting for our own validation in a Track A SFT run first before sending upstream).

## Why we need it

AutoVLA paper (and several other paper-grade VLA SFT recipes) prescribes a
**step-decay LR schedule**: `lr *= 0.98` every `decay_freq` optimizer steps
*after* warmup. This is *not* the same as the smooth linear / cosine / sqrt
warmup-stable-decay (WSD) family that torchtitan already ships in
`LRSchedulersContainer`.

Concrete deviation we hit: for our Qwen3-VL 8B Track A SFT
(`qwen3_vl_8b_planning_fsdp.toml`) we parked `lr_step_freq=696` in the TOML
and approximated with `decay_type="linear"` while waiting for upstream
support. Linear decay over the same horizon undershoots LR in the first
~70 % of training and overshoots at the tail vs paper-grade step-decay; the
end-of-training LR differs by ~2x. We want the real schedule.

torchtitan today exposes `decay_type ∈ {"linear","sqrt","cosine"}` via a
`LambdaLR` factory inside `LRSchedulersContainer.Config.build()`. No
existing option produces a multiplicative step-decay.

## What changed

| File | Change | LOC |
|---|---|---|
| `torchtitan/components/lr_scheduler.py` | add `decay_type="step"` Literal option, add `decay_freq: int` + `decay_factor: float` config fields, add `linear_warmup_step_decay` lambda branch in `Config.build()`, input validation | +89 / -1 |
| `tests/unit_tests/test_lr_scheduler_step_decay.py` | **new** unit test (5 cases): warmup ramp + step decay schedule, `min_lr_factor` floor, `decay_ratio` ignored under step decay, invalid `decay_factor` raises, invalid `decay_freq` raises | +161 |

Total diff: 2 files, +250 / -1.

## Design highlights

- **Composes with existing warmup.** The new `linear_warmup_step_decay`
  lambda reuses the same `(current_step + 1) / warmup_steps` ramp as the
  WSD path, so step-decay users get the exact same warmup curve as the
  linear/cosine users — no surprises during warmup.

- **Same plumbing as existing schedulers.** Step-decay is implemented as a
  branch in the existing `LRSchedulersContainer.Config.build()` and uses
  the same `LambdaLR` wrapper. No new scheduler subclass, no changes to
  `step()` / `state_dict()` / `load_state_dict()`. Checkpoint resume
  continues to work because only `last_epoch` is stateful (the
  multiplicative factor is recomputed deterministically from
  `last_epoch`).

- **`decay_ratio` is ignored under step decay** (a warning is logged if
  the user sets it). Step-decay runs continuously from end-of-warmup
  through the final training step, parameterised entirely by
  `decay_freq` + `decay_factor`. We chose not to silently overload
  `decay_ratio` to mean "stop decaying after this fraction"; that would
  be a different schedule and deserves its own flag.

- **`min_lr_factor` is still respected** as a floor on the post-warmup
  factor, mirroring its behaviour in the WSD branch. Tested explicitly
  in `test_min_lr_factor_floor`.

- **Why not `torch.optim.lr_scheduler.StepLR` directly?** torchtitan
  composes all of warmup + decay inside a single `LambdaLR` so that
  `LRSchedulersContainer` only needs one scheduler per optimizer.
  Chaining a `LinearLR` + `StepLR` via `SequentialLR` would have required
  modifying `LRSchedulersContainer.__init__` and the resharding-friendly
  state_dict assumption (see class docstring "Limitations"). Re-using
  the existing `LambdaLR` pipe is the minimal-blast-radius design.

## TOML example

```toml
[lr_scheduler]
warmup_steps = 174
decay_type   = "step"
decay_freq   = 696    # every 696 optimizer steps after warmup
decay_factor = 0.98   # multiply lr by 0.98
```

This produces:
- Steps 0 … 173: linear warmup from `lr / 174` up to `lr`.
- Steps 174 … 869: lr unchanged (interval 0).
- Steps 870 … 1565: lr × 0.98.
- Steps 1566 … 2261: lr × 0.98^2.
- … and so on for the rest of training.

## Test command

```bash
cd /workspace/DriveLM_VLM_Project/torchtitan_qwen25
/usr/bin/python3 -m pytest tests/unit_tests/test_lr_scheduler_step_decay.py -v
# 5 passed
/usr/bin/python3 -m pytest tests/unit_tests/test_lr_scheduler.py -v
# 6 passed (no existing tests broken)
```

## Reviewer hints (for when we open the PR upstream)

- **Tag**: torchtitan maintainers (likely `@fegin`, `@tianyu-l`, `@wconstab` —
  check recent contributors to `torchtitan/components/lr_scheduler.py` and
  the WSD PR https://github.com/pytorch/torchtitan/pulls?q=warmup+stable+decay
  before sending).
- **Highlight**:
  1. Composes with the existing linear warmup — same ramp formula as the
     WSD path, no schedule-discontinuity at the warmup/decay boundary.
  2. Zero changes to `LRSchedulersContainer.{step,state_dict,load_state_dict}`,
     so the resharding-friendly state_dict invariant (docstring
     "Limitations") still holds.
  3. New `decay_type` value is purely additive; `"linear"` / `"sqrt"` /
     `"cosine"` paths are byte-identical (we ran the pre-existing 6 tests
     and they all pass).
  4. Megatron-LM exposes the same schedule as
     `--lr-decay-style=step` with `--lr-decay-iters` controlling
     `decay_freq`-equivalent; we are catching up to that surface.
- **Bikeshed flags we'd accept feedback on**:
  - Naming: `decay_type="step"` vs `"step_decay"`. The user spec
    proposed `"step_decay"`; we used `"step"` to be consistent with the
    existing one-word Literal values (`linear`, `sqrt`, `cosine`). Easy
    to flip.
  - Default `decay_factor=0.98`: chosen because that's the AutoVLA
    paper value and the most common pattern in long-running SFT.
    Could equally well default to `0.5` (Keras' default).
  - Currently `decay_ratio` is silently ignored under step-decay with
    a warning; we could instead raise a `ValueError` if the user
    sets both. We chose warning so that copy-paste-then-flip-decay-type
    works.

## Caveats / unresolved tradeoffs

- **Checkpoint resume not explicitly tested in this PR.** Resume
  *should* work because `LambdaLR.state_dict()` only persists
  `last_epoch`, and the multiplicative factor is recomputed from
  `last_epoch` deterministically on every `step()`. But we did not write
  an integration test that saves a ckpt mid-training and reloads —
  we'll add one (or rely on Track A run logs showing continuous LR
  across resume) before opening the PR. Flagging here so the maintainers
  can flag it too.
- **Interaction with `total_steps` override.** `total_steps` is honoured
  only insofar as warmup is clamped to `total_steps` (existing
  behaviour). Step-decay itself does not care about total_steps because
  it never "finishes" — it just keeps applying the factor every
  `decay_freq` steps until training stops. This is intentional but
  worth a one-line note in upstream docs.
- **No `LambdaLR` -> `SequentialLR(LinearLR, StepLR)` refactor.** We
  considered using PyTorch's built-in `StepLR` chained after a
  `LinearLR` via `SequentialLR`, but that would have required
  refactoring `LRSchedulersContainer.__init__` to support multiple
  scheduler types per optimizer (currently hard-coded to `LambdaLR`).
  That's a bigger change than this PR wants to scope; reusing
  `LambdaLR` keeps the diff small and the state_dict semantics
  unchanged. If maintainers prefer the `SequentialLR` route, we can
  follow up.

## Local refs

- **Branch**: `step_decay_lr_scheduler` (off `qwen25_vl_video_vla`)
- **Commit SHA**: `b521799d2d0fd0a3df3bc73d339ac3076990573d`
- **Remote**: `git@github.com:QIU023/torchtitan.git`
- **Pushed**: yes (`origin/step_decay_lr_scheduler` exists, tracking set up)
- **Files**:
  - `/workspace/DriveLM_VLM_Project/torchtitan_qwen25/torchtitan/components/lr_scheduler.py`
  - `/workspace/DriveLM_VLM_Project/torchtitan_qwen25/tests/unit_tests/test_lr_scheduler_step_decay.py`

## Next steps (before opening upstream PR)

1. Validate against a Track A SFT smoke run (~200 steps) with
   `decay_type="step"`, `decay_freq=696`, `decay_factor=0.98`, confirm
   LR trace logged from `tensorboard` / `wandb` matches the analytical
   curve.
2. Add a tiny ckpt-resume integration test (save at step 50, reload,
   verify LR at step 51 matches the no-resume trace).
3. Switch `qwen3_vl_8b_planning_fsdp.toml` from `decay_type="linear"`
   + parked `lr_step_freq=696` to the new `decay_type="step"` plumbing
   — **this is the follow-up PR, intentionally not part of this diff**.
4. Open the upstream PR with the contents above.
