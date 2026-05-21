"""Smoke test for evaluate_planning_l2_collision (CPU-only).

Verifies:
  1. evaluate_planning_l2_collision is importable from planning_eval with
     the expected signature.
  2. The expected-key contract is honoured on the trivial num_samples=0
     fast path (which exercises argument plumbing + return-dict shape
     without needing a real model forward).
  3. The standalone CLI main() still imports cleanly (no regression).
  4. The qformer/pixelshuffle distinction is enforced:
       - external_projector=None, projector_type=None -> vanilla path (OK)
       - projector_type='pixelshuffle' raises NotImplementedError as spec'd.

We deliberately do NOT exercise an end-to-end greedy-decode of real
Qwen2.5-VL on CPU — that would take many minutes per sample and isn't
useful as a "smoke". The real validation comes from the in-loop
[VAL-FULL] line on the next 8xGPU training launch.
"""
from __future__ import annotations

import inspect
import os
import sys
from types import SimpleNamespace

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)


def main() -> int:
    print("[smoke] step 1: import evaluate_planning_l2_collision")
    from planning_eval import (
        evaluate_planning_l2_collision,
        _StandaloneAcceleratorShim,
        main as cli_main,  # noqa: F401  (smoke 3: just ensure the CLI still parses)
    )

    print("[smoke] step 2: verify signature")
    sig = inspect.signature(evaluate_planning_l2_collision)
    expected = {
        "model", "processor", "val_dataset", "accelerator",
        "external_projector", "projector_type", "batch_size",
        "num_samples", "max_new_tokens", "video_fps",
        "planning_cams", "silent",
    }
    actual = set(sig.parameters.keys())
    missing = expected - actual
    extra = actual - expected
    assert not missing, f"signature missing params: {missing}"
    assert not extra, f"signature has unexpected extra params: {extra}"
    # All optional params (everything after val_dataset/accelerator) must be kw-only.
    kw_only_required = expected - {"model", "processor", "val_dataset", "accelerator"}
    for name in kw_only_required:
        p = sig.parameters[name]
        assert p.kind == inspect.Parameter.KEYWORD_ONLY, (
            f"param {name} expected KEYWORD_ONLY, got {p.kind}"
        )
    print(f"        signature OK: {sig}")

    print("[smoke] step 3: num_samples=0 fast-path returns dict with expected keys")
    # A stub dataset with __len__ = 0 -> evaluate function takes the n_total=0
    # early return branch (does NOT touch model/processor at all). This still
    # exercises:
    #   - accelerator-shim attribute reads (.device, .process_index, ...)
    #   - the expected-keys return contract that train_lora's [VAL-FULL] log
    #     line depends on (L2_avg / L2_1s/2s/3s / collision_*).

    class _EmptyDataset:
        def __init__(self):
            self.planning_cams = ["CAM_FRONT"]
            self.num_future = 6

        def __len__(self):
            return 0

    acc_shim = _StandaloneAcceleratorShim(device=torch.device("cpu"), rank=0, world_size=1)
    out = evaluate_planning_l2_collision(
        model=None,
        processor=None,
        val_dataset=_EmptyDataset(),  # type: ignore[arg-type]
        accelerator=acc_shim,
        external_projector=None,
        projector_type=None,
        batch_size=2,
        num_samples=0,
        silent=True,
    )
    expected_keys = {
        "L2_avg", "L2_1s", "L2_2s", "L2_3s",
        "noavg_L2_avg", "noavg_L2_1s", "noavg_L2_2s", "noavg_L2_3s",
        "collision_avg", "collision_1s", "collision_2s", "collision_3s",
        "n_scored", "wall_seconds", "protocol_l2",
    }
    missing_out = expected_keys - set(out.keys())
    assert not missing_out, f"output dict missing keys: {missing_out}"
    assert out["n_scored"] == 0, out
    print(f"        OK ({len(out)} keys returned, n_scored=0)")

    print("[smoke] step 4: external_projector=None routes through vanilla path")
    # Already covered above (we passed external_projector=None and the function
    # did not raise on the projector-type gating). Explicit re-check for clarity:
    assert "L2_avg" in out, "vanilla path must produce L2_avg key"
    print("        OK")

    print("[smoke] step 5: pixelshuffle/resampler stubs raise NotImplementedError")
    # The function checks projector_type='pixelshuffle' BEFORE the empty-ds
    # early-return only if external_projector is not None. We need a real-ish
    # projector stub to trigger the check; use a no-op nn.Module so the
    # not-None test fires.
    class _DummyProjector(torch.nn.Module):
        num_queries = 64

    for bad_type in ("pixelshuffle", "resampler"):
        try:
            evaluate_planning_l2_collision(
                model=None,
                processor=SimpleNamespace(tokenizer=SimpleNamespace(
                    padding_side="right",
                    convert_tokens_to_ids=lambda _t: 0,
                )),
                val_dataset=_EmptyDataset(),  # type: ignore[arg-type]
                accelerator=acc_shim,
                external_projector=_DummyProjector(),
                projector_type=bad_type,
                batch_size=1,
                num_samples=1,
                silent=True,
            )
        except NotImplementedError as e:
            assert bad_type in str(e), f"expected '{bad_type}' in error, got {e!r}"
            print(f"        OK ({bad_type} -> NotImplementedError as spec'd)")
            continue
        # If we got here, no NIE was raised — fail loudly.
        raise AssertionError(f"projector_type={bad_type!r} did NOT raise NotImplementedError")

    print("[smoke] ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
