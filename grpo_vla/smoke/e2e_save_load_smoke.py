#!/usr/bin/env python3
"""E2E save -> load -> logit parity smoke for GRPO B.5' actor.

Per `feedback_e2e_save_load_smoke`: ANY training that adds nn.Modules outside
the base model needs a 5-step smoke + save + reload + parity check FIRST.

GRPO does not add new modules, but veRL wraps the actor in FSDP and writes
ckpts via its own format. A reload mismatch caught here saves 6-10 h of
overnight wall-clock.

Strategy (single GPU, single process; isolated from veRL):
  1. Load base B.5' ckpt into Qwen2.5-VL processor + model (bf16).
  2. Build ONE val sample via grpo_vla.dataset_adapter.build_verl_nuscenes_dataset.
  3. Construct a model input via processor(text=prompt, videos=clips, images=imgs).
  4. Forward -> capture logits_before (last 60 token positions).
  5. save_pretrained -> /workspace/.../grpo_vla/smoke/ckpt_resave/.
  6. Reload from resave dir.
  7. Forward identical batch -> logits_after.
  8. Assert max|logits_before - logits_after| < 1e-3.

Exit codes:
  0  PASS  -- prints "SMOKE PASS  delta=<float>"
  1  FAIL  -- prints "SMOKE FAIL  <reason>"
"""
from __future__ import annotations

import shutil
import sys
import traceback
from pathlib import Path

import torch

GRPO_DIR = Path("/workspace/DriveLM_VLM_Project/grpo_vla")
BASE_CKPT = Path(
    "/workspace/DriveLM_VLM_Project/checkpoints_qwen25/"
    "nusc_planning_b5prime_3cam_multimodal/final"
)
RESAVE_DIR = GRPO_DIR / "smoke" / "ckpt_resave"
CFG_PATH = GRPO_DIR / "configs" / "grpo_b5prime_3cam.yaml"
TOL = 1e-3  # bf16 round-trip tolerance


def _load_model_and_proc(path: Path):
    from transformers import AutoProcessor

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as ModelCls
    except Exception:
        from transformers import AutoModelForCausalLM as ModelCls  # type: ignore

    proc = AutoProcessor.from_pretrained(str(path), trust_remote_code=True)
    model = ModelCls.from_pretrained(
        str(path),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to("cuda:0").eval()
    return proc, model


def main() -> int:
    if not torch.cuda.is_available():
        print("SMOKE FAIL  no CUDA")
        return 1
    if not CFG_PATH.exists():
        print(f"SMOKE FAIL  missing config {CFG_PATH}")
        return 1

    # Import Agent C adapter
    sys.path.insert(0, str(GRPO_DIR))
    sys.path.insert(0, str(GRPO_DIR.parent))
    try:
        try:
            from grpo_vla.dataset_adapter import (  # type: ignore
                build_verl_nuscenes_dataset,
            )
        except Exception:
            from dataset_adapter import build_verl_nuscenes_dataset  # type: ignore
    except Exception:
        print("SMOKE FAIL  cannot import dataset_adapter")
        traceback.print_exc()
        return 1

    try:
        import yaml
        with open(CFG_PATH) as f:
            cfg = yaml.safe_load(f) or {}
        dcfg = cfg.get("data", cfg)
    except Exception:
        print("SMOKE FAIL  cannot parse config")
        traceback.print_exc()
        return 1

    try:
        proc, model = _load_model_and_proc(BASE_CKPT)
    except Exception:
        print("SMOKE FAIL  cannot load base ckpt")
        traceback.print_exc()
        return 1

    # Build one val sample
    try:
        ds = build_verl_nuscenes_dataset(dcfg, proc, split="val")
        sample = ds[0]
    except Exception:
        print("SMOKE FAIL  cannot build val sample (dataset_adapter)")
        traceback.print_exc()
        return 1

    # Build model inputs via the same processor
    mm = sample.get("multi_modal_data", {}) or {}
    videos = mm.get("video") or None
    images = mm.get("image") or None
    try:
        inputs = proc(
            text=[sample["prompt"]],
            videos=videos,
            images=images,
            return_tensors="pt",
        )
    except TypeError:
        # Older processor doesn't accept videos= ; try image-only
        flat_images = []
        for clip in (videos or []):
            flat_images.extend(clip)
        for img in (images or []):
            flat_images.append(img)
        inputs = proc(
            text=[sample["prompt"]],
            images=flat_images or None,
            return_tensors="pt",
        )

    inputs = {
        k: (v.to("cuda:0") if torch.is_tensor(v) else v) for k, v in inputs.items()
    }

    with torch.no_grad():
        out1 = model(**inputs)
        logits_before = out1.logits[:, -60:, :].detach().float().clone()

    # Save + reload
    if RESAVE_DIR.exists():
        shutil.rmtree(RESAVE_DIR)
    RESAVE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        model.save_pretrained(str(RESAVE_DIR), safe_serialization=True)
        proc.save_pretrained(str(RESAVE_DIR))
    except Exception:
        print("SMOKE FAIL  save_pretrained crashed")
        traceback.print_exc()
        return 1

    del model
    torch.cuda.empty_cache()

    try:
        _, model2 = _load_model_and_proc(RESAVE_DIR)
    except Exception:
        print("SMOKE FAIL  reload from RESAVE_DIR crashed")
        traceback.print_exc()
        return 1

    with torch.no_grad():
        out2 = model2(**inputs)
        logits_after = out2.logits[:, -60:, :].detach().float().clone()

    delta = (logits_before - logits_after).abs().max().item()
    print(f"logit max-abs-delta = {delta:.3e}  (tol={TOL:.0e})")
    if delta > TOL:
        print(f"SMOKE FAIL  logit drift {delta:.3e} > {TOL:.0e}")
        return 1
    print(f"SMOKE PASS  delta={delta:.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
