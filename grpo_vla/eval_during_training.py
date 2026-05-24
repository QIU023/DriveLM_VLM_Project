#!/usr/bin/env python3
"""Mid-training val-L2 evaluator for GRPO B.5'.

Invoked every 50 steps by `launch_grpo_b5prime.sh` against the latest ckpt
under `<save_root>/global_step_<N>/actor/`. Runs N val samples through the
actor via SGLang's OpenAI-compatible /v1/chat/completions endpoint and
computes paper-aligned planning metrics.

Contracts (consumed):
  - grpo_vla.reward.compute_reward(response_token_ids, gt_wp, bbox_3d_list,
       ego_state, ...) -> dict with keys r_total, r_l2, r_collision,
       n_collisions, malformed, pred_wp.
  - grpo_vla.dataset_adapter.build_verl_nuscenes_dataset(cfg, processor,
       split='val') -> Dataset of dicts {prompt, multi_modal_data,
       extra_info}.

Results append to:
  /workspace/.../grpo_vla/logs/eval_curve.jsonl  (one line per call)
  /workspace/.../grpo_vla/logs/eval_curve.tb/    (tensorboard)

Usage (from launcher):
  /usr/bin/python3 eval_during_training.py \
      --step 150 \
      --ckpt /workspace/.../global_step_150/actor \
      --n-samples 200 \
      --sglang-url http://localhost:30001 \
      --config /workspace/.../grpo_vla/configs/grpo_b5prime_3cam.yaml
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

GRPO_DIR = Path("/workspace/DriveLM_VLM_Project/grpo_vla")
LOG_DIR = GRPO_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
EVAL_JSONL = LOG_DIR / "eval_curve.jsonl"
TB_DIR = LOG_DIR / "eval_curve.tb"


def _append_jsonl(row: Dict[str, Any]) -> None:
    with open(EVAL_JSONL, "a") as f:
        f.write(json.dumps(row) + "\n")


def _write_tb(step: int, metrics: Dict[str, float]) -> None:
    try:
        from torch.utils.tensorboard import SummaryWriter

        w = SummaryWriter(str(TB_DIR))
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and v == v:  # skip NaN
                w.add_scalar(f"val/{k}", float(v), int(step))
        w.flush()
        w.close()
    except Exception:
        traceback.print_exc()


def _pil_to_b64_png(img) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _build_openai_messages(sample: Dict[str, Any]) -> List[dict]:
    """Convert one VeRLNuScenesDataset sample to an OpenAI chat-completions
    payload. SGLang's /v1/chat/completions expects image_url blocks; videos
    are flattened to one image_url per frame in order."""
    prompt = sample["prompt"]
    mm = sample.get("multi_modal_data", {}) or {}
    content: List[dict] = []
    # Flatten any videos to image frames, then any standalone images
    for clip in mm.get("video", []) or []:
        for img in clip:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{_pil_to_b64_png(img)}"},
            })
    for img in mm.get("image", []) or []:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{_pil_to_b64_png(img)}"},
        })
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--ckpt", type=str, required=True,
                    help="Path to actor ckpt dir (informational; SGLang serves it)")
    ap.add_argument("--n-samples", type=int, default=200)
    ap.add_argument("--sglang-url", type=str, default="http://localhost:30001")
    ap.add_argument("--config", type=str,
                    default=str(GRPO_DIR / "configs" / "grpo_b5prime_3cam.yaml"))
    ap.add_argument("--model", type=str, default="actor")
    ap.add_argument("--timeout-s", type=int, default=900)
    args = ap.parse_args()

    sys.path.insert(0, str(GRPO_DIR))
    sys.path.insert(0, str(GRPO_DIR.parent))  # allow grpo_vla.reward import
    try:
        from grpo_vla.reward import compute_reward  # type: ignore
        from grpo_vla.dataset_adapter import (  # type: ignore
            build_verl_nuscenes_dataset,
        )
    except Exception:
        try:
            from reward import compute_reward  # type: ignore
            from dataset_adapter import build_verl_nuscenes_dataset  # type: ignore
        except Exception:
            print("EVAL FAIL  cannot import reward + dataset_adapter")
            traceback.print_exc()
            return 1

    try:
        import numpy as np
        import requests
        import yaml
        from transformers import AutoProcessor
    except Exception:
        print("EVAL FAIL  missing numpy/requests/yaml/transformers")
        traceback.print_exc()
        return 1

    # Build the val dataset
    try:
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
        # veRL config nests data section under `data`; fall back to flat.
        dcfg = cfg.get("data", cfg)
        # processor only used for chat-template render (text-only).
        proc_path = (
            cfg.get("actor_rollout_ref", {}).get("model", {}).get("path")
            or cfg.get("model_path")
            or args.ckpt
        )
        processor = AutoProcessor.from_pretrained(proc_path, trust_remote_code=True)
        ds = build_verl_nuscenes_dataset(dcfg, processor, split="val")
    except Exception:
        print("EVAL FAIL  cannot build val dataset")
        traceback.print_exc()
        return 1

    n_total = min(args.n_samples, len(ds))
    print(f"eval step={args.step}  n_total={n_total}  ckpt={args.ckpt}")

    t0 = time.time()
    chat_ep = args.sglang_url.rstrip("/") + "/v1/chat/completions"
    l2_list: List[float] = []
    reward_list: List[float] = []
    coll_list: List[int] = []
    n_malformed = 0
    n_done = 0

    # We need to tokenize the response back to ids for compute_reward.
    # The processor's tokenizer is sufficient.
    try:
        tok = processor.tokenizer  # type: ignore[attr-defined]
    except Exception:
        tok = None

    for i in range(n_total):
        if time.time() - t0 > args.timeout_s:
            print(f"EVAL TIMEOUT after {n_done}/{n_total}")
            break
        try:
            sample = ds[i]
        except Exception:
            n_done += 1
            continue
        try:
            messages = _build_openai_messages(sample)
            r = requests.post(
                chat_ep,
                json={
                    "model": args.model,
                    "messages": messages,
                    "temperature": 0.0,
                    "max_tokens": 50,
                },
                timeout=60,
            )
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
        except Exception:
            n_malformed += 1
            n_done += 1
            continue

        # Tokenize the response to feed the reward fn
        if tok is None:
            n_malformed += 1
            n_done += 1
            continue
        resp_ids = tok.encode(content, add_special_tokens=False)

        extra = sample["extra_info"]
        gt_wp = np.asarray(extra["gt_waypoints"], dtype=np.float32)
        try:
            out = compute_reward(
                resp_ids,
                gt_wp,
                extra.get("bbox_3d_list", []),
                extra.get("ego_state", {"speed_mps": 0.0}),
                horizon_s=float(extra.get("horizon_s", 3.0)),
            )
        except Exception:
            n_malformed += 1
            n_done += 1
            continue

        if out.get("malformed"):
            n_malformed += 1
        else:
            l2_list.append(-float(out["r_l2"]))  # r_l2 is negative L2; we log +L2
            coll_list.append(int(out.get("n_collisions", 0)))
        reward_list.append(float(out["r_total"]))
        n_done += 1

    elapsed = time.time() - t0
    metrics = {
        "n": n_done,
        "n_malformed": n_malformed,
        "malformed_rate": (n_malformed / max(n_done, 1)),
        "l2_avg": float(sum(l2_list) / len(l2_list)) if l2_list else float("nan"),
        "collision_rate": (
            float(sum(coll_list) / len(coll_list)) if coll_list else float("nan")
        ),
        "reward_mean": (
            float(sum(reward_list) / len(reward_list)) if reward_list else float("nan")
        ),
        "elapsed_s": elapsed,
    }
    row = {"step": int(args.step), "ckpt": args.ckpt, **metrics}
    _append_jsonl(row)
    _write_tb(int(args.step), metrics)
    print(json.dumps(row, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
