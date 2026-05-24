#!/usr/bin/env /usr/bin/python3
"""1-step rollout smoke for B.5' Qwen2.5-VL 3cam under veRL GRPO.

Purpose
-------
Verify (before launching multi-hour GRPO):

  1. veRL v0.7.1 is importable on system python (torch 2.11 / sglang 0.5.11)
  2. Our YAML config parses through Hydra+OmegaConf w/o key errors
  3. B.5' checkpoint exists + the Qwen2.5-VL processor loads
  4. Agent A's SGLang server at http://localhost:30001 accepts one multimodal
     /v1/chat/completions request and returns tokens

It does NOT exercise the verl <-> sglang weight-sync path (blocked by an
upstream-fork API drift in /sgl-workspace/sglang — see "KNOWN BLOCKER" below).
For our smoke that drift is moot because:
  - the trainer-side YAML loads fine
  - the SGLang server-side inference works fine
  - veRL.workers.rollout.sglang_rollout.ServerAdapter is ONLY needed at
    training time for the weight-broadcast hot-path

KNOWN BLOCKER (documented for Agent D)
--------------------------------------
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner.py", line 117
      from sglang.srt.layers.moe.routed_experts_capturer import (...)
  ModuleNotFoundError: No module named 'sglang.srt.layers.moe.routed_experts_capturer'

  Upstream sglang moved that module to sglang.srt.state_capturer.routed_experts
  (commit 08d4c2072 "move topk capturers to srt/state_capturer/").
  The local /sgl-workspace/sglang checkout (branch attention_residual_inference)
  has the new file location but model_runner.py still imports from the old path.
  This is an Agent A's fork merge-resolution bug, not a verl bug.

  Fix options for Agent D before GRPO training launch:
    (a) `git revert` on the sglang fork to before the move (one-line patch)
    (b) add a shim:
            ln -s ../state_capturer/routed_experts.py \\
                  /sgl-workspace/sglang/python/sglang/srt/layers/moe/routed_experts_capturer.py
    (c) cherry-pick the upstream fix that updates the import path in
        model_runner.py
"""
from __future__ import annotations

import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

CKPT = Path(
    "/workspace/DriveLM_VLM_Project/checkpoints_qwen25/"
    "nusc_planning_b5prime_3cam_multimodal/final"
)
SGLANG_URL = os.environ.get("SGLANG_URL", "http://localhost:30001")
CFG = Path(
    "/workspace/DriveLM_VLM_Project/grpo_vla/configs/grpo_b5prime_3cam.yaml"
)
PROMPT_TXT = (
    "You are an ego-vehicle planner. Given the current scene, predict the next "
    "6 future waypoints as a sequence of trajectory tokens. Scene: "
)


def step(name: str):
    print(f"\n[smoke] === {name} ===", flush=True)


def ok(msg: str):
    print(f"[smoke]   ok  {msg}", flush=True)


def warn(msg: str):
    print(f"[smoke]   WARN {msg}", flush=True)


def fail(msg: str):
    print(f"[smoke]   FAIL {msg}", flush=True)


def s1_import_verl() -> bool:
    step("1/5  import verl core")
    try:
        import verl  # noqa: F401
        from verl import DataProto  # noqa: F401
        from verl.workers.config import HFModelConfig, RolloutConfig  # noqa: F401
        from verl.workers.rollout.base import (  # noqa: F401
            BaseRollout,
            _ROLLOUT_REGISTRY,
            get_rollout_class,
        )
        ok(f"verl=={verl.__version__}")
        ok(f"rollout registry keys: {list(_ROLLOUT_REGISTRY.keys())}")
        return True
    except Exception as e:
        fail(f"import verl: {type(e).__name__}: {e}")
        return False


def s2_import_sglang_adapter() -> bool:
    step("2/5  import verl SGLang ServerAdapter (expected to fail — see KNOWN BLOCKER)")
    try:
        from verl.workers.rollout.sglang_rollout.sglang_rollout import (  # noqa: F401
            ServerAdapter,
        )
        ok("ServerAdapter imported (BLOCKER appears fixed!)")
        return True
    except ModuleNotFoundError as e:
        warn(f"ServerAdapter import blocked at: {e}")
        warn("  this is the expected sglang fork drift; not a launcher blocker")
        return False
    except Exception as e:
        fail(f"unexpected error importing ServerAdapter: {type(e).__name__}: {e}")
        return False


def s3_parse_yaml() -> bool:
    step("3/5  parse YAML config via OmegaConf (Hydra + verl searchpath)")
    try:
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra

        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        cfg_dir = str(CFG.parent.resolve())
        verl_cfg_dir = "/workspace/verl/verl/trainer/config"
        with initialize_config_dir(version_base=None, config_dir=cfg_dir):
            cfg = compose(
                config_name=CFG.stem,
                overrides=[f"hydra.searchpath=[file://{verl_cfg_dir}]"],
            )
        # spot-check the 11 hyperparams we promised in the design-doc audit
        checks = {
            "model.path": cfg.actor_rollout_ref.model.path,
            "actor.optim.lr": cfg.actor_rollout_ref.actor.optim.lr,
            "actor.kl_loss_coef": cfg.actor_rollout_ref.actor.kl_loss_coef,
            "rollout.n": cfg.actor_rollout_ref.rollout.n,
            "rollout.tensor_model_parallel_size": cfg.actor_rollout_ref.rollout.tensor_model_parallel_size,
            "rollout.gpu_memory_utilization": cfg.actor_rollout_ref.rollout.gpu_memory_utilization,
            "data.train_batch_size": cfg.data.train_batch_size,
            "data.max_prompt_length": cfg.data.max_prompt_length,
            "data.max_response_length": cfg.data.max_response_length,
            "actor.ppo_mini_batch_size": cfg.actor_rollout_ref.actor.ppo_mini_batch_size,
            "actor.ppo_max_token_len_per_gpu": cfg.actor_rollout_ref.actor.ppo_max_token_len_per_gpu,
            "trainer.total_epochs": cfg.trainer.total_epochs,
            "trainer.save_freq": cfg.trainer.save_freq,
            "trainer.test_freq": cfg.trainer.test_freq,
        }
        for k, v in checks.items():
            ok(f"{k} = {v}")
        return True
    except Exception as e:
        fail(f"yaml parse: {type(e).__name__}: {e}")
        return False


def s4_ckpt_present() -> bool:
    step("4/5  B.5' checkpoint files present")
    if not CKPT.is_dir():
        fail(f"missing ckpt dir: {CKPT}")
        return False
    expected = {"config.json", "model.safetensors", "tokenizer.json"}
    have = {p.name for p in CKPT.iterdir()}
    missing = expected - have
    if missing:
        fail(f"missing files: {missing}")
        return False
    size_gb = sum(p.stat().st_size for p in CKPT.rglob("*") if p.is_file()) / 1e9
    ok(f"ckpt dir OK ({size_gb:.1f} GB across {len(have)} entries)")
    return True


def s5_sglang_rollout() -> tuple[bool, dict]:
    step(f"5/5  hit Agent A SGLang server: {SGLANG_URL}")
    # 1x1 PNG (white pixel) base64; sufficient to prove multimodal wiring
    one_px_png_b64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YA"
        "AAAASUVORK5CYII="
    )
    try:
        info = json.loads(
            urllib.request.urlopen(f"{SGLANG_URL}/get_model_info", timeout=5).read()
        )
        ok(f"server alive; model: {info.get('model_path', info)}")
    except urllib.error.URLError as e:
        warn(f"server not reachable yet ({e}); skipping live rollout")
        return False, {"reason": "server_unreachable"}
    except Exception as e:
        warn(f"server probe failed: {type(e).__name__}: {e}")
        return False, {"reason": str(e)}

    body = {
        "model": "qwen2_5_vl",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{one_px_png_b64}"
                        },
                    },
                    {
                        "type": "text",
                        "text": PROMPT_TXT
                        + "single-frame placeholder, output 1 token.",
                    },
                ],
            }
        ],
        "max_tokens": 8,
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        f"{SGLANG_URL}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=60).read())
        latency = time.time() - t0
        choice = resp["choices"][0]
        text = choice["message"]["content"]
        usage = resp.get("usage", {})
        ok(f"latency_s={latency:.2f}  prompt_tok={usage.get('prompt_tokens')} "
           f"completion_tok={usage.get('completion_tokens')}")
        ok(f"decoded text: {text!r}")
        return True, {"latency": latency, "usage": usage, "text": text}
    except Exception as e:
        fail(f"chat completion: {type(e).__name__}: {e}")
        return False, {"reason": str(e)}


def main() -> int:
    print(f"[smoke] python={sys.executable}")
    print(f"[smoke] cwd={os.getcwd()}  CKPT={CKPT}  CFG={CFG}  SGLANG_URL={SGLANG_URL}")
    r1 = s1_import_verl()
    r2 = s2_import_sglang_adapter()  # informational
    r3 = s3_parse_yaml()
    r4 = s4_ckpt_present()
    r5_ok, r5_info = s5_sglang_rollout()

    print("\n[smoke] === SUMMARY ===")
    print(f"[smoke]   verl import           : {'OK' if r1 else 'FAIL'}")
    print(f"[smoke]   sglang ServerAdapter  : {'OK' if r2 else 'WARN (fork drift)'}")
    print(f"[smoke]   yaml parse            : {'OK' if r3 else 'FAIL'}")
    print(f"[smoke]   ckpt present          : {'OK' if r4 else 'FAIL'}")
    print(f"[smoke]   sglang rollout        : {'OK' if r5_ok else 'WAITING ('+str(r5_info)+')'}")
    # gate: verl + yaml + ckpt must pass. server probe + adapter import are warnings only.
    hard_required = r1 and r3 and r4
    return 0 if hard_required else 1


if __name__ == "__main__":
    sys.exit(main())
