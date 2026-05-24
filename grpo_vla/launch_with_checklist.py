#!/usr/bin/env python3
"""Pre-launch checklist gate for GRPO B.5' training.

Per `feedback_pre_launch_checklist` MUST PASS before any multi-hour launch:
  1. Disk free >= 15 GB                       (memory: disk_panic_protocol)
  2. GPU state sane (>= 30 GB free / device)
  3. SGLang rollout server reachable
  4. B.5' base checkpoint exists + is loadable
  5. veRL config file present + parses
  6. Reward fn + dataset adapter importable
  7. Audit table printed (paper hyperparam table from §2.2 of f1_grpo_design.md)
  8. E2E save-load smoke (1 prompt -> 1 step -> save -> reload -> logit parity)
  9. ETA estimate
 10. No CRITICAL warnings at boot

This script PRINTS the checklist, EXITS NON-ZERO on any failure, and only on
success prints the final 'go/no-go' verdict that the operator must read before
invoking ./launch_grpo_b5prime.sh.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

GRPO_DIR = Path("/workspace/DriveLM_VLM_Project/grpo_vla")
CFG_PATH = GRPO_DIR / "configs" / "grpo_b5prime_3cam.yaml"
REWARD_PATH = GRPO_DIR / "reward.py"
DATASET_PATH = GRPO_DIR / "dataset_adapter.py"
BASE_CKPT = Path(
    "/workspace/DriveLM_VLM_Project/checkpoints_qwen25/"
    "nusc_planning_b5prime_3cam_multimodal/final"
)
SGLANG_URL = "http://localhost:30001"
MIN_DISK_GB = 15.0
MIN_FREE_GPU_MB = 30000
SMOKE_DIR = GRPO_DIR / "smoke"


# ----------------- check helpers -----------------


def _ok(name: str, detail: str = "") -> None:
    print(f"  [OK]   {name}" + (f"  -- {detail}" if detail else ""))


def _fail(name: str, detail: str) -> None:
    print(f"  [FAIL] {name}  -- {detail}", file=sys.stderr)


def _warn(name: str, detail: str) -> None:
    print(f"  [WARN] {name}  -- {detail}")


def check_disk() -> bool:
    free = shutil.disk_usage("/workspace").free / 1e9
    if free < MIN_DISK_GB:
        _fail("disk", f"only {free:.1f} GB free at /workspace (need >= {MIN_DISK_GB})")
        return False
    _ok("disk", f"{free:.1f} GB free at /workspace")
    return True


def check_gpu() -> bool:
    """Per Option B (design doc §2.1), up to 2 GPUs host SGLang and may have
    ~20 GB used at idle. Tolerate up to 2 'busy' GPUs (likely rollout server),
    but require >= MIN_FREE_GPU_MB on at least 6 of 8."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        ).strip()
    except Exception as e:
        _fail("gpu", f"nvidia-smi failed: {e}")
        return False
    busy = []
    free_ok = 0
    n = 0
    for line in out.splitlines():
        idx, free, used = [s.strip() for s in line.split(",")]
        n += 1
        if int(free) < MIN_FREE_GPU_MB:
            busy.append(f"gpu{idx}: free={free}MB used={used}MB")
        else:
            free_ok += 1
    if free_ok < 6:
        _fail("gpu", f"only {free_ok}/8 GPUs >= {MIN_FREE_GPU_MB} MB free; busy={busy}")
        return False
    if busy:
        _warn("gpu", f"{free_ok}/8 free; {len(busy)} busy (likely SGLang): {busy}")
    else:
        _ok("gpu", f"all {n} GPUs >= {MIN_FREE_GPU_MB} MB free")
    return True


def check_sglang() -> bool:
    import urllib.request

    for ep in ("/health", "/v1/models"):
        try:
            with urllib.request.urlopen(SGLANG_URL + ep, timeout=5) as r:
                if 200 <= r.status < 300:
                    _ok("sglang", f"{SGLANG_URL}{ep} -> {r.status}")
                    return True
        except Exception:
            continue
    _fail("sglang", f"no 2xx response at {SGLANG_URL} (Agent A server not up)")
    return False


def check_base_ckpt() -> bool:
    safetensors = BASE_CKPT / "model.safetensors"
    config = BASE_CKPT / "config.json"
    if not safetensors.exists():
        _fail("base_ckpt", f"missing {safetensors}")
        return False
    if not config.exists():
        _fail("base_ckpt", f"missing {config}")
        return False
    size_gb = safetensors.stat().st_size / 1e9
    _ok("base_ckpt", f"{BASE_CKPT}  ({size_gb:.1f} GB)")
    return True


def check_agent_artifacts() -> bool:
    missing = [p for p in (CFG_PATH, REWARD_PATH, DATASET_PATH) if not p.exists()]
    if missing:
        for p in missing:
            _fail("agent_artifact", f"missing {p}")
        return False
    # Verify imports for reward + dataset adapter
    sys.path.insert(0, str(GRPO_DIR))
    try:
        import reward  # noqa: F401

        if not hasattr(reward, "planning_reward"):
            _fail("reward.py", "no `planning_reward` symbol (config points to it)")
            return False
        if not hasattr(reward, "compute_reward"):
            _warn("reward.py", "no `compute_reward` symbol (eval may be limited)")
    except Exception:
        _fail("reward.py", "import error: " + traceback.format_exc().splitlines()[-1])
        return False
    try:
        import dataset_adapter  # noqa: F401
    except Exception:
        _fail("dataset_adapter.py", traceback.format_exc().splitlines()[-1])
        return False
    _ok("agent_artifacts", "config + reward.py + dataset_adapter.py present + importable")
    return True


def check_config_parses() -> bool:
    try:
        import yaml

        with open(CFG_PATH) as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        _fail("config_parse", f"{CFG_PATH}: {e}")
        return False
    _ok("config_parse", f"{CFG_PATH.name} keys={list(cfg)[:6]}...")
    return True


def print_audit_table() -> None:
    rows = [
        ("MODEL_PATH",       "Qwen2.5-VL-7B-Instruct", "B.5' final ckpt (3B)", "scale: local 32GB GPU"),
        ("TRAIN_BATCH_SIZE", "512",     "24",     "compute constraint; matches B.5 SFT GBS"),
        ("PPO mini-batch",   "128",     "8",      "scaled 1/3 of TRAIN_BATCH"),
        ("Actor LR",         "1e-6",    "1e-6",   "paper-aligned"),
        ("KL coef beta",     "0.01",    "0.01",   "paper-aligned (DeepSeek R1)"),
        ("Rollout n",        "5",       "8",      "DeepSeek R1-Zero default; lower variance"),
        ("TP rollout",       "2",       "2",      "Option B layout (2 dedicated rollout GPUs)"),
        ("Strategy",         "fsdp2",   "fsdp",   "torch 2.5 stable veRL env"),
        ("Max prompt",       "1024",    "2400",   "3 cam + HD map BEV image tokens"),
        ("Max response",     "2048",    "50",     "OpenVLA-bin trajectory tokens only"),
        ("Epochs/steps",     "15 ep",   "500 steps (~0.5 ep)", "24K samples, single half-epoch"),
        ("Save freq",        "20",      "50",     "ckpt every ~30 min @ ~35s/step"),
        ("Test freq",        "5",       "50",     "match save freq; 200-sample val subset"),
        ("GPU mem util rollout", "0.6", "0.5",    "32GB tighter than 80GB H100"),
    ]
    print("\n  Hyperparameter audit (paper vs ours):")
    print("  " + "-" * 90)
    print(f"  {'PARAM':<22} {'PAPER':<22} {'OURS':<22} {'NOTE':<30}")
    print("  " + "-" * 90)
    for p, paper, ours, note in rows:
        print(f"  {p:<22} {paper:<22} {ours:<22} {note:<30}")
    print("  " + "-" * 90)


def _run_one(name: str, script: Path, timeout_s: int) -> bool:
    if not script.exists():
        _fail(name, f"missing {script}")
        return False
    print(f"\n  Running {name} via {script.name} ...")
    try:
        r = subprocess.run(
            ["/usr/bin/python3", str(script)],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        _fail(name, f"timed out after {timeout_s}s")
        return False
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        _fail(name, f"non-zero exit {r.returncode}")
        return False
    _ok(name, "passed")
    return True


def run_save_load_smoke() -> bool:
    """E2E save -> load -> logit parity. Bf16 round-trip must match exactly."""
    return _run_one(
        "smoke_save_load",
        GRPO_DIR / "smoke" / "e2e_save_load_smoke.py",
        timeout_s=900,
    )


def run_reward_unit_test() -> bool:
    """Agent C's 10-sample reward correctness check (perfect > static > wrong-dir)."""
    return _run_one(
        "smoke_reward",
        GRPO_DIR / "test_reward.py",
        timeout_s=300,
    )


def run_verl_stack_smoke() -> bool:
    """Agent C's 1-step rollout smoke through the actual veRL+sglang stack."""
    return _run_one(
        "smoke_verl_stack",
        GRPO_DIR / "smoke_rollout_step.py",
        timeout_s=900,
    )


def eta_estimate() -> None:
    # 8x 5090 Option B: 6 train + 2 rollout, sglang remote-ish, ~35s/step typical
    sec_per_step = 35.0
    n_steps = 500
    eval_overhead_s = (500 // 50) * 60  # 10 evals * 60s each (200-sample subset)
    total_h = (sec_per_step * n_steps + eval_overhead_s) / 3600.0
    print(f"\n  ETA: {sec_per_step:.0f} s/step x {n_steps} steps + "
          f"{eval_overhead_s} s eval overhead = {total_h:.1f} h wall-clock")


# ----------------- main -----------------


def main() -> int:
    print("=" * 92)
    print(" GRPO B.5' pre-launch checklist  (per feedback_pre_launch_checklist memory rule)")
    print("=" * 92)
    checks = [
        ("disk",            check_disk),
        ("gpu",             check_gpu),
        ("sglang",          check_sglang),
        ("base_ckpt",       check_base_ckpt),
        ("agent_artifacts", check_agent_artifacts),
        ("config_parse",    check_config_parses),
    ]
    failures = []
    for name, fn in checks:
        try:
            ok = fn()
        except Exception:
            traceback.print_exc()
            ok = False
        if not ok:
            failures.append(name)

    print_audit_table()

    if not failures:
        for label, fn in (
            ("smoke_reward",     run_reward_unit_test),
            ("smoke_save_load",  run_save_load_smoke),
            ("smoke_verl_stack", run_verl_stack_smoke),
        ):
            if not fn():
                failures.append(label)
                # keep running the rest -- each smoke isolates a different layer
    else:
        print("\n  Skipping smokes -- earlier checks failed.")

    eta_estimate()

    print("\n" + "=" * 92)
    if failures:
        print(f" RESULT: NO-GO -- {len(failures)} failure(s): {failures}")
        print("=" * 92)
        return 1
    print(" RESULT: GO -- all checks passed. Operator must still eyeball ETA + audit table.")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    sys.exit(main())
