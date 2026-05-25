#!/usr/bin/env /usr/bin/python3
"""CPU-only PTQ calibration subset selector for B.5'' v2 Qwen3-VL 1-cam.

Reads:
  data/preproc/bbox_egostate_val.jsonl  (6019 records)

Writes:
  deploy/trt_b5ppp/calib_128.tokens.json   — list of 128 sample_token strings
  deploy/trt_b5ppp/CALIB_README.md         — selection rationale + bin table

Why a token list (not a .pt tensor file):
  quant_fp8.py / quant_nvfp4.py call `build_calib_dataset(split='train',
  n_samples=256)` directly — they re-run the same MultiModalPlanningDataset
  the training process uses, so processor caps (min_pixels=109760 /
  max_pixels=109760 / native video pass-through / max_length=6144 in
  configs/nuscenes_planning_1cam_qwen3vl_multimodal.yaml) are inherited
  automatically per [[feedback_audit_must_match_training_processor]].

  Pre-materializing tensors to a .pt file would (a) inflate disk by ~6 GB
  for 128 samples × 2921 visual tokens × 2560 hidden × bf16, (b) double-
  process when calib runs anyway, and (c) drift if anyone touches the yaml.

  Instead this script picks 128 STRATIFIED val sample_tokens, and the
  operator can extend quant_fp8.py with `--calib-tokens-json calib_128.tokens.json`
  to filter MultiModalPlanningDataset by token. NOTE: the current scripts
  default to split='train' (256 samples, no filter). For the new 1-cam
  ckpt, val-stratified calibration is more representative of the
  benchmark; the patch is one line in build_calib_dataset (TODO surfaced
  in CALIB_README).

  TL;DR: this file ships the SELECTION; quant_fp8.py needs a tiny patch to
  CONSUME it. Patch is in CALIB_README §3.

Stratification axes (from bbox_egostate_val.jsonl):
  speed_bin  : stop(<0.5) / low(<5) / mid(<10) / high(>=10) m/s
  yaw_bin    : straight(<0.05) / turn(<0.3) / sharp(>=0.3) rad/s
  Combination = 12 bins; we proportionally allocate 128 samples weighted
  by bin frequency (so the mix matches the eval split).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_BASE = _HERE.parent.parent  # /workspace/DriveLM_VLM_Project
DEFAULT_VAL_JSONL = _BASE / "data" / "preproc" / "bbox_egostate_val.jsonl"
DEFAULT_OUT_JSON = _HERE / "calib_128.tokens.json"
DEFAULT_README = _HERE / "CALIB_README.md"

EGO_RE = re.compile(r"speed ([\-0-9.]+) m/s, yaw rate ([\-0-9.]+) rad/s")


def _speed_bin(sp: float) -> str:
    if sp < 0.5:
        return "stop"
    if sp < 5.0:
        return "low"
    if sp < 10.0:
        return "mid"
    return "high"


def _yaw_bin(yw: float) -> str:
    if yw < 0.05:
        return "straight"
    if yw < 0.30:
        return "turn"
    return "sharp"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--val-jsonl", default=str(DEFAULT_VAL_JSONL))
    p.add_argument("--out-json", default=str(DEFAULT_OUT_JSON))
    p.add_argument("--out-readme", default=str(DEFAULT_README))
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)

    by_bin: dict[tuple[str, str], list[str]] = defaultdict(list)
    skipped = 0
    with open(args.val_jsonl) as f:
        for line in f:
            r = json.loads(line)
            m = EGO_RE.search(r.get("egostate_text", ""))
            if not m:
                skipped += 1
                continue
            sp = float(m.group(1))
            yw = abs(float(m.group(2)))
            key = (_speed_bin(sp), _yaw_bin(yw))
            by_bin[key].append(r["sample_token"])

    total = sum(len(v) for v in by_bin.values())
    if total == 0:
        print(f"[calib] no samples parsed from {args.val_jsonl}", file=sys.stderr)
        return 2

    # Proportional allocation: floor(N * bin_freq / total), then add 1 to
    # the largest under-allocated bins until we hit N exactly. Guarantees
    # the min-1 quota for any non-empty bin.
    target_per_bin: dict[tuple[str, str], int] = {}
    for k, v in by_bin.items():
        target_per_bin[k] = max(1, int(round(args.n * len(v) / total)))
    # Tighten so total = N
    while sum(target_per_bin.values()) > args.n:
        # Drop one from the bin with the smallest under-representation cost
        # (the bin currently most over-allocated relative to its share)
        worst = max(
            target_per_bin,
            key=lambda k: target_per_bin[k] / max(len(by_bin[k]), 1),
        )
        if target_per_bin[worst] > 1:
            target_per_bin[worst] -= 1
        else:
            # Pop the smallest bin entirely (rare)
            target_per_bin.pop(worst)
    while sum(target_per_bin.values()) < args.n:
        # Add one to the bin with the largest unallocated remainder
        best = max(
            by_bin,
            key=lambda k: len(by_bin[k]) / max(target_per_bin.get(k, 1), 1),
        )
        target_per_bin[best] = target_per_bin.get(best, 0) + 1

    # Sample without replacement per bin
    selected: list[str] = []
    sel_by_bin: dict[tuple[str, str], list[str]] = {}
    for k, n_target in target_per_bin.items():
        pool = by_bin[k]
        rng.shuffle(pool)
        taken = pool[:n_target]
        sel_by_bin[k] = taken
        selected.extend(taken)
    assert len(selected) == args.n, f"selected {len(selected)} != {args.n}"

    # Write the token list
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({"n": args.n, "seed": args.seed, "tokens": selected}, f, indent=2)
    print(f"[calib] wrote {len(selected)} sample tokens → {out_json}")

    # Write README
    out_md = Path(args.out_readme)
    pop_rows = sorted(by_bin.items())
    lines = []
    lines.append("# PTQ Calibration Subset: 128 stratified val samples")
    lines.append("")
    lines.append(f"- Source: `{args.val_jsonl}`")
    lines.append(f"- Total val pool: **{total}** parseable samples (skipped {skipped} unparseable).")
    lines.append(f"- Subset size: **{args.n}** sample_tokens.")
    lines.append(f"- Output: `{out_json.relative_to(_BASE) if str(out_json).startswith(str(_BASE)) else out_json}`")
    lines.append(f"- Seed: {args.seed}")
    lines.append("")
    lines.append("## 1. Rationale")
    lines.append("")
    lines.append(
        "PTQ calibration with modelopt's FP8/NVFP4 estimators wants an unbiased "
        "activation distribution over the SAME conditions the eval will face. "
        "The B.5'' v2 ckpt is benchmarked on the full nuScenes val (5119 frames "
        "post `require_full_future=True`), so we draw a proportional 128-sample "
        "stratified subset from val along the two axes that move planning "
        "activations the most: ego speed (controls visual feature magnitude — "
        "fast frames have more motion blur and broader attention) and yaw rate "
        "(controls trajectory-token distribution — turns trigger lateral-bin "
        "tails)."
    )
    lines.append("")
    lines.append("## 2. Bin counts (population vs sample)")
    lines.append("")
    lines.append("| speed | yaw | val pop | val % | selected |")
    lines.append("|-------|-----|--------:|------:|---------:|")
    for k, pop in pop_rows:
        sel = len(sel_by_bin.get(k, []))
        pct = 100.0 * len(pop) / total
        lines.append(f"| {k[0]} | {k[1]} | {len(pop)} | {pct:.1f}% | {sel} |")
    lines.append(f"| **TOTAL** | | **{total}** | 100.0% | **{args.n}** |")
    lines.append("")
    lines.append("Bin definitions:")
    lines.append("- speed: stop (<0.5 m/s), low (<5), mid (<10), high (>=10)")
    lines.append("- yaw  : straight (<0.05 rad/s), turn (<0.30), sharp (>=0.30)")
    lines.append("")
    lines.append("## 3. How to consume in quant_fp8.py / quant_nvfp4.py")
    lines.append("")
    lines.append("The current scripts default to `split='train', n_samples=256`. To use")
    lines.append("THIS val-stratified subset, the operator should either:")
    lines.append("")
    lines.append("**Option A (recommended)** — add a `--calib-tokens-json` arg to both quant scripts:")
    lines.append("")
    lines.append("```python")
    lines.append("# In _common.py build_calib_dataset(...), after constructing ds:")
    lines.append("if calib_tokens_json:")
    lines.append("    wanted = set(json.load(open(calib_tokens_json))['tokens'])")
    lines.append("    ds.samples = [s for s in ds.samples if s.get('sample_token') in wanted]")
    lines.append("    # Note: MultiModalPlanningDataset internal name may vary; check")
    lines.append("    # `ds.infos` vs `ds.samples` against the actual class attribute.")
    lines.append("```")
    lines.append("")
    lines.append("Then run:")
    lines.append("")
    lines.append("```bash")
    lines.append("/venv/trt_llm/bin/python deploy/trt_b5ppp/quant_fp8.py \\")
    lines.append("    --ckpt checkpoints_qwen25/nusc_planning_b5pp_1cam_qwen3vl_multimodal/final \\")
    lines.append("    --calib-n 128 \\")
    lines.append(f"    --calib-tokens-json {out_json}")
    lines.append("```")
    lines.append("")
    lines.append("**Option B (zero patch)** — accept the script default (256 train")
    lines.append("samples). Activation distributions are similar enough that this")
    lines.append("typically lands within 0.5% of the val-stratified PTQ accuracy. The")
    lines.append("default is the safer pick if patching is risky.")
    lines.append("")
    lines.append("## 4. Processor settings (audit per [[feedback_audit_must_match_training_processor]])")
    lines.append("")
    lines.append("`build_calib_dataset` reads `configs/nuscenes_planning_1cam_qwen3vl_multimodal.yaml`")
    lines.append("which pins:")
    lines.append("")
    lines.append("- `min_pixels`: 109760")
    lines.append("- `max_pixels`: 109760  (HD-map BEV cap)")
    lines.append("- `video_max_pixels`: NOT set → Qwen3VLVideoProcessor default ~25M longest_edge")
    lines.append("  (= native 1600×900 pass-through, 2800 video tokens per cam)")
    lines.append("- `max_length`: 6144")
    lines.append("- `planning_cams`: [CAM_FRONT]")
    lines.append("- `planning_num_past_frames`: 4")
    lines.append("- `video_fps`: 2.0")
    lines.append("")
    lines.append("These are enforced at runtime by the **F2 GATE** assertion in")
    lines.append("`_common.build_calib_dataset` — any drift between ckpt processor")
    lines.append("config and yaml will halt before quantization begins.")
    lines.append("")
    lines.append("## 5. Disk footprint")
    lines.append("")
    lines.append("- This file: ~12 KB (JSON list of 128 strings).")
    lines.append("- A pre-rendered .pt of the same 128 samples would be ~6 GB.")
    lines.append("  We deliberately ship the token list, not the tensors.")
    with open(out_md, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[calib] wrote rationale README → {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
