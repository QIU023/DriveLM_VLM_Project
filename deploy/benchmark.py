"""TTFT + decode latency harness for the TRT-LLM Qwen2.5-VL VLA engine.

Runs INSIDE the nvcr.io/nvidia/pytorch:25.01-py3 container. Will refuse to run
on the training host because importing tensorrt_llm against torch 2.11+cu130
crashes the cuda driver (host is on driver 580 / CUDA 13.0; TRT-LLM wheels
are still built for CUDA 12.8 in 1.x).

Workload mirrors the on-vehicle VLA pattern: batch=1, real nuScenes val
samples (single CAM_FRONT video clip + HD-map BEV + bbox text + ego speed),
~14 trajectory tokens to decode (1 <traj_start> + 12 bin tokens for 6
waypoints x 2 dims + 1 <traj_end>). TTFT dominates total latency at batch=1;
that's the number the visual-token compression work moves and the one the
automotive 10 Hz budget cares about.

Measurement notes:
- Discard first 3 runs (cold cache / autotuner warmup) per deploy/README §1.
- TTFT = time from generate() call to first decoded token.
- decode latency = (total_time - ttft) / (n_tokens - 1); reported as ms/token.
- p99 is meaningful with --n-runs >= 50.
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple


# --- TRT-LLM import guard ---------------------------------------------------
# Import is wrapped because the script is sometimes invoked on the host by
# mistake (e.g. running directly from the repo root). The host has neither
# tensorrt nor tensorrt_llm — importing them via the cu130 torch wheel
# segfaults the cuda runtime, which would crash any concurrent training job.

_TRTLLM_IMPORT_ERROR: Optional[Exception] = None
try:
    import tensorrt_llm  # type: ignore  # noqa: F401
    from tensorrt_llm import LLM, SamplingParams  # type: ignore
    _HAS_TRTLLM = True
except Exception as e:  # noqa: BLE001 — any import failure means "host"
    _HAS_TRTLLM = False
    _TRTLLM_IMPORT_ERROR = e


def _fatal_host(err: Optional[Exception]) -> None:
    print(
        "[benchmark] FATAL: tensorrt_llm is not importable in this interpreter.\n"
        "  This script must run INSIDE the nvcr.io/nvidia/pytorch:25.01-py3\n"
        "  container — see deploy/README.md §1 for the bring-up command.\n"
        f"  underlying error: {err!r}",
        file=sys.stderr,
    )
    sys.exit(2)


# --- dataset wiring ---------------------------------------------------------
# We reuse the training-time MultiModalPlanningDataset so the benchmark stays
# in lock-step with the actual deployed prompt shape. Importing transformers
# inside the NGC container is fine (container ships transformers 4.x), but the
# trajectory-tokenizer + dataset modules are pure-python and load anywhere.

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))


def _build_val_samples(tokenizer_dir: str, n: int) -> List[dict]:
    """Pull N real nuScenes val samples through MultiModalPlanningDataset.

    Returns a list of dicts with keys {prompt_text, video_clip, hdmap_image,
    input_ids, gt_trajectory_tokens}. Each dict is fed verbatim into the
    TRT-LLM runner so prompt length distribution matches deployment.
    """
    from transformers import AutoProcessor  # type: ignore

    from multimodal_planning_dataset import MultiModalPlanningDataset  # type: ignore

    processor = AutoProcessor.from_pretrained(tokenizer_dir)
    processor.tokenizer.padding_side = "left"

    infos_val = os.environ.get(
        "INFOS_VAL",
        str(_REPO_ROOT / "data/uniad_infos/nuscenes_infos_temporal_val.pkl"),
    )
    nusc_root = os.environ.get("NUSC_ROOT", str(_REPO_ROOT / "data/nuscenes"))
    hdmap_dir = os.environ.get("HDMAP_DIR", str(_REPO_ROOT / "data/preproc/hdmap_bev"))
    bbox_jsonl = os.environ.get(
        "BBOX_JSONL", str(_REPO_ROOT / "data/preproc/bbox_egostate_val.jsonl")
    )

    ds = MultiModalPlanningDataset(
        infos_path=infos_val,
        nusc_root=nusc_root,
        processor=processor,
        max_length=4096,
        num_past_frames=4,
        num_future_waypoints=6,
        video_fps=2.0,
        vla_loss_mode="answer_and_traj",
        max_samples=n,
        require_full_future=True,
        planning_cams=["CAM_FRONT"],
        require_all_cams=True,
        hdmap_dir=hdmap_dir,
        bbox_jsonl=bbox_jsonl,
        split="val",
        modality_dropout_p=0.0,
    )

    out: List[dict] = []
    for i in range(min(n, len(ds))):
        s = ds[i]
        out.append({
            "input_ids": s["input_ids"],
            "pixel_values": s.get("pixel_values"),
            "image_grid_thw": s.get("image_grid_thw"),
            "pixel_values_videos": s.get("pixel_values_videos"),
            "video_grid_thw": s.get("video_grid_thw"),
            "second_per_grid_ts": s.get("second_per_grid_ts"),
            "prompt_len": int(s["_meta_prompt_len"]),
            "gt_action_len": int(s["_meta_action_len"]),
            "token": s["_meta_token"],
        })
    return out


# --- TRT-LLM runner ---------------------------------------------------------
# The TRT-LLM 1.x Python API exposes both a streaming `LLM.generate_async`
# (pytorch backend) and the lower-level Executor. The high-level LLM is
# version-stable for batch-1 inference, but the multimodal input plumbing
# differs across releases:
#
#   * TRT-LLM 1.0-1.2 with --backend trtllm (engine path): use the Qwen2VL
#     example's MultimodalModelRunner — see
#     tensorrt_llm/runtime/multimodal_model_runner.py
#   * TRT-LLM 1.3+ with --backend pytorch: pass `multi_modal_data=` directly
#     to LLM(...).generate_async.
#
# We auto-detect via attribute presence and fall back gracefully. The
# *timing* logic is identical either way — we measure wall-clock at the
# first streamed token and at completion.


def _build_runner(engine_dir: str, vision_engine_dir: Optional[str], tokenizer_dir: str):
    """Return a callable: (sample_dict, max_new_tokens) -> token iterator.

    Each yielded item is a generated token id (int). First yield delimits
    TTFT; total yield count delimits decode latency.
    """
    # Path A: prefer the multimodal runner if the engine_dir contains a
    # vision sub-engine (legacy 1.0-1.2 style).
    vis_path = Path(vision_engine_dir) if vision_engine_dir else None
    try:
        from tensorrt_llm.runtime import MultimodalModelRunner  # type: ignore
        if vis_path and vis_path.exists():
            runner = MultimodalModelRunner(
                visual_engine_dir=str(vis_path),
                llm_engine_dir=engine_dir,
                hf_model_dir=tokenizer_dir,
            )

            def _gen_legacy(sample: dict, max_new_tokens: int):
                # The legacy runner's API is generate_streaming(...) -> iterator.
                yield from runner.generate_streaming(
                    pre_prompt_ids=sample["input_ids"],
                    pixel_values=sample.get("pixel_values_videos"),  # video path is dominant
                    image_grid_thw=sample.get("video_grid_thw"),
                    image=sample.get("pixel_values"),
                    extra_image_grid_thw=sample.get("image_grid_thw"),
                    max_new_tokens=max_new_tokens,
                    temperature=0.0,
                )

            return _gen_legacy
    except ImportError:
        pass

    # Path B: pytorch backend LLM API (1.3+). multi_modal_data is a dict
    # of {modality: tensor_or_list}; exact keys follow HF processor outputs.
    llm = LLM(
        model=engine_dir,
        tokenizer=tokenizer_dir,
        backend="pytorch",
        # batch=1, single ego vehicle — see deploy/README §5.
        max_batch_size=1,
        max_seq_len=4608,
    )

    def _gen_pytorch(sample: dict, max_new_tokens: int):
        sp = SamplingParams(max_tokens=max_new_tokens, temperature=0.0)
        mm: dict = {}
        if sample.get("pixel_values") is not None:
            mm["image"] = {
                "pixel_values": sample["pixel_values"],
                "image_grid_thw": sample.get("image_grid_thw"),
            }
        if sample.get("pixel_values_videos") is not None:
            mm["video"] = {
                "pixel_values": sample["pixel_values_videos"],
                "video_grid_thw": sample.get("video_grid_thw"),
                "second_per_grid_ts": sample.get("second_per_grid_ts"),
            }
        req = llm.generate_async(
            inputs={"prompt_token_ids": sample["input_ids"].tolist(),
                    "multi_modal_data": mm},
            sampling_params=sp,
            streaming=True,
        )
        prev_len = 0
        for out in req:
            ids = out.outputs[0].token_ids
            for tid in ids[prev_len:]:
                yield int(tid)
            prev_len = len(ids)

    return _gen_pytorch


def _time_one(gen_iter, max_new_tokens: int) -> Tuple[float, float, int]:
    """Returns (ttft_s, total_s, n_decoded)."""
    t0 = time.perf_counter()
    ttft: Optional[float] = None
    n = 0
    for _tok in gen_iter:
        now = time.perf_counter()
        if ttft is None:
            ttft = now - t0
        n += 1
        if n >= max_new_tokens:
            break
    total = time.perf_counter() - t0
    return ttft or 0.0, total, n


def _pct(xs: List[float], q: float) -> float:
    # Manual percentile to avoid numpy dep in the container path.
    if not xs:
        return 0.0
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round((q / 100.0) * (len(s) - 1)))))
    return s[k]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--engine-dir", required=True,
                    help="LM TRT engine dir (output of trtllm-build)")
    ap.add_argument("--vision-engine-dir", default=None,
                    help="vision sub-engine dir (legacy 1.0-1.2 runner). "
                         "Omit on 1.3+ pytorch backend.")
    ap.add_argument("--tokenizer-dir", required=True,
                    help="HF tokenizer dir — pass the original HF ckpt root")
    ap.add_argument("--n-runs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=20,
                    help="14 is the planning trajectory; 20 leaves margin")
    ap.add_argument("--n-warmup", type=int, default=3,
                    help="discarded — TRT autotuner + KV warmup")
    ap.add_argument("--csv-out", default=None,
                    help="write per-run rows for analysis")
    ap.add_argument("--n-samples", type=int, default=8,
                    help="distinct val samples to cycle through across n-runs")
    args = ap.parse_args()

    if not _HAS_TRTLLM:
        _fatal_host(_TRTLLM_IMPORT_ERROR)

    if args.batch_size != 1:
        # The automotive workload is batch=1 by construction (single ego, one
        # frame at a time at 10 Hz). Higher batch can be benched, but the
        # latency story we report is meaningless then; warn loudly.
        print(f"[benchmark] WARNING: batch_size={args.batch_size} != 1. "
              "TTFT / decode numbers will not reflect the deployment regime.",
              file=sys.stderr)

    print(f"[benchmark] loading {args.n_samples} val samples ...", flush=True)
    samples = _build_val_samples(args.tokenizer_dir, args.n_samples)
    print(f"[benchmark] got {len(samples)} samples; building runner ...", flush=True)

    runner = _build_runner(args.engine_dir, args.vision_engine_dir, args.tokenizer_dir)

    # Warmup
    for w in range(args.n_warmup):
        s = samples[w % len(samples)]
        _time_one(runner(s, args.max_new_tokens), args.max_new_tokens)
        print(f"[benchmark] warmup {w + 1}/{args.n_warmup} done", flush=True)

    rows: List[dict] = []
    for r in range(args.n_runs):
        s = samples[r % len(samples)]
        ttft, total, n = _time_one(runner(s, args.max_new_tokens), args.max_new_tokens)
        decode_per_tok = (total - ttft) / max(1, n - 1) if n > 1 else 0.0
        rows.append({
            "run": r,
            "sample_token": s["token"],
            "prompt_len": s["prompt_len"],
            "n_decoded": n,
            "ttft_ms": 1e3 * ttft,
            "decode_ms_per_tok": 1e3 * decode_per_tok,
            "total_ms": 1e3 * total,
        })
        if (r + 1) % 10 == 0:
            print(f"[benchmark] run {r + 1}/{args.n_runs} ttft={1e3 * ttft:.1f}ms "
                  f"decode={1e3 * decode_per_tok:.2f}ms/tok total={1e3 * total:.1f}ms",
                  flush=True)

    ttfts = [r["ttft_ms"] for r in rows]
    decodes = [r["decode_ms_per_tok"] for r in rows]
    totals = [r["total_ms"] for r in rows]
    tput = [1e3 / d if d > 0 else 0.0 for d in decodes]

    print("\n=== TRT-LLM Qwen2.5-VL VLA benchmark ===")
    print(f"engine_dir:    {args.engine_dir}")
    print(f"runs:          {args.n_runs}  (after {args.n_warmup} warmup)")
    print(f"batch_size:    {args.batch_size}")
    print(f"max_new_tok:   {args.max_new_tokens}")
    print(f"samples cycled: {len(samples)}")
    print()
    print(f"{'metric':<28} {'mean':>10} {'p50':>10} {'p99':>10}")
    print(f"{'-' * 60}")
    print(f"{'TTFT (ms)':<28} {statistics.mean(ttfts):>10.2f} "
          f"{_pct(ttfts, 50):>10.2f} {_pct(ttfts, 99):>10.2f}")
    print(f"{'decode (ms/tok)':<28} {statistics.mean(decodes):>10.3f} "
          f"{_pct(decodes, 50):>10.3f} {_pct(decodes, 99):>10.3f}")
    print(f"{'decode throughput (tok/s)':<28} {statistics.mean(tput):>10.1f} "
          f"{_pct(tput, 50):>10.1f} {_pct(tput, 99):>10.1f}")
    print(f"{'total request (ms)':<28} {statistics.mean(totals):>10.2f} "
          f"{_pct(totals, 50):>10.2f} {_pct(totals, 99):>10.2f}")
    print()
    # Automotive gate. See deploy/README §5: 10 Hz budget => 100 ms total per
    # planning cycle; TTFT is the dominant component at batch=1.
    p99_ttft = _pct(ttfts, 99)
    if p99_ttft < 100.0:
        print(f"[PASS] p99 TTFT {p99_ttft:.1f} ms < 100 ms automotive gate")
    else:
        print(f"[WARN] p99 TTFT {p99_ttft:.1f} ms >= 100 ms — over automotive 10 Hz budget")

    if args.csv_out:
        out_path = Path(args.csv_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[benchmark] per-run CSV -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
