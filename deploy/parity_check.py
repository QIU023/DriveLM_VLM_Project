"""HF bf16 vs TRT-LLM FP4 parity check on 5 nuScenes val samples.

The deployment story for the application demo hinges on FP4 not silently
shifting the trajectory output relative to the bf16 ckpt that the eval
numbers were measured on. This script runs the same 5 val samples through
both engines and reports:

  * token-level exact-match rate (the trajectory token stream is short — 14
    tokens — so a single bin flip is visible)
  * position of first divergence per non-matching sample
  * per-sample decoded-trajectory L2 vs GT (so we also see whether FP4
    matters in *trajectory-space* units, not just token-space)
  * logit KL on the first 3 trajectory positions, if logits are accessible
    from both runners (HF: trivial; TRT-LLM: only on some 1.x runners — we
    fall back to "n/a" if not)

Three run modes — they exist because the host and the container have
incompatible CUDA stacks (host=cu130/torch2.11; container=cu128/TRT-LLM):

  parity_check.py --hf-only         # host (or container) — HF reference only
  parity_check.py --trt-only        # container only — TRT engine
  parity_check.py                   # both, in one process (rare; user
                                    # combines the two JSON outputs by hand)

Each mode writes a JSON file with the per-sample raw outputs so the user
can combine the two by hand:
  --hf-only  -> <out>/parity_hf.json
  --trt-only -> <out>/parity_trt.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))


# --- shared dataset loader --------------------------------------------------
# We re-pull the same 5 samples on both runs by fixing the dataset's
# max_samples to 5 and not shuffling. That gives byte-identical processor
# outputs on both sides, which is the precondition for token-level diff to
# be meaningful (any drift in tokenization would show up as 100 % mismatch).

def _load_samples(tokenizer_dir: str, n: int):
    from transformers import AutoProcessor  # type: ignore
    from multimodal_planning_dataset import MultiModalPlanningDataset  # type: ignore

    processor = AutoProcessor.from_pretrained(tokenizer_dir)
    processor.tokenizer.padding_side = "left"

    ds = MultiModalPlanningDataset(
        infos_path=os.environ.get(
            "INFOS_VAL",
            str(_REPO_ROOT / "data/uniad_infos/nuscenes_infos_temporal_val.pkl"),
        ),
        nusc_root=os.environ.get("NUSC_ROOT", str(_REPO_ROOT / "data/nuscenes")),
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
        hdmap_dir=os.environ.get("HDMAP_DIR", str(_REPO_ROOT / "data/preproc/hdmap_bev")),
        bbox_jsonl=os.environ.get(
            "BBOX_JSONL", str(_REPO_ROOT / "data/preproc/bbox_egostate_val.jsonl")
        ),
        split="val",
        modality_dropout_p=0.0,
    )
    return processor, [ds[i] for i in range(min(n, len(ds)))]


def _decode_traj(ids: List[int]) -> np.ndarray:
    """Decode token-id list -> (T, 2) waypoints via the project tokenizer."""
    from trajectory_tokenizer import TrajectoryTokenizer  # type: ignore
    return TrajectoryTokenizer().decode(ids)


def _trim_to_traj_block(ids: List[int]) -> List[int]:
    """Strip leading "Predicted trajectory:" tokens; keep <traj_start>..<traj_end>."""
    from trajectory_tokenizer import TRAJ_START_ID, TRAJ_END_ID  # type: ignore
    try:
        s = ids.index(TRAJ_START_ID)
    except ValueError:
        return ids
    try:
        e = ids.index(TRAJ_END_ID, s) + 1
    except ValueError:
        e = len(ids)
    return ids[s:e]


# --- HF path (training host) ------------------------------------------------
# transformers 5.6.0 deprecated `torch_dtype` in favor of `dtype`. We use the
# new name (per ENV_FREEZE.md). If running on a container with an older
# transformers (4.x in NGC pytorch:25.01-py3), we fall back to torch_dtype.

def run_hf(hf_dir: str, tokenizer_dir: str, n: int) -> List[Dict[str, Any]]:
    import torch  # type: ignore
    from transformers import AutoModelForImageTextToText  # type: ignore

    if not torch.cuda.is_available():
        # We DON'T want this script ever silently CPU-running 3B bf16 — that
        # would take ~30 min/sample. Bail loudly.
        raise SystemExit("[parity] FATAL: CUDA not available; refusing to run "
                         "HF bf16 on CPU (would take ~30 min/sample).")

    processor, samples = _load_samples(tokenizer_dir, n)
    kw = {"attn_implementation": "sdpa"}
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            hf_dir, dtype=torch.bfloat16, **kw,
        ).cuda()
    except TypeError:
        model = AutoModelForImageTextToText.from_pretrained(  # transformers <5.0
            hf_dir, torch_dtype=torch.bfloat16, **kw,
        ).cuda()
    model.eval()

    out: List[Dict[str, Any]] = []
    for s in samples:
        prompt_len = int(s["_meta_prompt_len"])
        # Re-build the inputs dict for model.generate (the dataset already
        # baked the action tokens INTO input_ids — we slice them off so the
        # generate call must produce them).
        input_ids = s["input_ids"][:prompt_len].unsqueeze(0).cuda()
        attn = (input_ids != processor.tokenizer.pad_token_id).long()
        gen_kwargs: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attn,
            "max_new_tokens": 20,
            "do_sample": False,
            "num_beams": 1,
            "use_cache": True,
            "pad_token_id": processor.tokenizer.pad_token_id or 0,
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        for k in ("pixel_values", "image_grid_thw",
                  "pixel_values_videos", "video_grid_thw",
                  "second_per_grid_ts"):
            v = s.get(k)
            if v is not None:
                gen_kwargs[k] = v.unsqueeze(0).cuda() if v.dim() == 2 else v.cuda()

        with __import__("torch").inference_mode():
            res = model.generate(**gen_kwargs)
        gen_ids = res.sequences[0, prompt_len:].tolist()
        # First-3 logits, softmaxed to a dict the TRT side can compare against.
        first3 = []
        for t in range(min(3, len(res.scores))):
            logits = res.scores[t][0].float().cpu().numpy()
            # Keep top-20 ids+probs to keep JSON small.
            top = np.argsort(-logits)[:20]
            first3.append({
                "top_ids": top.tolist(),
                "top_logits": logits[top].tolist(),
            })

        trim = _trim_to_traj_block(gen_ids)
        wp_pred = _decode_traj(trim)
        wp_gt = s["_meta_waypoints"].cpu().numpy()
        valid = s["_meta_valid_mask"].cpu().numpy()
        if wp_pred.shape == wp_gt.shape:
            err = (wp_pred - wp_gt) * valid[:, None]
            l2 = float(np.sqrt((err ** 2).sum(axis=1)).mean())
        else:
            l2 = float("nan")

        out.append({
            "token": s["_meta_token"],
            "gen_ids": gen_ids,
            "traj_block_ids": trim,
            "first3_topk": first3,
            "wp_pred": wp_pred.tolist(),
            "wp_gt": wp_gt.tolist(),
            "valid": valid.tolist(),
            "l2_to_gt": l2,
        })
    return out


# --- TRT path (NGC container) -----------------------------------------------

def run_trt(engine_dir: str, vision_engine_dir: Optional[str],
            tokenizer_dir: str, n: int) -> List[Dict[str, Any]]:
    try:
        import tensorrt_llm  # type: ignore  # noqa: F401
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            "[parity] FATAL: tensorrt_llm not importable. The --trt path runs "
            "INSIDE the NGC pytorch:25.01-py3 container (deploy/README.md §1).\n"
            f"  err: {e!r}"
        )

    # Import the runner builder from benchmark.py to keep the wiring in one
    # place — we'd otherwise duplicate the multimodal_data plumbing here.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from benchmark import _build_runner  # type: ignore

    _, samples = _load_samples(tokenizer_dir, n)
    runner = _build_runner(engine_dir, vision_engine_dir, tokenizer_dir)

    out: List[Dict[str, Any]] = []
    for s in samples:
        gen_ids: List[int] = []
        for tid in runner(s, 20):
            gen_ids.append(int(tid))
        trim = _trim_to_traj_block(gen_ids)
        wp_pred = _decode_traj(trim)
        wp_gt = s["_meta_waypoints"].cpu().numpy()
        valid = s["_meta_valid_mask"].cpu().numpy()
        if wp_pred.shape == wp_gt.shape:
            err = (wp_pred - wp_gt) * valid[:, None]
            l2 = float(np.sqrt((err ** 2).sum(axis=1)).mean())
        else:
            l2 = float("nan")

        out.append({
            "token": s["_meta_token"],
            "gen_ids": gen_ids,
            "traj_block_ids": trim,
            # TRT-LLM streaming runner doesn't expose per-step logits in 1.x
            # generic API; we leave the slot in the schema for parity with the
            # HF side but mark it n/a.
            "first3_topk": None,
            "wp_pred": wp_pred.tolist(),
            "wp_gt": wp_gt.tolist(),
            "valid": valid.tolist(),
            "l2_to_gt": l2,
        })
    return out


# --- comparison helpers (used when both sides ran in the same process) ------

def _compare(hf: List[Dict[str, Any]], trt: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    exact = 0
    for h, t in zip(hf, trt):
        a = h["traj_block_ids"]
        b = t["traj_block_ids"]
        is_exact = a == b
        first_div: Optional[int] = None
        if not is_exact:
            for i, (x, y) in enumerate(zip(a, b)):
                if x != y:
                    first_div = i
                    break
            else:
                first_div = min(len(a), len(b))
        kl = []
        if h.get("first3_topk"):
            for tk in h["first3_topk"]:
                logits = np.array(tk["top_logits"], dtype=np.float64)
                # softmax over the kept top-20 (approx; sufficient for trend)
                p = np.exp(logits - logits.max())
                p /= p.sum()
                # We don't have TRT logits, so leave a placeholder; computing
                # KL against a one-hot from the TRT chosen token would always
                # be infinite/zero — use surprisal of TRT's pick under HF as a
                # proxy.
                trt_pick = t["traj_block_ids"][len(kl)] if len(kl) < len(t["traj_block_ids"]) else -1
                if trt_pick in tk["top_ids"]:
                    j = tk["top_ids"].index(trt_pick)
                    kl.append({"trt_pick_id": int(trt_pick),
                               "hf_logp": float(np.log(max(p[j], 1e-12)))})
                else:
                    kl.append({"trt_pick_id": int(trt_pick), "hf_logp": None})

        if is_exact:
            exact += 1
        rows.append({
            "token": h["token"],
            "exact_match": is_exact,
            "first_div_pos": first_div,
            "hf_l2": h["l2_to_gt"],
            "trt_l2": t["l2_to_gt"],
            "delta_l2": (t["l2_to_gt"] - h["l2_to_gt"])
                        if not (np.isnan(h["l2_to_gt"]) or np.isnan(t["l2_to_gt"])) else None,
            "first3_hf_logp_at_trt_pick": kl or None,
        })
    return {
        "n": len(rows),
        "exact_match_rate": exact / max(1, len(rows)),
        "per_sample": rows,
    }


# --- main -------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--hf-dir", default=None,
                    help="HF checkpoint dir (required for --hf-only/both)")
    ap.add_argument("--engine-dir", default=None,
                    help="LM TRT engine dir (required for --trt-only/both)")
    ap.add_argument("--vision-engine-dir", default=None,
                    help="vision sub-engine dir (TRT-LLM 1.0-1.2 legacy runner)")
    ap.add_argument("--tokenizer-dir", required=True,
                    help="HF tokenizer dir (typically same as --hf-dir)")
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--out-dir", default=None,
                    help="dir to dump parity_hf.json / parity_trt.json / parity_combined.json")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--hf-only", action="store_true",
                   help="HF bf16 reference only (training host)")
    g.add_argument("--trt-only", action="store_true",
                   help="TRT engine only (NGC container)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else _REPO_ROOT / "deploy" / "parity_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    do_hf = args.hf_only or not args.trt_only
    do_trt = args.trt_only or not args.hf_only

    hf_res: Optional[list] = None
    trt_res: Optional[list] = None

    if do_hf:
        if not args.hf_dir:
            print("[parity] --hf-dir required for HF path", file=sys.stderr)
            return 64
        print(f"[parity] HF bf16 forward on {args.n_samples} samples ...")
        hf_res = run_hf(args.hf_dir, args.tokenizer_dir, args.n_samples)
        (out_dir / "parity_hf.json").write_text(json.dumps(hf_res, indent=2))
        print(f"[parity] wrote {out_dir / 'parity_hf.json'}")

    if do_trt:
        if not args.engine_dir:
            print("[parity] --engine-dir required for TRT path", file=sys.stderr)
            return 64
        print(f"[parity] TRT-LLM FP4 forward on {args.n_samples} samples ...")
        trt_res = run_trt(args.engine_dir, args.vision_engine_dir,
                          args.tokenizer_dir, args.n_samples)
        (out_dir / "parity_trt.json").write_text(json.dumps(trt_res, indent=2))
        print(f"[parity] wrote {out_dir / 'parity_trt.json'}")

    if hf_res is not None and trt_res is not None:
        combined = _compare(hf_res, trt_res)
        (out_dir / "parity_combined.json").write_text(json.dumps(combined, indent=2))
        print()
        print(f"[parity] exact-match rate: {combined['exact_match_rate']:.2%}  "
              f"(n={combined['n']})")
        for r in combined["per_sample"]:
            tag = "OK " if r["exact_match"] else f"DIFF@{r['first_div_pos']}"
            dl = "    " if r["delta_l2"] is None else f"{r['delta_l2']:+.4f}"
            print(f"  {tag}  token={r['token']}  delta_l2={dl}m")

    return 0


if __name__ == "__main__":
    sys.exit(main())
