#!/usr/bin/env python3
"""TRT-LLM conversion stub for Qwen2.5-VL-3B VLA.

Runs INSIDE the nvcr.io/nvidia/pytorch:25.01-py3 container (NOT on host —
host has CUDA 13.0 + torch 2.11 which conflicts with the TRT-LLM stack).

Wraps the version-specific TRT-LLM `convert_checkpoint.py` for Qwen2-VL,
the FP4 quantization step (via nvidia-modelopt), and `trtllm-build`.

Standard flow (target: RTX 5090, sm_120, NVFP4):

  1. Host: merge LoRA → plain HF dir   (only if LoRA-only ckpt)
       /usr/bin/python3 scripts/merge_lora_to_base.py \\
         --base /workspace/models/Qwen2.5-VL-3B-Instruct \\
         --lora checkpoints_qwen25/<run>/final \\
         --out  /tmp/merged_hf
  2. Container: this script does
       (a) quantize FP4 ckpt   → /models/ckpt_nvfp4/
       (b) trtllm-build engine → /models/engine_5090/
       (c) build vision engine → /models/engine_5090/vision/

Expected runtime on 5090 for 3B:
       quantize  ~5-10 min  (calib_size=512 nuScenes samples)
       LM build  ~3-5 min   (single GPU, MAX_BS=1, MAX_SEQ_LEN=4608)
       ViT build ~1-2 min   (separate engine — TRT-LLM Qwen2VL example pattern)
       parity vs HF: ~1-2 h (manual; numerical tolerance 1e-3 for FP4)

USAGE (sketch — flags may move across TRT-LLM versions, see deploy/README §1):
    python deploy/trt_convert_qwen25vl.py \\
        --hf-dir /models/qwen25vl-vla-merged \\
        --out    /models/engine_5090 \\
        --calib-jsonl data/preproc/bbox_egostate_val.jsonl \\
        --calib-n 512 \\
        --max-bs 1 --max-input-len 4096 --max-seq-len 4608
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], dry: bool) -> None:
    print("$", " ".join(cmd), flush=True)
    if dry:
        return
    rc = subprocess.run(cmd, check=False).returncode
    if rc != 0:
        print(f"[trt_convert] FAILED (rc={rc}): {cmd[0]}", file=sys.stderr)
        sys.exit(rc)


def _verify_container() -> None:
    try:
        import tensorrt_llm  # type: ignore
        import tensorrt  # type: ignore
        print(f"[trt_convert] tensorrt={tensorrt.__version__} "
              f"tensorrt_llm={tensorrt_llm.__version__}")
    except ImportError as e:
        print(
            "[trt_convert] FATAL: TRT-LLM stack missing — run this script "
            "INSIDE the NGC pytorch:25.01-py3 container, not on the host:\n"
            "  docker run --rm -it --gpus '\"device=0\"' --ipc=host \\\n"
            "    -v /workspace/DriveLM_VLM_Project:/work -v /workspace/models:/models \\\n"
            "    nvcr.io/nvidia/pytorch:25.01-py3 bash\n"
            "  pip install tensorrt_llm\n"
            f"  err: {e}",
            file=sys.stderr,
        )
        sys.exit(2)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--hf-dir", required=True,
                   help="LoRA-merged HF dir (Qwen2.5-VL-3B with VLA head)")
    p.add_argument("--out", required=True,
                   help="output root for ckpt_nvfp4/ + engine_5090/")
    p.add_argument("--calib-jsonl",
                   default="data/preproc/bbox_egostate_val.jsonl",
                   help="nuScenes calibration set (uses real-data activations)")
    p.add_argument("--calib-n", type=int, default=512)
    p.add_argument("--max-bs", type=int, default=1,
                   help="batch=1 mirrors single-ego automotive inference")
    p.add_argument("--max-input-len", type=int, default=4096)
    p.add_argument("--max-seq-len", type=int, default=4608)
    p.add_argument("--skip-quant", action="store_true",
                   help="reuse existing /ckpt_nvfp4 (e.g. after engine-rebuild)")
    p.add_argument("--dry-run", action="store_true",
                   help="print commands without executing")
    args = p.parse_args()

    if not args.dry_run:
        _verify_container()

    out = Path(args.out)
    ckpt_dir = out / "ckpt_nvfp4"
    engine_dir = out / "engine_5090"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    engine_dir.mkdir(parents=True, exist_ok=True)

    # --- (a) FP4 quantize (skip if reusing) -------------------------------------
    if not args.skip_quant:
        # Path moves across versions. The current script defers to the
        # quantize_fp4.sh wrapper which is hand-maintained when versions change.
        env = os.environ.copy()
        env.update({"MODEL_DIR": args.hf_dir, "OUT_DIR": str(ckpt_dir),
                    "CALIB_SIZE": str(args.calib_n), "DTYPE": "bfloat16"})
        cmd = ["bash", str(Path(__file__).parent / "quantize_fp4.sh")]
        print(f"[trt_convert] quantize → {ckpt_dir}")
        print("$ MODEL_DIR={} OUT_DIR={} CALIB_SIZE={} bash quantize_fp4.sh".format(
            args.hf_dir, ckpt_dir, args.calib_n))
        if not args.dry_run:
            subprocess.run(cmd, env=env, check=False)

    # --- (b) LM engine build ----------------------------------------------------
    env = os.environ.copy()
    env.update({
        "CKPT_DIR": str(ckpt_dir),
        "ENGINE_DIR": str(engine_dir),
        "MAX_BS": str(args.max_bs),
        "MAX_INPUT_LEN": str(args.max_input_len),
        "MAX_SEQ_LEN": str(args.max_seq_len),
    })
    print(f"[trt_convert] build LM engine → {engine_dir}")
    if not args.dry_run:
        subprocess.run(
            ["bash", str(Path(__file__).parent / "build_engine.sh")],
            env=env, check=False,
        )

    # --- (c) Vision engine build ------------------------------------------------
    # The vision-tower engine is a separate artifact in TRT-LLM Qwen2VL example.
    # Build script name varies by version (build_visual_engine.py vs
    # build_multimodal_engine.py). We do NOT shell it out automatically here;
    # see deploy/README §1 for the version-locked invocation.
    print(f"[trt_convert] NEXT (manual): build vision engine into {engine_dir}/vision")
    print("  → cd $TRT_LLM_EXAMPLES/qwen2vl && "
          "python build_visual_engine.py --model_path {} --output_dir {}".format(
              args.hf_dir, engine_dir / "vision"))

    print(f"[trt_convert] DONE. Engines: {engine_dir}")
    print("  → run latency benchmark: python deploy/benchmark.py --engine-dir " + str(engine_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
