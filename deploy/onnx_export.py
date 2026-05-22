"""EXPERIMENTAL: ONNX export of the Qwen2.5-VL LM submodule.

Backup deployment path for environments where TRT-LLM is not available but
ONNX Runtime + TensorRT (the *non*-LLM runtime — i.e. pure TRT) is. The
vision tower stays in PyTorch (HF Qwen2.5-VL processor + native ViT); only
the language model is exported. The TRT engine built from this ONNX will
NOT have paged KV cache, in-flight batching, or the gpt_attention plugin —
it's strictly a "can we serve this at all" backup for the demo.

Not tested with FP4 — sm_120 FP4 via stock TRT (vs TRT-LLM) is even more
sparsely supported than the TRT-LLM path. INT8 calibration via TRT calibrator
is the realistic accuracy/perf compromise here.

Usage:
    python deploy/onnx_export.py \
        --hf-dir /workspace/models/Qwen2.5-VL-3B-Instruct \
        --out    /models/qwen25vl_lm.onnx \
        --opset  17 \
        --seq-len 4096

Refuses to run if CUDA isn't available (the export traces the model in fp16
on GPU; CPU export of 3B would take >1 h and exceeds host RAM).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--seq-len", type=int, default=4096,
                    help="static seq dim for the traced graph; pick the max "
                         "input you actually serve")
    ap.add_argument("--dtype", default="float16", choices=("float16", "bfloat16"))
    args = ap.parse_args()

    # Defensive: never let this allocate on training GPUs by accident. The
    # caller must explicitly opt in by unsetting CUDA_VISIBLE_DEVICES.
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") == "":
        print("[onnx_export] CUDA_VISIBLE_DEVICES is empty — set it to a free "
              "GPU (e.g. CUDA_VISIBLE_DEVICES=0) and re-run. Refusing to "
              "default to GPU 0 because training may be running there.",
              file=sys.stderr)
        return 2

    import torch  # type: ignore
    from transformers import AutoModelForImageTextToText  # type: ignore

    if not torch.cuda.is_available():
        print("[onnx_export] FATAL: CUDA not available.", file=sys.stderr)
        return 2

    dtype = getattr(torch, args.dtype)
    kw = {"attn_implementation": "eager"}  # ONNX export needs eager attn
    try:
        full = AutoModelForImageTextToText.from_pretrained(args.hf_dir, dtype=dtype, **kw)
    except TypeError:
        full = AutoModelForImageTextToText.from_pretrained(args.hf_dir, torch_dtype=dtype, **kw)
    full.eval().cuda()

    # The HF Qwen2.5-VL class wraps {vision_tower, model, lm_head}. We export
    # the inner LM only. The attribute layout has shifted across transformers
    # versions; we probe a couple of candidates.
    lm = None
    for path in ("model.language_model", "language_model", "model"):
        node = full
        ok = True
        for part in path.split("."):
            if not hasattr(node, part):
                ok = False
                break
            node = getattr(node, part)
        if ok and hasattr(node, "forward"):
            lm = node
            break
    if lm is None:
        print("[onnx_export] FATAL: could not locate language_model submodule; "
              f"attrs at top: {list(vars(full).keys())[:20]}", file=sys.stderr)
        return 3

    bs, sl = 1, args.seq_len
    input_ids = torch.zeros((bs, sl), dtype=torch.long, device="cuda")
    attn = torch.ones((bs, sl), dtype=torch.long, device="cuda")
    pos = torch.arange(sl, dtype=torch.long, device="cuda").unsqueeze(0)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[onnx_export] exporting LM -> {out_path} (opset={args.opset}, dtype={args.dtype})")

    # We pass position_ids explicitly because Qwen2.5-VL's RoPE needs them and
    # the default trace would otherwise capture the wrong device path.
    torch.onnx.export(
        lm,
        args=(input_ids,),
        kwargs={"attention_mask": attn, "position_ids": pos, "use_cache": False},
        f=str(out_path),
        input_names=["input_ids", "attention_mask", "position_ids"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "position_ids": {0: "batch", 1: "seq"},
            "last_hidden_state": {0: "batch", 1: "seq"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )
    sz_mb = out_path.stat().st_size / 1e6
    print(f"[onnx_export] wrote {out_path} ({sz_mb:.1f} MB)")
    print("[onnx_export] next steps (NOT executed here):")
    print("  trtexec --onnx={out} --saveEngine={out}.engine --fp16 --useCudaGraph"
          .format(out=out_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
