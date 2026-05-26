"""Run + benchmark + accuracy-check a built Qwen3-VL ViT TRT engine.

Loads the engine, runs the real input ([11200,1536]), compares pooler output to
the HF bf16 ViT reference, and times warmup + N timed forwards.

Usage:
  python run_vit_trt_engine.py --engine engines/vit_bf16/vit.engine --dtype bf16
"""
import argparse
import os
import sys
import time
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import apply_venv_shims  # noqa
apply_venv_shims()

import numpy as np
import torch
import tensorrt as trt

CKPT = "/workspace/DriveLM_VLM_Project/checkpoints_qwen25/nusc_planning_b5pp_1cam_qwen3vl_multimodal/final"
SEQ = 11200
HERE = os.path.dirname(os.path.abspath(__file__))

TRT_TO_TORCH = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.BF16: torch.bfloat16,
    trt.DataType.INT32: torch.int32,
    trt.DataType.INT8: torch.int8,
}


def hf_reference(dtype):
    from transformers import Qwen3VLForConditionalGeneration
    model = Qwen3VLForConditionalGeneration.from_pretrained(CKPT, dtype=dtype, device_map="cuda:0")
    visual = model.model.visual.eval()
    visual.config._attn_implementation = "sdpa"
    data = torch.load("/tmp/vit_real_input.pt")
    pv = data["pv"].to("cuda:0", dtype)
    grid = data["grid"].to("cuda:0")
    with torch.no_grad():
        out = visual(pv, grid_thw=grid)
    ref = out.pooler_output.float().cpu().clone()
    # also time HF
    torch.cuda.synchronize()
    for _ in range(5):
        with torch.no_grad():
            visual(pv, grid_thw=grid)
    torch.cuda.synchronize()
    ts = []
    for _ in range(15):
        torch.cuda.synchronize(); t0 = time.time()
        with torch.no_grad():
            visual(pv, grid_thw=grid)
        torch.cuda.synchronize(); ts.append((time.time() - t0) * 1000)
    hf_ms = float(np.median(ts))
    del model, visual
    torch.cuda.empty_cache()
    return ref, pv.float().cpu(), hf_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tdtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    print("[ref] computing HF reference + timing...")
    ref_pooler, pv_cpu, hf_ms = hf_reference(tdtype)
    print(f"      HF {args.dtype} ViT median = {hf_ms:.2f} ms")

    print("[trt] loading engine", args.engine)
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(args.engine, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()

    # discover IO tensors
    inputs, outputs = [], []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        (inputs if mode == trt.TensorIOMode.INPUT else outputs).append(name)
    print("      inputs:", inputs, "outputs:", outputs)

    in_name = inputs[0]
    in_dtype = TRT_TO_TORCH[engine.get_tensor_dtype(in_name)]
    d_in = pv_cpu.to("cuda:0", in_dtype).contiguous()
    ctx.set_input_shape(in_name, tuple(d_in.shape))
    ctx.set_tensor_address(in_name, d_in.data_ptr())

    out_bufs = {}
    for name in outputs:
        shape = tuple(ctx.get_tensor_shape(name))
        odt = TRT_TO_TORCH[engine.get_tensor_dtype(name)]
        buf = torch.empty(shape, dtype=odt, device="cuda:0")
        out_bufs[name] = buf
        ctx.set_tensor_address(name, buf.data_ptr())

    stream = torch.cuda.Stream()
    ctx.execute_async_v3(stream.cuda_stream)
    stream.synchronize()

    # accuracy: pooler is first output
    pooler_name = outputs[0]
    trt_pooler = out_bufs[pooler_name].float().cpu()
    rel = (trt_pooler - ref_pooler).norm() / ref_pooler.norm()
    print(f"[acc] TRT pooler vs HF {args.dtype} rel-L2 = {rel.item():.4e}  shape={tuple(trt_pooler.shape)}")

    # latency
    for _ in range(8):
        ctx.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    ts = []
    for _ in range(args.n):
        torch.cuda.synchronize(); t0 = time.time()
        ctx.execute_async_v3(stream.cuda_stream)
        stream.synchronize(); ts.append((time.time() - t0) * 1000)
    ts = np.array(ts)
    print(f"[lat] TRT {args.dtype} engine: mean={ts.mean():.2f} ms  median={np.median(ts):.2f} ms  "
          f"min={ts.min():.2f}  std={ts.std():.2f}  (n={args.n})")
    print(f"[lat] speedup vs HF {args.dtype} eager: {hf_ms/np.median(ts):.2f}x")

    res = {
        "engine": args.engine, "dtype": args.dtype,
        "rel_l2_vs_hf": rel.item(),
        "trt_median_ms": float(np.median(ts)), "trt_mean_ms": float(ts.mean()),
        "trt_min_ms": float(ts.min()),
        "hf_eager_median_ms": hf_ms,
        "speedup_vs_hf_eager": hf_ms / float(np.median(ts)),
    }
    out = args.out or f"{HERE}/results/vit_trt_{args.dtype}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print("[saved]", out)


if __name__ == "__main__":
    main()
