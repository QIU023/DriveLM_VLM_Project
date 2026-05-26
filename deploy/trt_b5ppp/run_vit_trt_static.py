"""Run + benchmark + verify a static-quant (fp8/fp4) ViT TRT engine.

Accuracy is rel-L2 of the engine pooler vs the HF **fp32** ground truth (the
honest baseline; bf16/fp16 engines are also reported vs fp32 in the table).
Latency: 8 warmup + N timed (median), real [11200,1536] input. Speedup is vs
HF bf16 eager (same machine/input) to match the bf16/fp16 rows.
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
    trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16,
    trt.DataType.BF16: torch.bfloat16, trt.DataType.INT32: torch.int32,
    trt.DataType.INT8: torch.int8,
}


def hf_refs():
    """Return (fp32_pooler_ref, bf16_eager_median_ms, pv_cpu_fp32)."""
    from transformers import Qwen3VLForConditionalGeneration
    data = torch.load("/tmp/vit_real_input.pt")
    grid = data["grid"].to("cuda:0")
    pv_cpu = data["pv"].float().cpu()

    # fp32 ground truth
    m32 = Qwen3VLForConditionalGeneration.from_pretrained(CKPT, dtype=torch.float32, device_map="cuda:0")
    v32 = m32.model.visual.eval(); v32.config._attn_implementation = "sdpa"
    with torch.no_grad():
        ref = v32(data["pv"].to("cuda:0", torch.float32), grid_thw=grid).pooler_output.float().cpu().clone()
    del m32, v32; torch.cuda.empty_cache()

    # bf16 eager timing (speedup baseline)
    mb = Qwen3VLForConditionalGeneration.from_pretrained(CKPT, dtype=torch.bfloat16, device_map="cuda:0")
    vb = mb.model.visual.eval(); vb.config._attn_implementation = "sdpa"
    pvb = data["pv"].to("cuda:0", torch.bfloat16)
    with torch.no_grad():
        for _ in range(5):
            vb(pvb, grid_thw=grid)
    torch.cuda.synchronize()
    ts = []
    for _ in range(15):
        torch.cuda.synchronize(); t0 = time.time()
        with torch.no_grad():
            vb(pvb, grid_thw=grid)
        torch.cuda.synchronize(); ts.append((time.time()-t0)*1000)
    bf16_ms = float(np.median(ts))
    del mb, vb; torch.cuda.empty_cache()
    return ref, bf16_ms, pv_cpu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print("[ref] HF fp32 ground truth + bf16-eager timing...")
    ref_pooler, bf16_ms, pv_cpu = hf_refs()
    print(f"      HF bf16 eager median = {bf16_ms:.2f} ms")

    print("[trt] loading", args.engine)
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(args.engine, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()

    inputs, outputs = [], []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        (inputs if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else outputs).append(name)

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
    ctx.execute_async_v3(stream.cuda_stream); stream.synchronize()

    pooler = out_bufs[outputs[0]].float().cpu()
    rel = (pooler - ref_pooler).norm() / ref_pooler.norm()
    print(f"[acc] {args.label} pooler vs HF-fp32 rel-L2 = {rel.item():.4e}  shape={tuple(pooler.shape)}")

    for _ in range(8):
        ctx.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    ts = []
    for _ in range(args.n):
        torch.cuda.synchronize(); t0 = time.time()
        ctx.execute_async_v3(stream.cuda_stream)
        stream.synchronize(); ts.append((time.time()-t0)*1000)
    ts = np.array(ts)
    med = float(np.median(ts))
    print(f"[lat] {args.label}: mean={ts.mean():.2f} median={med:.2f} min={ts.min():.2f} std={ts.std():.2f} (n={args.n})")
    print(f"[lat] speedup vs HF bf16 eager ({bf16_ms:.1f} ms): {bf16_ms/med:.2f}x")

    res = {"engine": args.engine, "label": args.label,
           "rel_l2_vs_hf_fp32": rel.item(), "trt_median_ms": med,
           "trt_mean_ms": float(ts.mean()), "trt_min_ms": float(ts.min()),
           "hf_bf16_eager_median_ms": bf16_ms, "speedup_vs_hf_bf16_eager": bf16_ms/med}
    out = args.out or f"{HERE}/results/vit_trt_{args.label}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print("[saved]", out)


if __name__ == "__main__":
    main()
