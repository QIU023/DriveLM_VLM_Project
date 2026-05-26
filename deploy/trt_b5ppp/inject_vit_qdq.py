"""Inject STATIC fp8/fp4 QuantizeLinear/DequantizeLinear pairs into the ViT ONNX.

Bypasses ORT calibration entirely: per-tensor INPUT amax comes from the PyTorch
calibration (collect_vit_amax.py); per-tensor WEIGHT amax is computed directly
from the ONNX weight initializer. We insert Q->DQ on the activation input AND on
the weight of every Linear (MatMul-with-weight + merger Gemm). The attention
score BMMs (activation x activation, the [2,16,5600,5600] tensors) are LEFT in
fp16 -- that is exactly what made ORT OOM, and quantizing them is both risky for
accuracy and unnecessary for the GEMM speedup.

FP8 : per-tensor e4m3, scale = amax / 448.
FP4 : per-tensor (global) NVFP4-style two-level. ONNX opset-21 fp4 QDQ block
      scaling is not portably supported by TRT's parser here, so for FP4 we emit
      per-tensor fp4 (e2m1) QDQ with a single fp32 scale = amax / 6.0 (the e2m1
      max). TRT strongly-typed FP4 then runs the fp4 GEMM.

Usage:
  python inject_vit_qdq.py --mode fp8 --src engines/vit_bf16/onnx/vit.onnx \
      --amax results/vit_input_amax.json --out engines/vit_fp8/onnx/vit_qdq.onnx
"""
import argparse
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper
import onnx_graphsurgeon as gs
import ml_dtypes

HERE = os.path.dirname(os.path.abspath(__file__))

FP8_MAX = 448.0
FP4_MAX = 6.0  # e2m1 max representable


def node_to_module(node_name):
    """'/blocks.0/attn/qkv/MatMul' -> 'blocks.0.attn.qkv'  (drop leading / and op suffix)."""
    s = node_name.lstrip("/")
    parts = s.split("/")
    return ".".join(parts[:-1])  # drop the op (MatMul/Gemm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fp8", "fp4"], required=True)
    ap.add_argument("--src", default=f"{HERE}/engines/vit_bf16/onnx/vit.onnx")
    ap.add_argument("--amax", default=f"{HERE}/results/vit_input_amax.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--graph-dtype", choices=["fp16", "fp32"], default="fp16",
                    help="dtype of the surrounding (non-quantized) graph; DQ output + scales match it")
    args = ap.parse_args()
    graph_np_dtype = np.float16 if args.graph_dtype == "fp16" else np.float32

    if args.mode == "fp8":
        qmax = FP8_MAX
        q_dtype = TensorProto.FLOAT8E4M3FN
        np_zp = np.array(0.0, dtype=ml_dtypes.float8_e4m3fn)
    else:
        qmax = FP4_MAX
        q_dtype = TensorProto.FLOAT4E2M1
        np_zp = np.array(0.0, dtype=ml_dtypes.float4_e2m1fn)

    with open(args.amax) as f:
        in_amax = json.load(f)

    print(f"[1] loading ONNX {args.src}")
    model = onnx.load(args.src, load_external_data=True)
    graph = gs.import_onnx(model)
    tmap = {t.name: t for t in graph.tensors().values()}

    # build module->input_amax lookup
    # collect weight-bearing MatMul / Gemm nodes
    n_q = 0
    missing = []
    for node in list(graph.nodes):
        if node.op not in ("MatMul", "Gemm"):
            continue
        # find weight input (a Constant tensor) and activation input (Variable)
        w_inp = None
        act_inp = None
        for inp in node.inputs:
            if isinstance(inp, gs.Constant):
                if w_inp is None:  # first constant = weight
                    w_inp = inp
            else:
                if act_inp is None:
                    act_inp = inp
        if w_inp is None or act_inp is None:
            continue  # attention BMM (no weight) -> skip, stays fp16
        mod = node_to_module(node.name)
        if mod not in in_amax:
            missing.append((node.name, mod))
            continue
        ia = float(in_amax[mod])

        # ---- weight Q/DQ (static, weight amax from the initializer) ----
        w = w_inp.values  # numpy, shape [in,out] for MatMul, [out,in] for Gemm
        wa = float(np.abs(w.astype(np.float32)).max())
        wscale = np.array(wa / qmax, dtype=np.float32)
        w_dq_out = quant_dequant_tensor(
            graph, w_inp, wscale, q_dtype, np_zp, graph_np_dtype, name=f"{node.name}_w")
        node.inputs = [w_dq_out if x is w_inp else x for x in node.inputs]

        # ---- activation Q/DQ (static, calibrated input amax) ----
        iscale = np.array(ia / qmax, dtype=np.float32)
        a_dq_out = quant_dequant_tensor(
            graph, act_inp, iscale, q_dtype, np_zp, graph_np_dtype, name=f"{node.name}_a")
        node.inputs = [a_dq_out if x is act_inp else x for x in node.inputs]
        n_q += 1

    print(f"[2] quantized {n_q} weight-GEMM nodes ({args.mode}); skipped {len(missing)} missing-amax")
    if missing:
        print("    missing (first 5):", missing[:5])

    graph.cleanup().toposort()
    out_model = gs.export_onnx(graph)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    onnx.save(out_model, args.out, save_as_external_data=True,
              all_tensors_to_one_file=True, location=os.path.basename(args.out) + ".data",
              size_threshold=1024)
    print("[saved]", args.out)


def quant_dequant_tensor(graph, src, scale_np, q_dtype, np_zp, graph_np_dtype, name):
    """Insert QuantizeLinear -> DequantizeLinear after `src` tensor.
    Returns the DequantizeLinear output tensor (Variable) in graph_np_dtype.
    Scale is cast to graph dtype so QuantizeLinear input/scale types agree."""
    scale_c = gs.Constant(f"{name}_scale", scale_np.astype(graph_np_dtype).reshape([]))
    # zero point as the quantized dtype (signals fp8/fp4 to ONNX/TRT)
    zp_onnx_dtype = q_dtype
    if np_zp is not None:
        zp_c = gs.Constant(f"{name}_zp", np_zp.reshape([]))
        q_inputs = [src, scale_c, zp_c]
        dq_inputs_zp = zp_c
    else:
        # no numpy fp4 zp available: emit a typed zero via onnx tensor later;
        # fall back to scale-only (TRT infers dtype from QuantizeLinear output type)
        q_inputs = [src, scale_c]
        dq_inputs_zp = None

    q_out = gs.Variable(f"{name}_q", dtype=None)
    q_node = gs.Node(op="QuantizeLinear", name=f"{name}_QuantizeLinear",
                     inputs=q_inputs, outputs=[q_out])
    dq_out = gs.Variable(f"{name}_dq", dtype=graph_np_dtype)
    dq_inputs = [q_out, scale_c] + ([dq_inputs_zp] if dq_inputs_zp is not None else [])
    dq_node = gs.Node(op="DequantizeLinear", name=f"{name}_DequantizeLinear",
                      inputs=dq_inputs, outputs=[dq_out])
    graph.nodes.append(q_node)
    graph.nodes.append(dq_node)
    return dq_out


if __name__ == "__main__":
    main()
