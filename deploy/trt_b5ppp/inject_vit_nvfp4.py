"""Inject NVFP4 weight quantization into the fp16 ViT ONNX using the canonical
TRT 2-DequantizeLinear pattern (same structure modelopt emits for FP4 deploy):

  sw_f8 (FLOAT8E4M3FN per-block scales) --DQ1(per-tensor fp32 sw2)--> sw_f16
  w_f4  (FLOAT4E2M1 weight, K-blocked 16) --DQ2(scale=sw_f16, axis, block=16)--> w16

NVFP4 = two-level: per-block (16) fp8-e4m3 scale + one fp32 per-tensor global
scale. Block axis = the K (reduction) axis. We compute everything statically in
numpy/ml_dtypes (no ORT). Activations optionally quantized to fp4 with a static
per-tensor scale (calibrated amax) so TRT fires the fp4 GEMM; --weight-only keeps
activations fp16.
"""
import argparse
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import apply_venv_shims  # noqa
apply_venv_shims()

import numpy as np
import onnx_graphsurgeon as gs
import onnx
import ml_dtypes

HERE = os.path.dirname(os.path.abspath(__file__))
BLOCK = 16
FP8_MAX = 448.0
FP4_MAX = 6.0


def nvfp4_weight(wk):
    """wk: [N, K] fp32 (K = reduction = last axis). Returns:
       w_f4   ml_dtypes.float4_e2m1fn [N, K]  (dequantizable fp4 weight)
       sw_f8  ml_dtypes.float8_e4m3fn [N, K//BLOCK]  (per-block scales)
       sw2    np.float32 scalar (per-tensor global scale)
    Mirrors modelopt get_weights_scaling_factor{,_2} + quantize."""
    N, K = wk.shape
    amax = np.abs(wk).max()
    sw2 = np.float32((amax / FP4_MAX) / FP8_MAX)  # per-tensor global scale
    blk = wk.reshape(N, K // BLOCK, BLOCK)
    per_block_amax = np.abs(blk).max(axis=-1)             # [N, K//BLOCK]
    per_block_scale = (per_block_amax / FP4_MAX)          # fp32
    # quantize block scale to fp8 (divided by global) then back -> effective scale
    sw_f8 = (per_block_scale / sw2).clip(0, FP8_MAX).astype(ml_dtypes.float8_e4m3fn)
    eff_scale = (sw_f8.astype(np.float32) * sw2)          # [N, K//BLOCK]
    eff_scale = np.where(eff_scale == 0, 1.0, eff_scale)  # all-zero block -> no-op scale
    scaled = blk / eff_scale[..., None]                   # weight in fp4 range
    w_f4 = scaled.reshape(N, K).astype(ml_dtypes.float4_e2m1fn)
    return w_f4, sw_f8, sw2


def build_weight_2dq(graph, base, w_f4, sw_f8, sw2, is_gemm, lowp_inits):
    """2-DQ subgraph. w_f4/sw_f8 are [N,K]/[N,K//16] (K-last). Returns the fp16
    weight tensor in the layout the consumer node expects (MatMul:[K,N], Gemm:[N,K]).
    fp4/fp8 initializers are recorded in lowp_inits for post-export injection
    (gs's exporter mis-sizes packed fp4); referenced here as bare gs.Variables."""
    w_f4_t = gs.Variable(f"{base}_w_f4")
    sw_f8_t = gs.Variable(f"{base}_sw_f8")
    lowp_inits[w_f4_t.name] = ("fp4", w_f4)
    lowp_inits[sw_f8_t.name] = ("fp8", sw_f8)
    # fp16 per-tensor scale so DQ1 (and the whole weight chain) outputs fp16,
    # matching the fp16 activation operand at the MatMul.
    sw2_t = gs.Constant(f"{base}_sw2", np.array(sw2, dtype=np.float16).reshape([]))

    sw_f16 = gs.Variable(f"{base}_sw_f16", dtype=np.float16)
    dq1 = gs.Node("DequantizeLinear", f"{base}_swDQ", inputs=[sw_f8_t, sw2_t], outputs=[sw_f16])
    w16 = gs.Variable(f"{base}_w16_klast", dtype=np.float16)
    dq2 = gs.Node("DequantizeLinear", f"{base}_wDQ", inputs=[w_f4_t, sw_f16], outputs=[w16],
                  attrs={"axis": -1, "block_size": BLOCK})
    graph.nodes.extend([dq1, dq2])

    if is_gemm:
        return w16  # Gemm wants [N,K]
    wt = gs.Variable(f"{base}_w16_kn", dtype=np.float16)
    graph.nodes.append(gs.Node("Transpose", f"{base}_wT", inputs=[w16], outputs=[wt],
                               attrs={"perm": [1, 0]}))
    return wt  # MatMul wants [K,N]


def build_activation_fp4(graph, base, act_inp, amax, lowp_inits):
    """Static per-tensor fp4 Q/DQ on the activation (scale = amax/6)."""
    scale = np.float16(amax / FP4_MAX)
    sc = gs.Constant(f"{base}_a_sc", np.array(scale, dtype=np.float16).reshape([]))
    zp = gs.Variable(f"{base}_a_zp")
    lowp_inits[zp.name] = ("fp4", np.array(0.0, dtype=ml_dtypes.float4_e2m1fn).reshape([]))
    a_q = gs.Variable(f"{base}_a_q", dtype=None)
    a_dq = gs.Variable(f"{base}_a_dq", dtype=np.float16)
    graph.nodes.append(gs.Node("QuantizeLinear", f"{base}_aQ",
                               inputs=[act_inp, sc, zp], outputs=[a_q]))
    graph.nodes.append(gs.Node("DequantizeLinear", f"{base}_aDQ",
                               inputs=[a_q, sc, zp], outputs=[a_dq]))
    return a_dq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=f"{HERE}/engines/vit_fp16/onnx/vit.onnx")
    ap.add_argument("--amax", default=f"{HERE}/results/vit_input_amax.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--weight-only", action="store_true")
    args = ap.parse_args()

    with open(args.amax) as f:
        in_amax = json.load(f)

    print(f"[1] loading {args.src}")
    model = onnx.load(args.src, load_external_data=True)
    graph = gs.import_onnx(model)

    n_w = n_a = 0
    skipped = []
    lowp_inits = {}  # name -> (kind, numpy_array) injected post-export
    for node in list(graph.nodes):
        if node.op not in ("MatMul", "Gemm"):
            continue
        w_inp = next((i for i in node.inputs if isinstance(i, gs.Constant)), None)
        act_inp = next((i for i in node.inputs if not isinstance(i, gs.Constant)), None)
        if w_inp is None or act_inp is None:
            continue
        mod = ".".join(node.name.lstrip("/").split("/")[:-1])
        is_gemm = node.op == "Gemm"
        w = w_inp.values.astype(np.float32)
        wk = w if is_gemm else w.T  # make K the last axis -> [N,K]
        if wk.shape[-1] % BLOCK != 0:
            skipped.append((node.name, "K%16", wk.shape)); continue

        w_f4, sw_f8, sw2 = nvfp4_weight(wk)
        w16 = build_weight_2dq(graph, node.name, w_f4, sw_f8, sw2, is_gemm, lowp_inits)
        node.inputs = [w16 if x is w_inp else x for x in node.inputs]
        n_w += 1

        if not args.weight_only:
            a = float(in_amax.get(mod, 0.0))
            if a > 0:
                aq = build_activation_fp4(graph, node.name, act_inp, a, lowp_inits)
                node.inputs = [aq if x is act_inp else x for x in node.inputs]
                n_a += 1

    print(f"[2] NVFP4 weights {n_w} nodes; activations {n_a}; skipped {len(skipped)}")
    for s in skipped[:8]:
        print("   skip", s)

    # do NOT cleanup (would drop our bare-Variable lowp tensors); just toposort
    graph.toposort()
    out_model = gs.export_onnx(graph)

    # inject fp4/fp8 initializers via numpy_helper (correct packing) and drop the
    # bare Variables gs turned into graph inputs.
    from onnx import numpy_helper
    g = out_model.graph
    existing = {i.name for i in g.initializer}
    for name, (kind, arr) in lowp_inits.items():
        if name in existing:
            continue
        g.initializer.append(numpy_helper.from_array(arr, name))
    # remove lowp names from graph inputs (gs promoted producerless Variables)
    lowp_names = set(lowp_inits)
    keep_inputs = [vi for vi in g.input if vi.name not in lowp_names]
    del g.input[:]
    g.input.extend(keep_inputs)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    onnx.save(out_model, args.out, save_as_external_data=True, all_tensors_to_one_file=True,
              location=os.path.basename(args.out) + ".data", size_threshold=1024)
    print("[saved]", args.out)


if __name__ == "__main__":
    main()
