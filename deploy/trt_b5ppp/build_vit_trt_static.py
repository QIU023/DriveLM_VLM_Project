"""Build a strongly-typed FP8/FP4 TRT engine from a STATIC-QDQ ViT ONNX
(produced by inject_vit_qdq.py). Reuses the proven bf16/fp16 builder path.

Usage:
  python build_vit_trt_static.py --mode fp8 --onnx engines/vit_fp8/onnx/vit_qdq.onnx \
      --out engines/vit_fp8/vit.engine
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import apply_venv_shims  # noqa
apply_venv_shims()

import tensorrt as trt

HERE = os.path.dirname(os.path.abspath(__file__))
SEQ = 11200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fp8", "fp4"], required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) | \
            (1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    print(f"[1] parsing {args.onnx}")
    if not parser.parse_from_file(args.onnx):
        for i in range(parser.num_errors):
            print("PARSE ERROR:", parser.get_error(i))
        raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 12 << 30)
    # STRONGLY_TYPED networks derive ALL precisions from the graph types (the
    # injected fp8/fp4 QuantizeLinear/DequantizeLinear). Setting BuilderFlag.FP8/
    # FP16 here is rejected ("condition: !config.getFlag(kFP8)") -- do NOT set them.

    profile = builder.create_optimization_profile()
    inp = network.get_input(0)
    profile.set_shape(inp.name, [SEQ, 1536], [SEQ, 1536], [SEQ, 1536])
    config.add_optimization_profile(profile)

    print(f"[2] building strongly-typed {args.mode.upper()} engine...")
    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"{args.mode} engine build failed")
    with open(args.out, "wb") as f:
        f.write(serialized)
    sz = os.path.getsize(args.out) / 1e6
    print(f"[3] built {args.mode} engine in {time.time()-t0:.1f}s -> {args.out} ({sz:.0f} MB)")


if __name__ == "__main__":
    main()
