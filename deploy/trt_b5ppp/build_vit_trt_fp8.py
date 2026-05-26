"""Build a REAL FP8 TensorRT engine for the Qwen3-VL ViT via the mature modelopt
ONNX QDQ path + TensorRT 10 strongly-typed FP8 build.

Pipeline:
  1. Reuse the fp32 ONNX exported by build_vit_trt_engine.py (engines/vit_bf16/onnx/vit.onnx)
  2. modelopt.onnx.quantization.quantize(..., quantize_mode='fp8', calibration_data=real)
     -> inserts calibrated FP8 QDQ nodes (mature tooling)
  3. Build a strongly-typed FP8 TRT engine from the QDQ ONNX (TRT auto-picks fp8 from QDQ).

Usage:
  python build_vit_trt_fp8.py --src-onnx engines/vit_bf16/onnx/vit.onnx
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import apply_venv_shims  # noqa
apply_venv_shims()

import numpy as np
import tensorrt as trt

HERE = os.path.dirname(os.path.abspath(__file__))
SEQ = 11200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-onnx", default=f"{HERE}/engines/vit_bf16/onnx/vit.onnx")
    ap.add_argument("--out", default=f"{HERE}/engines/vit_fp8")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    qdq_onnx = f"{args.out}/vit_fp8_qdq.onnx"

    from modelopt.onnx.quantization import quantize as onnx_quantize

    # The min-max calibrator runs the full ViT forward with ALL intermediate tensors
    # as outputs; the [2,16,5600,5600] attention scores (~4GB each) accumulate in ORT's
    # default growable arena and OOM at 32GB. Force the CUDA EP arena to allocate exactly
    # what's requested and free aggressively (kSameAsRequested), and cap the pool. This
    # lets each block's 4GB score buffer be released before the next block.
    import modelopt.onnx.quantization.ort_utils as _ort_utils
    _orig_prep = _ort_utils._prepare_ep_list

    def _patched_prep(calibration_eps):
        providers = _orig_prep(calibration_eps)
        out = []
        for p in providers:
            if isinstance(p, tuple) and p[0] == "CUDAExecutionProvider":
                opts = dict(p[1])
                opts["arena_extend_strategy"] = "kSameAsRequested"
                opts["gpu_mem_limit"] = 31 * 1024 * 1024 * 1024  # near-full 32GB RTX 5090
                out.append(("CUDAExecutionProvider", opts))
            else:
                out.append(p)
        return out

    _ort_utils._prepare_ep_list = _patched_prep

    calib = np.load("/tmp/vit_calib.npy").astype(np.float32)  # [N, 11200, 1536]
    ncalib = int(os.environ.get("VIT_NCALIB", "3"))
    calib = calib[:ncalib]
    # modelopt splits calib axis-0 by the model input's axis-0 (=SEQ). Stack N
    # samples along axis 0 -> [N*SEQ, 1536] so n_itr = N calibration iterations.
    n = calib.shape[0]
    calib = calib.reshape(n * SEQ, 1536)
    print(f"[1] calib data {calib.shape} (n_itr={n}), input name 'hidden_states'")
    calib_dict = {"hidden_states": calib}

    # Use 'max' (per-tensor amax) calibration, NOT the default 'entropy'. Entropy
    # builds 128-bin histograms over every tensor; the [2,16,5600,5600] attention
    # outputs make that intractably slow / OOM on this graph. 'max' is one pass, the
    # standard fp8 PTQ method, and quantizes the full graph cleanly (no node exclusion,
    # which desyncs modelopt's MLP gelu partition map -> "params not specified").
    print("[2] modelopt ONNX FP8 quantize (calibrated QDQ, max calib)...")
    t0 = time.time()
    onnx_quantize(
        args.src_onnx,
        quantize_mode="fp8",
        calibration_data=calib_dict,
        calibration_method="max",
        calibration_eps=["cuda:0", "cpu"],
        output_path=qdq_onnx,
        high_precision_dtype="fp16",
        use_external_data_format=True,
        log_level="INFO",
    )
    print(f"    QDQ onnx written {qdq_onnx} in {time.time()-t0:.1f}s")

    print("[3] building strongly-typed FP8 TRT engine...")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) | \
            (1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    with open(qdq_onnx, "rb") as f:
        if not parser.parse(f.read(), os.path.abspath(qdq_onnx)):
            for i in range(parser.num_errors):
                print("PARSE ERROR:", parser.get_error(i))
            raise RuntimeError("ONNX parse failed")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
    # strongly-typed: precision comes from the QDQ types in the graph; FP8 flag enables fp8 tactics
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.FP8)
    profile = builder.create_optimization_profile()
    inp = network.get_input(0)
    profile.set_shape(inp.name, [SEQ, 1536], [SEQ, 1536], [SEQ, 1536])
    config.add_optimization_profile(profile)

    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("FP8 engine build failed")
    engine_path = f"{args.out}/vit.engine"
    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"    built FP8 engine in {time.time()-t0:.1f}s -> {engine_path}")


if __name__ == "__main__":
    main()
