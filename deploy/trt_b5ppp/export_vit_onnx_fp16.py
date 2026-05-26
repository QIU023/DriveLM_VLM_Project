"""Export the ViT export-wrapper ONNX in FP16 (graph + weights fp16, input fp16).

The bf16/fp16 ENGINES were built from the fp32 ONNX + a BF16/FP16 builder flag
that recast the whole graph. STRONGLY_TYPED fp8/fp4 builds ignore those flags --
they take precision verbatim from the ONNX. So for a fast fp8/fp4 engine the
NON-quantized ops (attention, LayerNorm, softmax, elementwise) must already be
fp16 in the ONNX; otherwise they run in fp32 and the engine is slower than bf16.
This exports the wrapper in fp16 so injected fp8/fp4 GEMMs sit in an fp16 graph.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import apply_venv_shims  # noqa
apply_venv_shims()

import torch
from build_vit_trt_engine import (
    VisualExportWrapper, precompute_constants, CKPT, GRID, SEQ,
)
from transformers import Qwen3VLForConditionalGeneration

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    out_dir = f"{HERE}/engines/vit_fp16/onnx"
    os.makedirs(out_dir, exist_ok=True)
    onnx_path = f"{out_dir}/vit.onnx"
    device = "cuda:0"
    td = torch.float16

    print("[1] loading HF model (fp16)...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(CKPT, dtype=td, device_map=device)
    visual = model.model.visual.eval()
    visual.config._attn_implementation = "sdpa"
    pos_embeds, cos_f, sin_f, num_frames = precompute_constants(visual, device)
    wrapper = VisualExportWrapper(
        visual, pos_embeds.to(td), cos_f.to(td), sin_f.to(td), num_frames).eval()

    data = torch.load("/tmp/vit_real_input.pt")
    pv = data["pv"].to(device, td)

    print("[2] exporting fp16 ONNX...")
    # export on GPU in fp16 (CPU fp16 ops are flaky); small enough to fit
    torch.onnx.export(
        wrapper, (pv,), onnx_path, opset_version=17,
        input_names=["hidden_states"],
        output_names=["pooler", "deepstack0", "deepstack1", "deepstack2"],
        dynamo=False,
    )
    print("[saved]", onnx_path)


if __name__ == "__main__":
    main()
