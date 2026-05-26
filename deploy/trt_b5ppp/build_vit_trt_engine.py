"""Build a REAL TensorRT engine for the Qwen3-VL-4B vision tower (model.model.visual).

Strategy
--------
Qwen3-VL's ViT forward computes pos_embeds / rotary cos-sin / cu_seqlens from
`grid_thw` via python loops + `.tolist()` (data-dependent, not ONNX-traceable).
But the deploy target is a FIXED real input: grid_thw=[2,56,100] -> 11200 patch
tokens, single video => cu_seqlens=[0,11200] (one full-attention chunk).

So we precompute pos_embeds + (cos,sin) for this fixed grid ONCE (using the real
HF code), register them as buffers (baked into ONNX as constants), and export a
wrapper whose ONLY input is hidden_states [11200,1536]. The graph is then fully
static & traceable: patch_embed (Conv3d via Linear-equivalent) -> +pos_embed
-> 24 blocks (norm/qkv/rope-apply/full-SDPA/proj/mlp) -> merger + 3 deepstack
mergers. Output: pooler [2800,2560] + 3 deepstack [2800,2560].

Usage:
  python build_vit_trt_engine.py --dtype bf16   # or fp16
"""
import argparse
import os
import sys
import time
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import apply_venv_shims  # noqa
apply_venv_shims()

import torch
import torch.nn as nn
import tensorrt as trt
from transformers import Qwen3VLForConditionalGeneration
import transformers.models.qwen3_vl.modeling_qwen3_vl as qm

CKPT = "/workspace/DriveLM_VLM_Project/checkpoints_qwen25/nusc_planning_b5pp_1cam_qwen3vl_multimodal/final"
GRID = [[2, 56, 100]]
SEQ = 2 * 56 * 100  # 11200
HERE = os.path.dirname(os.path.abspath(__file__))


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class ExportAttention(nn.Module):
    """Single-chunk full attention (cu_seqlens=[0,SEQ]) — matches sdpa eager for one video."""

    def __init__(self, attn):
        super().__init__()
        self.qkv = attn.qkv
        self.proj = attn.proj
        self.num_heads = attn.num_heads
        self.head_dim = attn.qkv.weight.shape[0] // 3 // attn.num_heads
        self.scaling = self.head_dim ** -0.5

    def forward(self, hidden_states, cos, sin):
        # hidden_states: [NF, FL, hidden]; cos/sin: [FL, head_dim]
        nf, fl, _ = hidden_states.shape
        qkv = self.qkv(hidden_states).reshape(nf, fl, 3, self.num_heads, -1).permute(2, 0, 1, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [NF, FL, heads, dim]
        # apply rotary (float math, matches HF apply_rotary_pos_emb_vision)
        c = cos.unsqueeze(-2).float()  # [FL,1,dim]
        s = sin.unsqueeze(-2).float()
        qf, kf = q.float(), k.float()
        q = ((qf * c) + (rotate_half(qf) * s)).to(hidden_states.dtype)
        k = ((kf * c) + (rotate_half(kf) * s)).to(hidden_states.dtype)
        # [NF, FL, heads, dim] -> [NF, heads, FL, dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # full attention WITHIN each frame (batch dim NF = separate attention) -> no mask needed
        attn = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=self.scaling
        )
        # mirror HF sdpa_attention_forward: transpose(1,2) then reshape per frame
        attn = attn.transpose(1, 2).reshape(nf, fl, -1).contiguous()
        return self.proj(attn)


class ExportBlock(nn.Module):
    def __init__(self, blk):
        super().__init__()
        self.norm1 = blk.norm1
        self.norm2 = blk.norm2
        self.attn = ExportAttention(blk.attn)
        self.mlp = blk.mlp

    def forward(self, hidden_states, cos, sin):
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), cos, sin)
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class VisualExportWrapper(nn.Module):
    """Static wrapper: only input is hidden_states [SEQ,1536]; pos/cos/sin baked as buffers."""

    def __init__(self, visual, pos_embeds, cos_f, sin_f, num_frames):
        super().__init__()
        self.patch_embed = visual.patch_embed
        self.blocks = nn.ModuleList([ExportBlock(b) for b in visual.blocks])
        self.merger = visual.merger
        self.deepstack_merger_list = visual.deepstack_merger_list
        self.deepstack_visual_indexes = list(visual.deepstack_visual_indexes)
        self.num_frames = num_frames
        self.register_buffer("pos_embeds", pos_embeds)   # [SEQ, hidden]
        self.register_buffer("cos_f", cos_f)             # [FL, head_dim] per-frame
        self.register_buffer("sin_f", sin_f)

    def forward(self, hidden_states):
        hidden_states = self.patch_embed(hidden_states)          # [SEQ, hidden]
        hidden_states = hidden_states + self.pos_embeds
        hidden = hidden_states.shape[-1]
        nf = self.num_frames
        # [SEQ, hidden] -> [NF, FL, hidden] (frame-major token order)
        hidden_states = hidden_states.reshape(nf, -1, hidden)
        deepstack = []
        for i, blk in enumerate(self.blocks):
            hidden_states = blk(hidden_states, self.cos_f, self.sin_f)
            if i in self.deepstack_visual_indexes:
                idx = self.deepstack_visual_indexes.index(i)
                flat = hidden_states.reshape(-1, hidden)
                deepstack.append(self.deepstack_merger_list[idx](flat))
        pooler = self.merger(hidden_states.reshape(-1, hidden))
        return (pooler, deepstack[0], deepstack[1], deepstack[2])


def precompute_constants(visual, device):
    grid = torch.tensor(GRID, dtype=torch.long, device=device)
    num_frames = int(grid[0, 0])  # t
    frame_len = SEQ // num_frames
    with torch.no_grad():
        pos_embeds = visual.fast_pos_embed_interpolate(grid)  # [SEQ, hidden]
        rotary_pos_emb = visual.rot_pos_emb(grid)             # [SEQ, head_dim//2]
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        # per-frame cos/sin (identical across frames — verified) = first frame_len rows
        cos_f = cos[:frame_len].contiguous()
        sin_f = sin[:frame_len].contiguous()
    return pos_embeds, cos_f, sin_f, num_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tdtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    out_dir = args.out or f"{HERE}/engines/vit_{args.dtype}"
    onnx_dir = f"{out_dir}/onnx"
    os.makedirs(onnx_dir, exist_ok=True)
    device = "cuda:0"

    print(f"[1] loading HF model ({args.dtype})...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(CKPT, dtype=tdtype, device_map=device)
    visual = model.model.visual.eval()
    visual.config._attn_implementation = "sdpa"

    print("[2] precomputing pos_embeds/cos/sin (per-frame batched attn) for grid", GRID)
    pos_embeds, cos_f, sin_f, num_frames = precompute_constants(visual, device)
    print("    pos_embeds", tuple(pos_embeds.shape), "cos_f", tuple(cos_f.shape), "num_frames", num_frames)

    wrapper = VisualExportWrapper(
        visual, pos_embeds.to(tdtype), cos_f.to(tdtype), sin_f.to(tdtype), num_frames
    ).eval()

    # real-shaped input
    data = torch.load("/tmp/vit_real_input.pt")
    pv = data["pv"].to(device, tdtype)
    grid = data["grid"].to(device)

    # reference output from this wrapper in torch (should match full HF)
    with torch.no_grad():
        ref = wrapper(pv)
    print("[3] wrapper torch output: pooler", tuple(ref[0].shape))

    # verify wrapper matches the real HF visual()
    with torch.no_grad():
        hf_out = visual(pv, grid_thw=grid)
    rel = (ref[0].float() - hf_out.pooler_output.float()).norm() / hf_out.pooler_output.float().norm()
    print(f"    wrapper-vs-HF pooler rel-L2 = {rel.item():.3e}")

    # ONNX export in fp32 on CPU (frees GPU mem for the TRT builder), build engine in target dtype.
    print("[4] exporting ONNX (on CPU)...")
    wrapper_fp32 = wrapper.float().cpu()
    pv_fp32 = pv.float().cpu()
    onnx_path = f"{onnx_dir}/vit.onnx"
    torch.onnx.export(
        wrapper_fp32,
        (pv_fp32,),
        onnx_path,
        opset_version=17,
        input_names=["hidden_states"],
        output_names=["pooler", "deepstack0", "deepstack1", "deepstack2"],
        dynamo=False,
    )
    print("    exported", onnx_path)

    # free GPU memory held by HF model + wrappers before the TRT build
    del model, visual, wrapper, wrapper_fp32, pv, pv_fp32
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # build TRT engine (fixed shapes — real input only)
    print(f"[5] building TRT engine ({args.dtype})...")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read(), os.path.abspath(onnx_path)):
            for i in range(parser.num_errors):
                print("PARSE ERROR:", parser.get_error(i))
            raise RuntimeError("ONNX parse failed")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
    if args.dtype == "bf16":
        config.set_flag(trt.BuilderFlag.BF16)
    else:
        config.set_flag(trt.BuilderFlag.FP16)
    # fixed shapes: build a profile pinning hidden_states to [SEQ,1536]
    profile = builder.create_optimization_profile()
    inp = network.get_input(0)
    profile.set_shape(inp.name, [SEQ, 1536], [SEQ, 1536], [SEQ, 1536])
    config.add_optimization_profile(profile)

    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("engine build failed")
    print(f"    built in {time.time()-t0:.1f}s")
    engine_path = f"{out_dir}/vit.engine"
    with open(engine_path, "wb") as f:
        f.write(serialized)
    print("    saved", engine_path)
    # cleanup onnx (large) but keep for fp8 stage; keep it.
    with open(f"{out_dir}/meta.json", "w") as f:
        json.dump({"dtype": args.dtype, "grid": GRID, "seq": SEQ,
                   "wrapper_vs_hf_rel_l2": rel.item()}, f, indent=2)
    print("[done]")


if __name__ == "__main__":
    main()
