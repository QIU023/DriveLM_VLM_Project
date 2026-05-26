#!/usr/bin/env /venv/trt_llm/bin/python
"""REAL low-precision Linear replacements for the Qwen3-VL ViT (B.5'' deploy).

These are the *measured-speedup* real-GEMM paths found for model.model.visual.
Unlike modelopt fake-quant (no speedup) and modelopt mtq.compress (real GEMM but
its eager+torch.compile wrapper overhead WIPED the win at this ViT size), these
minimal-overhead static modules deliver a genuine measured speedup:

  bf16 ViT (2800-tok video, [2,56,100]):  ~104 ms
  fp8  Linear-only (this StaticFp8Linear): ~92-96 ms  (~1.09-1.13x)
  fp4  Linear-only (this StaticFp4Linear): ~76 ms     (~1.36x)

Speedup is Linear-ONLY (qkv/proj/mlp/merger). Attention is SDPA bf16 (flash-attn
broken in env -> no fp8 attention kernel). Linears are ~36ms of the ~104ms ViT;
attention+norms+patch_embed(conv) ~68ms are untouched, so ~1.5x is the hard
Linear-only ceiling. fp4 GEMM (Blackwell) is faster than fp8 GEMM, hence the
larger win.

ACCURACY CAVEAT: these use crude static per-tensor input scales from ONE calib
forward (raw-hidden rel-L2 ~0.20 fp8 / ~0.40 fp4). They prove the LATENCY win;
the deploy artifact needs proper calibrated/AMAX scales (use modelopt's
calibrated amax, applied through this minimal-overhead GEMM rather than the
modelopt RealQuantLinear wrapper) for accuracy parity.
"""
from __future__ import annotations
import torch
import torch.nn as nn

FP8_MAX = 448.0


class StaticFp8Linear(nn.Module):
    """Real fp8 GEMM Linear via torch._scaled_mm (fp8 x fp8 -> bf16).

    Weight pre-quantized to e4m3 with a static per-tensor scale; input
    dynamically quantized per-tensor each call. No torch.compile, no
    QTensorWrapper -> minimal overhead so the real GEMM win survives.
    """
    def __init__(self, lin: nn.Linear, in_amax: torch.Tensor, device="cuda:0"):
        super().__init__()
        w = lin.weight.data
        self.in_features = w.shape[1]
        self.wscale = (w.abs().amax() / FP8_MAX).float().to(device)
        self.w8 = (w / self.wscale).clamp(-FP8_MAX, FP8_MAX).to(
            torch.float8_e4m3fn).contiguous()
        self.iscale = (in_amax.to(device) / FP8_MAX).float()
        self.bias = lin.bias.data if lin.bias is not None else None
        self.out_dtype = w.dtype

    def forward(self, x):
        shp = x.shape
        x2 = x.reshape(-1, self.in_features)
        x8 = (x2 / self.iscale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        o = torch._scaled_mm(x8, self.w8.t(), scale_a=self.iscale,
                             scale_b=self.wscale, bias=self.bias,
                             out_dtype=self.out_dtype, use_fast_accum=True)
        return o.reshape(*shp[:-1], o.shape[-1])


class StaticFp4Linear(nn.Module):
    """Real NVFP4 GEMM Linear via TRT-LLM ops (fp4_quantize + nvfp4_gemm).

    Uses torch.ops.trtllm.fp4_quantize (uint8-packed fp4 + uint8 block scales)
    and torch.ops.trtllm.nvfp4_gemm. Requires `import tensorrt_llm._torch` first
    to register the ops, and a Blackwell GPU (sm_120). Only valid for Linears
    with in_features % 64 == 0 and out_features % 32 == 0.

    NOTE: this is the manual path because modelopt 0.37's NVFP4 RealQuantLinear
    GEMM is BROKEN against TRT-LLM 1.3.0rc15: nvfp4_gemm.py passes the e4m3
    weight `_scale` where the trtllm op wants a uint8 block scale, raising
    'mat2Scale dtype is Float8_e4m3fn, while Byte is expected'.
    """
    def __init__(self, lin: nn.Linear, in_amax: torch.Tensor, device="cuda:0"):
        super().__init__()
        w = lin.weight.data
        self.in_features = w.shape[1]
        self.n = w.shape[0]
        self.wgs = (448.0 * 6.0 / w.abs().amax().float()).to(device)
        self.wq, self.wsf = torch.ops.trtllm.fp4_quantize(
            w.contiguous(), self.wgs, 16, False)
        self.igs = (448.0 * 6.0 / in_amax.to(device).float())
        self.alpha = (1.0 / (self.wgs * self.igs)).to(device)
        self.bias = lin.bias.data if lin.bias is not None else None
        self.out_dtype = w.dtype

    def forward(self, x):
        shp = x.shape
        x2 = x.reshape(-1, self.in_features)
        xq, xsf = torch.ops.trtllm.fp4_quantize(x2, self.igs, 16, False)
        o = torch.ops.trtllm.nvfp4_gemm(xq, self.wq, xsf, self.wsf,
                                        self.alpha, self.out_dtype)
        if self.bias is not None:
            o = o + self.bias
        return o.reshape(*shp[:-1], self.n)


def collect_linear_input_amax(visual, sample_inputs, cond=None):
    """Run one forward, capture per-Linear input amax for static scales.

    sample_inputs: (pixel_values_videos, video_grid_thw) already on device.
    cond: optional filter on the nn.Linear (e.g. fp4 shape constraints).
    """
    amax, hooks = {}, []
    def mk(name):
        def h(mod, inp, out):
            amax[name] = max(amax.get(name, 0.0), float(inp[0].abs().amax()))
        return h
    for n, m in visual.named_modules():
        if isinstance(m, nn.Linear) and (cond is None or cond(m)):
            hooks.append(m.register_forward_hook(mk(n)))
    with torch.inference_mode():
        visual(*sample_inputs)
    for h in hooks:
        h.remove()
    return amax


def swap_linears(visual, cls, amax, device="cuda:0"):
    """Replace each named Linear in `amax` with `cls`. Returns count swapped."""
    n_swapped = 0
    for name, mod in list(visual.named_modules()):
        if isinstance(mod, nn.Linear) and name in amax:
            parent = visual
            for p in name.split(".")[:-1]:
                parent = getattr(parent, p)
            setattr(parent, name.split(".")[-1],
                    cls(mod, torch.tensor(amax[name]), device=device))
            n_swapped += 1
    return n_swapped
