# Agent A — torchtitan state-dict adapter for Qwen2.5-VL.
#
# Differences vs Qwen3-VL (which torchtitan's stock adapter handles):
#   *  No `q_norm` / `k_norm` keys on attention layers.
#   *  No `deepstack_merger_list.{}` keys on the vision encoder.
#   *  Vision encoder uses `Qwen2_5_VLRMSNorm` (one `weight` param, no
#      `bias`).  torchtitan's Qwen3VLVisionEncoder uses `nn.LayerNorm`
#      (weight + bias).  We map RMSNorm `weight` -> LayerNorm `weight` and
#      zero-initialise the LayerNorm `bias`.  This is an APPROXIMATION:
#      RMSNorm centers nothing, LayerNorm subtracts the mean before scaling,
#      so the two are not numerically equivalent.  For our planning-SFT
#      use-case the vision encoder is fine-tuned end-to-end from the HF
#      init, so the bias terms are free to learn the offset.  Marked as a
#      known divergence; revisit if vision quality regresses.
#   *  Vision encoder merger: HF has a single fused `model.visual.merger.mlp.0`
#      (Linear) + `mlp.2` (Linear) wrapped in `nn.Sequential`; torchtitan
#      names them `vision_encoder.merger.linear_fc1` / `linear_fc2`.  Maps
#      mlp.0 -> linear_fc1, mlp.2 -> linear_fc2.
#   *  Vision encoder MLP per block: HF has `mlp.gate_proj`, `mlp.up_proj`,
#      `mlp.down_proj` (Qwen2MLP gated activation).  torchtitan has
#      `mlp.linear_fc1`, `mlp.linear_fc2` (single FFN with GELU between).
#      THIS IS A SHAPE MISMATCH — HF intermediate_size=3420 with gated
#      activation (so two parallel 3420 projections); torchtitan expects a
#      single intermediate-dim linear.  We currently emit the HF
#      `mlp.gate_proj` as torchtitan's `linear_fc1` and `mlp.down_proj` as
#      `linear_fc2`, DROPPING `mlp.up_proj`.  This is a lossy approximation
#      retained ONLY because the vision encoder is fine-tuned end-to-end
#      from the partial init anyway.  Documented as an open question; if
#      vision quality matters we should either (a) extend the torchtitan
#      vision MLP to use a Qwen2-style gated activation or (b) initialise
#      from a Qwen3-VL vision encoder that already matches the structure.
#   *  Vision patch_embed: HF Qwen2.5-VL has `proj.weight` (Conv3d, no bias).
#      torchtitan has `proj.weight` (Linear) + `proj.bias` (Linear bias).
#      We flatten the Conv3d weight as in Qwen3-VL and zero-initialise the
#      Linear bias.
#
# The from_hf path is the only one exercised at training startup (loading
# from the R1' final HF ckpt).  The to_hf path is provided for symmetry but
# emits HF Qwen2.5-VL keys for *the subset of parameters we successfully
# loaded* — round-tripping is not exact under the approximations above.

from __future__ import annotations

import re
from typing import Any

import torch

from torchtitan.protocols.state_dict_adapter import StateDictAdapter

from .qwen2_5_vl_model import Qwen2_5_VLModel


class Qwen2_5_VLStateDictAdapter(StateDictAdapter):
    def __init__(
        self, model_config: Qwen2_5_VLModel.Config, hf_assets_path: str | None
    ):
        super().__init__(model_config, hf_assets_path)
        self.model_config = model_config

        # Per-key mapping HF -> torchtitan.  None means "drop on load".
        # Indexed keys use {} placeholders that re.sub replaces with the
        # actual layer/block index at load time.
        self.from_hf_map: dict[str, str | None] = {
            # ===== Language Model =====
            # Qwen2.5-VL HF nests the LM under model.language_model.* (same as
            # Qwen3-VL since transformers >=4.49).
            "model.language_model.embed_tokens.weight": "tok_embeddings.weight",
            # Attention — note: NO q_norm / k_norm (Qwen2.5 doesn't use them).
            "model.language_model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
            "model.language_model.layers.{}.self_attn.q_proj.bias": "layers.{}.attention.wq.bias",
            "model.language_model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
            "model.language_model.layers.{}.self_attn.k_proj.bias": "layers.{}.attention.wk.bias",
            "model.language_model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
            "model.language_model.layers.{}.self_attn.v_proj.bias": "layers.{}.attention.wv.bias",
            "model.language_model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
            "model.language_model.layers.{}.self_attn.rotary_emb.inv_freq": None,
            # Qwen2-style gated MLP (kept as gate_proj/up_proj/down_proj
            # because torchtitan's `make_ffn_config` produces a SwiGLU FFN
            # named w1 (gate), w3 (up), w2 (down) -- same as Qwen3).
            "model.language_model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
            "model.language_model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
            "model.language_model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
            # Layer norms (RMSNorm — same structure as Qwen3-VL).
            "model.language_model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
            "model.language_model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
            # Final norm and lm_head.
            "model.language_model.norm.weight": "norm.weight",
            "lm_head.weight": "output.weight",
            # ===== Vision Encoder =====
            # HF Qwen2.5-VL: model.visual.patch_embed.proj (Conv3d, no bias).
            # We map Conv3d weight -> Linear weight (reshape in code) and
            # synthesize a zero Linear bias.
            "model.visual.patch_embed.proj.weight": "vision_encoder.patch_embed.proj.weight",
            # HF: no learned absolute pos embed table; positions come from
            # rotary_pos_emb (computed each forward) and from the
            # per-resolution interpolation used in the merger.  torchtitan's
            # ViT keeps a `pos_embed` parameter table for interpolation —
            # we leave it at its random init (depth_init via __init__.py),
            # which is sub-optimal but doesn't break the load.
            #
            # Vision transformer blocks (HF "blocks", torchtitan "layers").
            # Block norm1 / norm2: HF uses RMSNorm (single .weight), torchtitan
            # uses LayerNorm (weight + bias).  We map weight->weight, zero bias.
            "model.visual.blocks.{}.norm1.weight": "vision_encoder.layers.{}.norm1.weight",
            "model.visual.blocks.{}.norm2.weight": "vision_encoder.layers.{}.norm2.weight",
            # Attention QKV / proj (Linear with bias — same shape as Qwen3-VL).
            "model.visual.blocks.{}.attn.qkv.weight": "vision_encoder.layers.{}.attn.qkv.weight",
            "model.visual.blocks.{}.attn.qkv.bias": "vision_encoder.layers.{}.attn.qkv.bias",
            "model.visual.blocks.{}.attn.proj.weight": "vision_encoder.layers.{}.attn.proj.weight",
            "model.visual.blocks.{}.attn.proj.bias": "vision_encoder.layers.{}.attn.proj.bias",
            # Vision MLP: HF Qwen2MLP (gated) vs torchtitan single-FFN.
            # APPROXIMATION: gate_proj -> linear_fc1 (drops up_proj), down_proj
            # -> linear_fc2.  See module-level docstring for caveats.
            "model.visual.blocks.{}.mlp.gate_proj.weight": "vision_encoder.layers.{}.mlp.linear_fc1.weight",
            "model.visual.blocks.{}.mlp.down_proj.weight": "vision_encoder.layers.{}.mlp.linear_fc2.weight",
            "model.visual.blocks.{}.mlp.up_proj.weight": None,  # dropped
            # Merger:  HF has ln_q (RMSNorm weight) + mlp.0 + mlp.2 (Linear).
            # torchtitan has merger.norm (LayerNorm weight+bias) +
            # linear_fc1 + linear_fc2.
            "model.visual.merger.ln_q.weight": "vision_encoder.merger.norm.weight",
            "model.visual.merger.mlp.0.weight": "vision_encoder.merger.linear_fc1.weight",
            "model.visual.merger.mlp.0.bias": "vision_encoder.merger.linear_fc1.bias",
            "model.visual.merger.mlp.2.weight": "vision_encoder.merger.linear_fc2.weight",
            "model.visual.merger.mlp.2.bias": "vision_encoder.merger.linear_fc2.bias",
        }

    # ------------------------------------------------------------------
    # HF -> torchtitan
    # ------------------------------------------------------------------

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert HuggingFace Qwen2.5-VL state dict to torchtitan format.

        Side-effects emitted to the output dict:
          * `vision_encoder.patch_embed.proj.bias`: synthesised as zeros
            (HF Qwen2.5-VL Conv3d patch_embed has no bias).
          * `vision_encoder.layers.{}.norm[12].bias`: synthesised as zeros
            (HF RMSNorm has no bias).
          * `vision_encoder.merger.norm.bias`: synthesised as zeros.
        """
        tt_state_dict: dict[str, Any] = {}

        # Qwen2.5-VL ties lm_head to embed_tokens by default.  Mirror Qwen3-VL
        # adapter behavior: synthesize lm_head from embeddings if missing.
        if "lm_head.weight" not in hf_state_dict:
            if "model.language_model.embed_tokens.weight" in hf_state_dict:
                hf_state_dict = dict(hf_state_dict)  # don't mutate caller
                hf_state_dict["lm_head.weight"] = hf_state_dict[
                    "model.language_model.embed_tokens.weight"
                ]
            elif "model.embed_tokens.weight" in hf_state_dict:
                # Legacy: pre-4.49 transformers stored LM under model.* not
                # model.language_model.*.
                hf_state_dict = dict(hf_state_dict)
                hf_state_dict["lm_head.weight"] = hf_state_dict[
                    "model.embed_tokens.weight"
                ]

        # Legacy key remapping: pre-4.49 used `model.layers.*` rather than
        # `model.language_model.layers.*`.  Promote so the rest of the
        # mapping table just works.
        remapped: dict[str, Any] = {}
        for k, v in hf_state_dict.items():
            if k.startswith("model.layers."):
                remapped["model.language_model." + k[len("model."):]] = v
            elif k == "model.embed_tokens.weight":
                remapped["model.language_model.embed_tokens.weight"] = v
            elif k == "model.norm.weight":
                remapped["model.language_model.norm.weight"] = v
            else:
                remapped[k] = v
        hf_state_dict = remapped

        # Track which torchtitan vision-encoder norms we've populated so we
        # can emit the zero-bias companions afterwards.
        seen_norm_layers: set[tuple[str, str]] = set()  # (layer_idx, norm_name)
        seen_merger_norm = False

        for hf_key, value in hf_state_dict.items():
            if re.search(r"\.\d+\.", hf_key):
                hf_abstract = re.sub(r"(\d+)", "{}", hf_key, count=1)
                idx_match = re.search(r"\d+", hf_key)
                if idx_match is None:
                    continue
                idx = idx_match.group(0)

                if hf_abstract not in self.from_hf_map:
                    continue
                tt_pattern = self.from_hf_map[hf_abstract]
                if tt_pattern is None:
                    continue
                tt_key = tt_pattern.format(idx)

                tt_value = value
                tt_state_dict[tt_key] = tt_value

                # Track vision-encoder block norm presence.
                m = re.match(
                    r"vision_encoder\.layers\.(\d+)\.(norm[12])\.weight", tt_key
                )
                if m is not None:
                    seen_norm_layers.add((m.group(1), m.group(2)))

            else:
                if hf_key not in self.from_hf_map:
                    continue
                tt_key = self.from_hf_map[hf_key]
                if tt_key is None:
                    continue
                tt_value = value

                # Conv3d (out, C, T, H, W) -> Linear (out, C*T*H*W).
                if hf_key == "model.visual.patch_embed.proj.weight":
                    tt_value = value.reshape(value.shape[0], -1)
                    tt_state_dict[tt_key] = tt_value
                    # Emit zero bias for the Linear patch embed.
                    tt_state_dict["vision_encoder.patch_embed.proj.bias"] = (
                        torch.zeros(tt_value.shape[0], dtype=tt_value.dtype)
                    )
                    continue

                # Merger ln_q (RMSNorm) -> merger.norm (LayerNorm): set bias=0.
                if hf_key == "model.visual.merger.ln_q.weight":
                    tt_state_dict[tt_key] = tt_value
                    tt_state_dict["vision_encoder.merger.norm.bias"] = (
                        torch.zeros_like(tt_value)
                    )
                    seen_merger_norm = True
                    continue

                tt_state_dict[tt_key] = tt_value

        # Synthesize zero bias for each block norm we loaded.
        for layer_idx, norm_name in seen_norm_layers:
            tt_key = f"vision_encoder.layers.{layer_idx}.{norm_name}.weight"
            bias_key = f"vision_encoder.layers.{layer_idx}.{norm_name}.bias"
            if tt_key in tt_state_dict and bias_key not in tt_state_dict:
                tt_state_dict[bias_key] = torch.zeros_like(tt_state_dict[tt_key])

        # Belt-and-braces: if loading produced merger norm weight but we
        # somehow missed the bias, synthesise it.
        if not seen_merger_norm:
            w = tt_state_dict.get("vision_encoder.merger.norm.weight")
            if w is not None and "vision_encoder.merger.norm.bias" not in tt_state_dict:
                tt_state_dict["vision_encoder.merger.norm.bias"] = torch.zeros_like(w)

        return tt_state_dict

    # ------------------------------------------------------------------
    # torchtitan -> HF (lossy; provided for round-trip symmetry only).
    # ------------------------------------------------------------------

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert torchtitan state dict to HF Qwen2.5-VL format.

        Round-trip caveat: vision MLP `linear_fc1` is mapped back to
        `mlp.gate_proj` only; the HF `mlp.up_proj` cannot be reconstructed
        (we dropped it on load).  Vision encoder LayerNorm biases are
        DROPPED on export (HF Qwen2.5-VL uses RMSNorm without bias).
        """
        to_hf_map = {v: k for k, v in self.from_hf_map.items() if v is not None}
        hf_state_dict: dict[str, Any] = {}

        for tt_key, value in state_dict.items():
            # Drop synthesized LayerNorm biases on the vision side.
            if re.match(r"vision_encoder\.layers\.\d+\.norm[12]\.bias", tt_key):
                continue
            if tt_key == "vision_encoder.patch_embed.proj.bias":
                continue
            if tt_key == "vision_encoder.merger.norm.bias":
                continue

            if re.search(r"\.\d+\.", tt_key):
                tt_abstract = re.sub(r"(\d+)", "{}", tt_key, count=1)
                if tt_abstract not in to_hf_map:
                    continue
                idx_match = re.search(r"\d+", tt_key)
                if idx_match is None:
                    continue
                hf_key = to_hf_map[tt_abstract].format(idx_match.group(0))
                hf_value = value
            else:
                if tt_key not in to_hf_map:
                    continue
                # Tied embeddings: don't emit lm_head if we have weight tying.
                if (
                    tt_key == "output.weight"
                    and getattr(self.model_config, "enable_weight_tying", False)
                ):
                    continue
                hf_key = to_hf_map[tt_key]
                hf_value = value
                # Linear weight -> Conv3d weight reshape for patch_embed.
                if tt_key == "vision_encoder.patch_embed.proj.weight":
                    encoder = self.model_config.vision_encoder
                    hf_value = value.reshape(
                        value.shape[0],
                        encoder.in_channels,
                        encoder.temporal_patch_size,
                        encoder.patch_size,
                        encoder.patch_size,
                    )

            hf_state_dict[hf_key] = hf_value

        return hf_state_dict
