# Agent A — torchtitan Qwen2.5-VL subclass.
#
# Adapts torchtitan's Qwen3VLModel to Qwen2.5-VL by *normalising the Config*
# at construction time, so that:
#
#   * QK-norm (Qwen3 attention's q_norm / k_norm RMSNorms) is dropped: we set
#     `attention.qk_norm = None` on every layer, which makes torchtitan's
#     `GQAttention.__init__` skip the RMSNorm modules and `GQAttention.forward`
#     short-circuit the normalisation branch.  (See
#     torchtitan/models/common/attention.py L683-L721.)
#   * DeepStack (multi-layer ViT feature injection into early LM layers) is
#     dropped: we set `vision_encoder.deepstack_visual_indices = []`, which
#     makes `Qwen3VLVisionEncoder.__init__` build an empty
#     `deepstack_merger_list` and makes the encoder's forward loop never emit
#     intermediate features.  In the LM, `Qwen3VLModel.__init__` records
#     `num_deepstack_layers = 0`, so the per-layer DeepStack injection in
#     `Qwen3VLModel.forward` is naturally skipped (`layer_idx < 0` is false).
#
# MRoPE (interleaved 3D temporal/height/width position encoding) is *kept*:
# Qwen2.5-VL uses the same `mrope_section` scheme as Qwen3-VL (see
# `apply_multimodal_rotary_pos_emb` in transformers.models.qwen2_5_vl), only
# the section sizes differ (Qwen2.5-VL-3B uses [16, 24, 24] for head_dim 128).
# The section sizes come in via the Config, so no code change is needed.
#
# What we deliberately do NOT subclass:
#   * Vision encoder layer-norm type (Qwen2.5-VL uses RMSNorm; torchtitan's
#     Qwen3VLVisionEncoder uses LayerNorm).  See the README / state-dict
#     adapter for the load-time approximation we apply.
#   * Vision encoder windowed attention (Qwen2.5-VL has window_size=112
#     except at fullatt_block_indexes=[7,15,23,31]; torchtitan uses
#     block-diagonal full attention).  The mismatch is documented as a
#     known divergence; for our planning SFT the vision encoder is fine-
#     tuned end-to-end from the HF init anyway.
#
# Compatibility surface: this subclass keeps the exact same Config schema as
# `Qwen3VLModel.Config`, the same forward() signature, and the same
# parameter tree (modulo the dropped q_norm/k_norm and deepstack_merger_list).
# It therefore plugs into torchtitan's existing
# `parallelize_qwen3_vl(model, ...)` entry point without modification.

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from torchtitan.models.qwen3_vl.model import Qwen3VLModel


class Qwen2_5_VLModel(Qwen3VLModel):
    """Qwen2.5-VL: same code paths as Qwen3VLModel with QK-norm + DeepStack off.

    Construction sequence:
      1.  Normalise the incoming Config:
          a.  Per-layer `attention.qk_norm = None` (drops Q-norm / K-norm).
          b.  `vision_encoder.deepstack_visual_indices = []`
              (drops intermediate-layer feature injection).
      2.  Delegate to `Qwen3VLModel.__init__(normalised_config)`.

    The forward pass is inherited unchanged: the `q_norm is not None`
    branch and the `layer_idx < num_deepstack_layers` branch already short-
    circuit under this Config.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Qwen3VLModel.Config):
        # Subclass with no schema changes — `_owner` auto-wires to
        # Qwen2_5_VLModel via Configurable.__init_subclass__.
        pass

    def __init__(self, config: Config):
        normalised = self._normalise_config(config)
        super().__init__(normalised)

    @staticmethod
    def _normalise_config(config: "Qwen2_5_VLModel.Config") -> "Qwen2_5_VLModel.Config":
        """Return a copy of `config` with QK-norm and DeepStack disabled,
        and with attention Q/K/V projections gaining a learnable bias.

        We use `dataclasses.replace` so the original config (held by the
        Trainer for serialisation) is untouched, and we replace per-layer
        attention configs in-place on the copy because torchtitan's layer
        configs are plain dataclass instances stored in a list.

        Bias note: Qwen2.5-VL HF (transformers.models.qwen2_5_vl) hard-codes
        `bias=True` on q_proj / k_proj / v_proj and `bias=False` on o_proj
        (modelling_qwen2_5_vl.py L704-L707).  Qwen3-VL by contrast has
        `attention_bias=False` everywhere.  We set bias=True on `wq` and
        `wkv` (the latter is shared by both `wk` and `wv` in GQAttention)
        and leave `wo` bias=False to match.
        """
        # Drop DeepStack by zeroing the visual-indices list (a no-op if it
        # is already empty).
        ve = config.vision_encoder
        if ve.deepstack_visual_indices:
            ve = dataclasses.replace(ve, deepstack_visual_indices=[])

        # Drop QK-norm and add Q/K/V bias on every transformer block.
        new_layers = []
        for layer_cfg in config.layers:
            attn = layer_cfg.attention
            updates = {}
            if getattr(attn, "qk_norm", None) is not None:
                updates["qk_norm"] = None
            if hasattr(attn, "wq") and not getattr(attn.wq, "bias", False):
                updates["wq"] = dataclasses.replace(attn.wq, bias=True)
            if hasattr(attn, "wkv") and not getattr(attn.wkv, "bias", False):
                updates["wkv"] = dataclasses.replace(attn.wkv, bias=True)
            if updates:
                attn = dataclasses.replace(attn, **updates)
            new_layers.append(dataclasses.replace(layer_cfg, attention=attn))

        return dataclasses.replace(config, vision_encoder=ve, layers=new_layers)
