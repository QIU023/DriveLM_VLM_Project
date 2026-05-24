"""
Phase3a probe: pass pre-computed multimodal embeddings to TRT-LLM 1.3.0rc15
via the Encode/Prefill disaggregated path (mm_disagg).
"""
import os
import sys
# Force unbuffered stdout/stderr so progress lines stream out even on SIGTERM
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# DO NOT set TLLM_MULTIMODAL_DISAGGREGATED — when set, modeling_qwen3vl.py
# forward() at line 1138 SKIPS `get_multimodal_embeddings()` entirely,
# leaving mm_embeds=[] and crashing at find_input_mm_embeds (line 339:
# "list index out of range"). When the env var is unset, the same line
# 1138 calls `get_multimodal_embeddings(encoder_forward_fn=..., params)`
# which has cache-hit logic: if param.multimodal_data["multimodal_embedding"]
# is already populated, encoder_forward_fn is NEVER invoked. The vision
# encoder is still constructed at init (costs ~5 GB), but the line-920
# ValueError is bypassed because the encoder forward never runs.
os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"

# Workaround venv bug: nvidia-cuda-tileiras metadata missing; tensorrt_llm
# unconditionally calls importlib.metadata.files() on it for sm12+ GPUs.
import importlib.metadata as _md
_orig_files = _md.files
def _files_shim(name):
    try:
        return _orig_files(name)
    except _md.PackageNotFoundError:
        if "tileiras" in name:
            return None
        raise
_md.files = _files_shim

CKPT = "/workspace/DriveLM_VLM_Project/checkpoints_qwen25/nusc_planning_b5pp_1cam_qwen3vl_multimodal/final"

import torch
from transformers import AutoConfig, AutoTokenizer
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.disaggregated_params import DisaggregatedParams
from tensorrt_llm._torch.shared_tensor import SharedTensorContainer
from tensorrt_llm.llmapi import KvCacheConfig

# B.5'' hidden_size=2560, deepstack_num_level=3 -> embed dim = 10240
EMBED_DIM = 2560 * (1 + 3)
NUM_MM_TOKENS = 10

def build_dummy_embed() -> torch.Tensor:
    return torch.randn(NUM_MM_TOKENS, EMBED_DIM, dtype=torch.bfloat16, device="cuda")

def main():
    print(f"[probe] Loading {CKPT} with TLLM_MULTIMODAL_DISAGGREGATED=1 ...", flush=True)
    llm = LLM(
        model=CKPT,
        max_batch_size=1,
        max_seq_len=2048,
        max_num_tokens=2048,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.5),
        trust_remote_code=True,
    )
    print("[probe] LLM loaded", flush=True)

    text_prompt = (
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>"
        "Describe this.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    embed = build_dummy_embed()
    print(f"[probe] dummy embed: shape={tuple(embed.shape)}, dtype={embed.dtype}, device={embed.device}", flush=True)
    handle = SharedTensorContainer.from_tensor(embed).dump_to_dict()
    print(f"[probe] shared handle keys: {sorted(handle.keys())}", flush=True)
    print(f"[probe] tensor_size in handle: {handle.get('tensor_size')}", flush=True)

    # Build mrope position_ids: required because Qwen3-VL uses M-RoPE and
    # _prepare_tp_inputs reads multimodal_data["mrope_config"]["mrope_position_ids"]
    # at model_engine.py:2422. The mm-disagg branch in llm.py:611-620 will
    # populate mrope_config IF we pass mrope_position_ids_handle on the
    # DisaggregatedParams.
    #
    # For this probe we approximate mrope_position_ids as a simple arange of
    # the EXPANDED prompt length (text_len - num_image_tokens + NUM_MM_TOKENS).
    # get_prompt_token_ids will produce the expanded ids, but we need a
    # position_ids tensor of shape (3, 1, expanded_len) for prepare_mrope_config.
    tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
    base_ids = tok(text_prompt, return_tensors="pt").input_ids[0]
    cfg = AutoConfig.from_pretrained(CKPT, trust_remote_code=True)
    image_token_id = cfg.image_token_id
    num_image_tokens_in_prompt = int((base_ids == image_token_id).sum().item())
    expanded_len = len(base_ids) - num_image_tokens_in_prompt + NUM_MM_TOKENS
    fake_pos_ids = torch.arange(expanded_len, dtype=torch.int32, device="cuda")
    mrope_position_ids = fake_pos_ids.view(1, 1, -1).expand(3, 1, -1).contiguous()
    mrope_position_deltas = torch.zeros(1, dtype=torch.int32, device="cuda")
    print(f"[probe] mrope_position_ids: shape={tuple(mrope_position_ids.shape)} dtype={mrope_position_ids.dtype}", flush=True)

    mrope_pos_handle = SharedTensorContainer.from_tensor(mrope_position_ids).dump_to_dict()
    mrope_delta_handle = SharedTensorContainer.from_tensor(mrope_position_deltas).dump_to_dict()

    # request_type="context_and_generation" makes the worker.submit assertion
    # at executor/base_worker.py:469-474 pass without requiring a
    # kv_cache_transceiver (which would be needed for true P-D disagg serving).
    # The mm-disagg branch in llm.py:581-625 is still entered because it only
    # gates on multimodal_embedding_handles being non-None.
    disagg = DisaggregatedParams(
        request_type="context_and_generation",
        multimodal_embedding_handles=[handle],
        mrope_position_ids_handle=mrope_pos_handle,
        mrope_position_deltas_handle=mrope_delta_handle,
    )

    sampling = SamplingParams(max_tokens=8, temperature=0.0)
    print("[probe] Calling llm.generate() with disaggregated_params ...", flush=True)
    out = llm.generate(
        [{"prompt": text_prompt}],
        sampling_params=sampling,
        disaggregated_params=disagg,
    )
    print("[probe] generate returned!", flush=True)
    for o in out:
        print("[probe] prompt_token_ids len:", len(o.prompt_token_ids))
        print("[probe] outputs[0].text:", repr(o.outputs[0].text))
        print("[probe] outputs[0].token_ids:", list(o.outputs[0].token_ids))
    print("[probe] SUCCESS - no ValueError at modeling_qwen3vl.py:920", flush=True)

if __name__ == "__main__":
    main()
