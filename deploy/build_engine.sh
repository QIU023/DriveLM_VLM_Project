#!/usr/bin/env bash
# Build the TRT-LLM engine(s) from the NVFP4 checkpoint.
#
# Engine is arch-locked: build on the SAME GPU you will run on (5090 -> sm_120).
# For the 3B VLA, single-GPU is correct — do NOT set tp_size>1 (3B fits one card;
# TP only adds NCCL overhead, and consumer Blackwell has no NVLink). See README §5.
#
# VERIFY (per README §1):
#   - vision engine builder script name: build_visual_engine.py /
#     build_multimodal_engine.py  (varies by version) — find it under the
#     qwen2vl/multimodal example dir
set -euo pipefail

CKPT_DIR="${CKPT_DIR:-/models/ckpt_nvfp4}"
ENGINE_DIR="${ENGINE_DIR:-/models/engine_5090}"
MAX_BS="${MAX_BS:-1}"            # batch=1 mirrors single-ego automotive inference
MAX_INPUT_LEN="${MAX_INPUT_LEN:-4096}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4608}"

# --- LLM engine ---
trtllm-build \
  --checkpoint_dir "${CKPT_DIR}" \
  --output_dir "${ENGINE_DIR}" \
  --gemm_plugin auto \
  --max_batch_size "${MAX_BS}" \
  --max_input_len "${MAX_INPUT_LEN}" \
  --max_seq_len "${MAX_SEQ_LEN}"

# --- vision engine (VERIFY script name in your version) ---
# python examples/<qwen2vl>/build_visual_engine.py \
#   --model_path /models/qwen25vl-drivelm-merged \
#   --output_dir "${ENGINE_DIR}/vision"

echo "Engine -> ${ENGINE_DIR} (sm-locked to this GPU)"
