#!/usr/bin/env bash
# vLLM baseline serving for Qwen2.5-VL VLA — faster setup than TRT-LLM,
# ~70-80% of TRT perf, useful as a sanity-check baseline before paying the
# TRT engine-rebuild cost on every config change.
#
# Time to bring up: ~1h cold (vLLM Blackwell wheel + AWQ quant on merged HF dir).
# Compare against TRT-LLM FP4 engine (deploy/benchmark.py) for the deck table.
#
# Usage:
#   bash deploy/vllm_baseline.sh \
#     /models/qwen25vl-vla-merged \
#     --quantization awq --max-model-len 4608

set -euo pipefail

MODEL_DIR="${1:-/models/qwen25vl-vla-merged}"
shift || true
PORT="${PORT:-8000}"

# Install once (cu130-friendly vLLM nightly required for Blackwell sm_120):
#   /usr/bin/python3 -m pip install vllm --pre  # check vllm release notes for sm_120
#
# vLLM does NOT need a container — host install works if cu130 torch is present.

if ! /usr/bin/python3 -c "import vllm" 2>/dev/null; then
  echo "[vllm_baseline] FATAL: vllm not installed."
  echo "  /usr/bin/python3 -m pip install vllm --pre"
  echo "  (verify Blackwell sm_120 support in release notes before installing)"
  exit 2
fi

/usr/bin/python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --port "$PORT" \
  --max-model-len 4608 \
  --max-num-seqs 1 \
  --dtype bfloat16 \
  "$@"
