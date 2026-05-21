#!/usr/bin/env bash
# NVFP4 quantization of the DriveLM Qwen2.5-VL LLM backbone (vision tower kept bf16).
#
# NVFP4 is Blackwell-native (RTX 5090 sm_120, DRIVE Thor). Produces a *portable*
# quantized checkpoint; the engine built from it (build_engine.sh) is arch-locked.
#
# VERIFY (per §1 of README — flags move across TRT-LLM versions):
#   - the quantize entry point: examples/quantization/quantize.py  (older)
#     OR `python -m modelopt...` / examples/<qwen2vl>/quantize.py   (newer)
#   - that --qformat nvfp4 is accepted by your installed modelopt
#   - that the script targets the language_model submodule for a VLM (some
#     versions need --model_type / a VLM-specific flag so it doesn't try to
#     quantize the vision encoder)
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/models/qwen25vl-drivelm-merged}"   # LoRA-merged HF dir
OUT_DIR="${OUT_DIR:-/models/ckpt_nvfp4}"
CALIB_SIZE="${CALIB_SIZE:-512}"
DTYPE="${DTYPE:-bfloat16}"

python examples/quantization/quantize.py \
  --model_dir "${MODEL_DIR}" \
  --qformat nvfp4 \
  --calib_size "${CALIB_SIZE}" \
  --dtype "${DTYPE}" \
  --output_dir "${OUT_DIR}"

echo "NVFP4 checkpoint -> ${OUT_DIR} (portable; rebuild engine per target GPU)"
