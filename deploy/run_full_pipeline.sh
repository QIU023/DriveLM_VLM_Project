#!/usr/bin/env bash
# Single-command end-to-end: HF checkpoint -> NVFP4 ckpt -> LM engine ->
# vision engine -> benchmark CSV. Runs INSIDE the NGC nvcr.io/nvidia/pytorch:25.01-py3
# container (host has cu130 + torch 2.11 which conflicts with TRT-LLM 1.x).
#
# Usage:
#   bash deploy/run_full_pipeline.sh \
#       --hf-dir   /models/qwen25vl-b5prime-merged \
#       --out      /models/engine_b5prime_5090 \
#       --calib-n  512 \
#       --bench-runs 50
#
# All steps are logged + tee'd to /work/deploy_logs/run_<stamp>.log. On any
# step failure the script aborts with that step's rc and prints a diagnostic
# pointing at the failed stage (so the user can re-run with --resume-from).

set -euo pipefail

# --- parse args -------------------------------------------------------------
HF_DIR=""
OUT_DIR=""
CALIB_N="512"
BENCH_RUNS="50"
RESUME_FROM=""   # quant | build_lm | build_vis | benchmark   (debug aid)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hf-dir)        HF_DIR="$2"; shift 2 ;;
    --out)           OUT_DIR="$2"; shift 2 ;;
    --calib-n)       CALIB_N="$2"; shift 2 ;;
    --bench-runs)    BENCH_RUNS="$2"; shift 2 ;;
    --resume-from)   RESUME_FROM="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "[pipeline] unknown arg: $1" >&2; exit 64 ;;
  esac
done

if [[ -z "${HF_DIR}" || -z "${OUT_DIR}" ]]; then
  echo "[pipeline] FATAL: --hf-dir and --out are required" >&2
  exit 64
fi
if [[ ! -d "${HF_DIR}" ]]; then
  echo "[pipeline] FATAL: HF dir does not exist: ${HF_DIR}" >&2
  exit 66
fi

# --- env / paths ------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="${LOG_DIR:-/work/deploy_logs}"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/run_${STAMP}.log"

CKPT_DIR="${OUT_DIR}/ckpt_nvfp4"
ENGINE_DIR="${OUT_DIR}/engine"
VISION_ENGINE_DIR="${OUT_DIR}/vision"
CSV_OUT="${OUT_DIR}/benchmark_${STAMP}.csv"
mkdir -p "${OUT_DIR}" "${CKPT_DIR}" "${ENGINE_DIR}" "${VISION_ENGINE_DIR}"

# Mirror stdout+stderr to log. Using `tee` not `>>` so user sees live progress.
exec > >(tee -a "${LOG}") 2>&1

echo "[pipeline] stamp=${STAMP}"
echo "[pipeline] hf_dir=${HF_DIR}"
echo "[pipeline] out=${OUT_DIR}"
echo "[pipeline] calib_n=${CALIB_N}  bench_runs=${BENCH_RUNS}"
echo "[pipeline] log=${LOG}"

# --- container guard --------------------------------------------------------
# Refuse to run on the host. The host has neither tensorrt nor tensorrt_llm
# importable; importing them under cu130 torch can wedge the CUDA driver and
# crash a concurrent training job.
if ! python3 -c "import tensorrt_llm" 2>/dev/null; then
  cat >&2 <<'EOF'
[pipeline] FATAL: tensorrt_llm not importable.
  This pipeline must run INSIDE the NGC pytorch:25.01-py3 container.
  Bring up the container per deploy/README.md §1, then re-run this script.
EOF
  exit 2
fi

# Surface the TRT-LLM version up-front so the log is reproducible.
python3 -c "import tensorrt_llm, tensorrt; \
print(f'[pipeline] tensorrt={tensorrt.__version__} tensorrt_llm={tensorrt_llm.__version__}')"

step_should_run() {
  # If --resume-from is set, skip earlier steps. Step ordering: quant=1,
  # build_lm=2, build_vis=3, benchmark=4.
  local this="$1"
  if [[ -z "${RESUME_FROM}" ]]; then return 0; fi
  declare -A rank=([quant]=1 [build_lm]=2 [build_vis]=3 [benchmark]=4)
  if [[ "${rank[$this]:-9}" -ge "${rank[$RESUME_FROM]:-9}" ]]; then return 0; fi
  return 1
}

fail() {
  local step="$1" rc="$2"
  echo "[pipeline] STEP FAILED: ${step} (rc=${rc})" >&2
  echo "[pipeline] log: ${LOG}" >&2
  echo "[pipeline] resume with: bash ${SCRIPT_DIR}/run_full_pipeline.sh \\" >&2
  echo "  --hf-dir ${HF_DIR} --out ${OUT_DIR} --resume-from ${step}" >&2
  exit "${rc}"
}

# --- (1) NVFP4 quantize -----------------------------------------------------
if step_should_run quant; then
  echo
  echo "[pipeline] === STEP 1: NVFP4 quantize ==="
  MODEL_DIR="${HF_DIR}" OUT_DIR="${CKPT_DIR}" CALIB_SIZE="${CALIB_N}" \
    bash "${SCRIPT_DIR}/quantize_fp4.sh" || fail quant $?
else
  echo "[pipeline] skip step 1 (resume-from=${RESUME_FROM})"
fi

# --- (2) build LM engine ----------------------------------------------------
if step_should_run build_lm; then
  echo
  echo "[pipeline] === STEP 2: build LM engine ==="
  CKPT_DIR="${CKPT_DIR}" ENGINE_DIR="${ENGINE_DIR}" MAX_BS=1 \
    MAX_INPUT_LEN=4096 MAX_SEQ_LEN=4608 \
    bash "${SCRIPT_DIR}/build_engine.sh" || fail build_lm $?
else
  echo "[pipeline] skip step 2 (resume-from=${RESUME_FROM})"
fi

# --- (3) build vision engine ------------------------------------------------
# Discover the multimodal builder. TRT-LLM 1.x moved this around several
# times — current main has it at examples/models/core/multimodal/
# build_multimodal_engine.py (was examples/qwen2vl/build_visual_engine.py in
# 0.x). We probe candidates and pick the first that exists; if none, error
# out with a discoverable command for the user. (Web-searched 2026-05-22 per
# README §1; expect this to drift again before Thor production cut.)
if step_should_run build_vis; then
  echo
  echo "[pipeline] === STEP 3: build vision engine ==="

  TRT_LLM_DIR="${TRT_LLM_DIR:-/work/TRT-LLM}"
  if [[ ! -d "${TRT_LLM_DIR}" ]]; then
    echo "[pipeline] TRT_LLM_DIR not set or not present; cloning version-pinned ..." >&2
    TLM_VER="$(python3 -c 'import tensorrt_llm; print(tensorrt_llm.__version__)')"
    git clone --depth 1 --branch "v${TLM_VER}" \
      https://github.com/NVIDIA/TensorRT-LLM.git "${TRT_LLM_DIR}" \
      || git clone --depth 1 https://github.com/NVIDIA/TensorRT-LLM.git "${TRT_LLM_DIR}"
  fi

  VIS_BUILDER=""
  for cand in \
      "${TRT_LLM_DIR}/examples/models/core/multimodal/build_multimodal_engine.py" \
      "${TRT_LLM_DIR}/examples/multimodal/build_multimodal_engine.py" \
      "${TRT_LLM_DIR}/examples/qwen2vl/build_visual_engine.py" \
      "${TRT_LLM_DIR}/examples/multimodal/build_visual_engine.py"; do
    if [[ -f "${cand}" ]]; then VIS_BUILDER="${cand}"; break; fi
  done
  if [[ -z "${VIS_BUILDER}" ]]; then
    # Last-ditch: find anything that looks like a vision builder.
    VIS_BUILDER="$(find "${TRT_LLM_DIR}/examples" -maxdepth 6 \
      -regex '.*build_\(multimodal\|visual\)_engine\.py' 2>/dev/null | head -n1)"
  fi
  if [[ -z "${VIS_BUILDER}" ]]; then
    echo "[pipeline] FATAL: could not locate the multimodal/vision engine builder." >&2
    echo "  Searched under ${TRT_LLM_DIR}/examples. Run:" >&2
    echo "    find ${TRT_LLM_DIR}/examples -name 'build_*engine.py'" >&2
    echo "  and set VIS_BUILDER env var, then re-run with --resume-from build_vis." >&2
    fail build_vis 70
  fi
  echo "[pipeline] vision builder: ${VIS_BUILDER}"

  python3 "${VIS_BUILDER}" \
      --model_type qwen2_vl \
      --model_path "${HF_DIR}" \
      --output_dir "${VISION_ENGINE_DIR}" \
      || fail build_vis $?
else
  echo "[pipeline] skip step 3 (resume-from=${RESUME_FROM})"
fi

# --- (4) benchmark ----------------------------------------------------------
if step_should_run benchmark; then
  echo
  echo "[pipeline] === STEP 4: benchmark ==="
  python3 "${SCRIPT_DIR}/benchmark.py" \
      --engine-dir "${ENGINE_DIR}" \
      --vision-engine-dir "${VISION_ENGINE_DIR}" \
      --tokenizer-dir "${HF_DIR}" \
      --n-runs "${BENCH_RUNS}" \
      --batch-size 1 \
      --csv-out "${CSV_OUT}" \
      || fail benchmark $?
else
  echo "[pipeline] skip step 4 (resume-from=${RESUME_FROM})"
fi

echo
echo "[pipeline] DONE."
echo "[pipeline] engine: ${ENGINE_DIR}"
echo "[pipeline] vision: ${VISION_ENGINE_DIR}"
echo "[pipeline] csv:    ${CSV_OUT}"
echo "[pipeline] log:    ${LOG}"
