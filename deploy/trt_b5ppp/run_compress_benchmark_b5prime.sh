#!/usr/bin/env bash
# ============================================================================
# P3 — 9-cell spatial-compression benchmark on B.5' (Qwen2.5-VL-3B, 3-cam,
# full-modal nuScenes planning). TRAINING-FREE: uses the EXISTING B.5' ckpt.
#
# NOTE: B.5' is Qwen2.5-VL — there is NO deepstack. planning_eval_compress_mm.py
# detects deepstack via getattr(real,"deepstack_features",None) and skips the
# deepstack branch when absent (verified), so it runs unchanged for Qwen2.5-VL.
#
# Matrix = baseline + {FasterVLM, PruMerge} x {2,4,8,16} = 9 cells.
# Each cell: full val (~5119), full-modal input (HD-map BEV + bbox + ego KEPT;
# only the 3-cam VIDEO tokens pruned, baseline ~8400 = 3 x 2800). Output per
# cell: L2 (TemAvg + NoAvg), collision_{1s,2s,3s,avg}, visual_tokens_in/out,
# ratio, method. Goal: P3 = max FasterVLM ratio that stays ~lossless vs 0.658.
#
# Dispatch: up to 8 cells concurrent, one per GPU; 9th waits for a free GPU.
# Per-cell resumable (skips existing JSON). Aborts if /workspace free < 15G.
#
# DO NOT run while SFT training owns the GPUs. The master orchestrator gates
# this behind "training procs gone".
#
# Usage:
#   bash deploy/trt_b5ppp/run_compress_benchmark_b5prime.sh
#   bash deploy/trt_b5ppp/run_compress_benchmark_b5prime.sh --dry-run
#   MAX_SAMPLES=200 bash deploy/trt_b5ppp/run_compress_benchmark_b5prime.sh  # smoke
# ============================================================================
set -euo pipefail

REPO_ROOT="/workspace/DriveLM_VLM_Project"
DEPLOY_DIR="${REPO_ROOT}/deploy/trt_b5ppp"
# B.5' = Qwen2.5-VL-3B 3-cam multimodal (existing, training-free).
CKPT="${CKPT:-${REPO_ROOT}/checkpoints_qwen25/nusc_planning_b5prime_3cam_multimodal/final}"
EVAL_PY="${REPO_ROOT}/scripts/planning_eval_compress_mm.py"
PY="/usr/bin/python3"

OUT_DIR="${REPO_ROOT}/eval_results/compress_bench_b5prime"
LOG_DIR="${REPO_ROOT}/logs/compress_bench_b5prime"

# 3-cam: mirror configs/nuscenes_planning_3cam_qwen3vl_multimodal.yaml resolution
# regime (B.5' Qwen2.5-VL used the SAME 3-cam set + max_length=12288).
PLANNING_CAMS="${PLANNING_CAMS:-CAM_FRONT,CAM_FRONT_LEFT,CAM_FRONT_RIGHT}"
MAX_LENGTH="${MAX_LENGTH:-12288}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"   # 0 = full val (~5119)
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-20}"
N_GPUS="${N_GPUS:-8}"
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { echo "[FATAL] $*" >&2; exit 1; }

check_disk() {
    local need_gb="${1:-15}" free_gb
    free_gb=$(df -BG /workspace | awk 'NR==2 {gsub("G","",$4); print $4}')
    (( free_gb < need_gb )) && die "disk free ${free_gb}G < ${need_gb}G floor; abort per disk-panic protocol"
    log "DISK OK: ${free_gb}G free (>= ${need_gb}G)"
}

# ---- Preflight -------------------------------------------------------------
[[ -d "${CKPT}" ]] || die "ckpt missing: ${CKPT}"
[[ -f "${CKPT}/model.safetensors" || -n "$(ls "${CKPT}"/model*.safetensors 2>/dev/null)" ]] \
    || die "ckpt has no model safetensors: ${CKPT}"
[[ -f "${EVAL_PY}" ]] || die "eval entrypoint missing: ${EVAL_PY}"
check_disk 15

# ---- Cell matrix: "name|method|ratio" -------------------------------------
CELLS=(
  "baseline|none|1"
  "fastervlm_r2|fastervlm|2"
  "fastervlm_r4|fastervlm|4"
  "fastervlm_r8|fastervlm|8"
  "fastervlm_r16|fastervlm|16"
  "prumerge_r2|prumerge|2"
  "prumerge_r4|prumerge|4"
  "prumerge_r8|prumerge|8"
  "prumerge_r16|prumerge|16"
)

run_cell() {
    local gpu="$1" name="$2" method="$3" ratio="$4"
    local out="${OUT_DIR}/${name}.json"
    local logf="${LOG_DIR}/${name}.log"
    if [[ -f "${out}" ]]; then
        log "SKIP ${name}: ${out} exists"
        return 0
    fi
    log "LAUNCH ${name} (method=${method} ratio=${ratio}) on GPU ${gpu}"
    if (( DRY_RUN )); then
        echo "DRYRUN: CUDA_VISIBLE_DEVICES=${gpu} ${PY} ${EVAL_PY} --ckpt ${CKPT} \
--planning-cams ${PLANNING_CAMS} --max-length ${MAX_LENGTH} \
--spatial-method ${method} --spatial-ratio ${ratio} --max-samples ${MAX_SAMPLES} \
--batch-size ${BATCH_SIZE} --max-new-tokens ${MAX_NEW_TOKENS} --output ${out}"
        return 0
    fi
    CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" "${EVAL_PY}" \
        --ckpt "${CKPT}" \
        --planning-cams "${PLANNING_CAMS}" \
        --max-length "${MAX_LENGTH}" \
        --spatial-method "${method}" \
        --spatial-ratio "${ratio}" \
        --max-samples "${MAX_SAMPLES}" \
        --batch-size "${BATCH_SIZE}" \
        --max-new-tokens "${MAX_NEW_TOKENS}" \
        --output "${out}" \
        > "${logf}" 2>&1
}

# ---- Scheduler: up to N_GPUS cells concurrent, one per GPU -----------------
declare -A PID_GPU=()
declare -a FREE_GPUS=()
for ((g=0; g<N_GPUS; g++)); do FREE_GPUS+=("$g"); done

reap_one() {
    local pid rc=0
    wait -n 2>/dev/null || true
    for pid in "${!PID_GPU[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid"; rc=$? || true
            FREE_GPUS+=("${PID_GPU[$pid]}")
            unset 'PID_GPU[$pid]'
            (( rc != 0 )) && log "WARN: a cell (pid ${pid}) exited rc=${rc}"
        fi
    done
}

log "=== P3 Stage A (B.5' Qwen2.5-VL-3B 3-cam): ${#CELLS[@]} cells, up to ${N_GPUS} concurrent, MAX_SAMPLES=${MAX_SAMPLES} ==="
log "    cams=${PLANNING_CAMS} max_length=${MAX_LENGTH} ckpt=${CKPT}"
for cell in "${CELLS[@]}"; do
    IFS='|' read -r name method ratio <<< "${cell}"
    if [[ -f "${OUT_DIR}/${name}.json" ]]; then
        log "SKIP ${name}: output exists"
        continue
    fi
    while (( ${#FREE_GPUS[@]} == 0 )); do reap_one; done
    gpu="${FREE_GPUS[0]}"; FREE_GPUS=("${FREE_GPUS[@]:1}")
    run_cell "${gpu}" "${name}" "${method}" "${ratio}" &
    PID_GPU[$!]="${gpu}"
done

while (( ${#PID_GPU[@]} > 0 )); do reap_one; done

log "=== P3 Stage A done. Per-cell JSON in ${OUT_DIR} ==="
ls -1 "${OUT_DIR}"/*.json 2>/dev/null || log "no JSONs (dry-run?)"
