#!/usr/bin/env bash
# ============================================================================
# P3 — 9-cell spatial-compression benchmark on B.5' (Qwen2.5-VL-3B, 3-cam,
# full-modal nuScenes planning). TRAINING-FREE: uses the EXISTING B.5' ckpt.
#
# DP design: each cell runs the FULL 5119 val under torchrun --nproc_per_node=8
# (8-GPU data-parallel, strided shard, rank0 gathers via all_gather_object —
# mirrors planning_eval.py's proven ~25-30x DP harness). Cells run SEQUENTIALLY
# (one 8-GPU cell at a time) — NOT 8 single-GPU cells in parallel, which
# oversubscribes CPU on the heavy 3-cam video-decode path and crawls.
#
# B.5' is Qwen2.5-VL (NO deepstack); planning_eval_compress_mm.py auto-skips the
# deepstack branch via getattr(real,"deepstack_features",None).
#
# Matrix = baseline + {FasterVLM,PruMerge} x {2,4,8,16} = 9 cells. Each cell:
# full val (~5119), full-modal (HD-map+bbox+ego kept, ONLY 3-cam video pruned).
# Output per cell: L2 TemAvg+NoAvg + collision + visual_tokens_in/out + ratio.
# Goal: P3 = max FasterVLM ratio that stays ~lossless vs the 0.658 baseline.
#
# Per-cell resumable (skips existing JSON). Aborts if /workspace free < 15G.
#
# Usage:
#   bash deploy/trt_b5ppp/run_compress_benchmark_b5prime.sh
#   bash deploy/trt_b5ppp/run_compress_benchmark_b5prime.sh --dry-run
#   MAX_SAMPLES=200 bash deploy/trt_b5ppp/run_compress_benchmark_b5prime.sh  # smoke
# ============================================================================
set -euo pipefail

REPO_ROOT="/workspace/DriveLM_VLM_Project"
CKPT="${CKPT:-${REPO_ROOT}/checkpoints_qwen25/nusc_planning_b5prime_3cam_multimodal/final}"
EVAL_PY="${REPO_ROOT}/scripts/planning_eval_compress_mm.py"

OUT_DIR="${REPO_ROOT}/eval_results/compress_bench_b5prime"
LOG_DIR="${REPO_ROOT}/logs/compress_bench_b5prime"

PLANNING_CAMS="${PLANNING_CAMS:-CAM_FRONT,CAM_FRONT_LEFT,CAM_FRONT_RIGHT}"
MAX_LENGTH="${MAX_LENGTH:-12288}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"          # 0 = full val (~5119)
BATCH_SIZE="${BATCH_SIZE:-2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-20}"
NPROC="${NPROC:-8}"                       # GPUs per cell (DP world size)
MASTER_PORT="${MASTER_PORT:-29577}"
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

mkdir -p "${OUT_DIR}" "${LOG_DIR}"
log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { echo "[FATAL] $*" >&2; exit 1; }

check_disk() {
    local need_gb="${1:-15}" free_gb
    free_gb=$(df -BG /workspace | awk 'NR==2 {gsub("G","",$4); print $4}')
    (( free_gb < need_gb )) && die "disk free ${free_gb}G < ${need_gb}G floor; abort"
    log "DISK OK: ${free_gb}G free"
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
  "fastervlm_r2|fastervlm|2"   "fastervlm_r4|fastervlm|4"
  "fastervlm_r8|fastervlm|8"   "fastervlm_r16|fastervlm|16"
  "prumerge_r2|prumerge|2"     "prumerge_r4|prumerge|4"
  "prumerge_r8|prumerge|8"     "prumerge_r16|prumerge|16"
)

run_cell() {  # sequential, 8-GPU DP per cell
    local name="$1" method="$2" ratio="$3"
    local out="${OUT_DIR}/${name}.json" logf="${LOG_DIR}/${name}.log"
    if [[ -f "${out}" ]]; then log "SKIP ${name}: ${out} exists"; return 0; fi
    log "RUN ${name} (method=${method} ratio=${ratio}) on ${NPROC}-GPU DP"
    local cmd=(torchrun --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}"
        "${EVAL_PY}" --ckpt "${CKPT}"
        --planning-cams "${PLANNING_CAMS}" --max-length "${MAX_LENGTH}"
        --spatial-method "${method}" --spatial-ratio "${ratio}"
        --max-samples "${MAX_SAMPLES}" --batch-size "${BATCH_SIZE}"
        --max-new-tokens "${MAX_NEW_TOKENS}" --output "${out}")
    if (( DRY_RUN )); then echo "DRYRUN: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ${cmd[*]}"; return 0; fi
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "${cmd[@]}" > "${logf}" 2>&1 \
        || die "cell ${name} failed; see ${logf}"
    local l2; l2=$(/usr/bin/python3 -c "import json;print(round(json.load(open('${out}'))['TemAvg']['L2_avg'],4))" 2>/dev/null || echo "?")
    log "DONE ${name}: L2_avg=${l2}"
}

log "=== P3 Stage A (B.5' Qwen2.5-VL-3B 3-cam): ${#CELLS[@]} cells SEQUENTIAL x ${NPROC}-GPU DP, MAX_SAMPLES=${MAX_SAMPLES} ==="
for cell in "${CELLS[@]}"; do
    IFS='|' read -r name method ratio <<< "${cell}"
    run_cell "${name}" "${method}" "${ratio}"
done
log "=== P3 Stage A done. Per-cell JSON in ${OUT_DIR} ==="
ls -1 "${OUT_DIR}"/*.json 2>/dev/null || log "no JSONs (dry-run?)"
