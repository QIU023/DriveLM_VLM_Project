#!/usr/bin/env bash
# B.5'' v2 (Qwen3-VL-4B 1-cam multimodal) TRT-LLM pipeline orchestrator.
#
# Chain: P7 (BF16 build) -> P8 (FP8 PTQ+build) -> P9 (NVFP4 PTQ+build)
#        -> P10 (latency: 3 precision x 3 batch sizes)
#        -> P11 (accuracy: full 5119 val L2 per precision)
#        -> P12 (REPORT.md aggregation)
#
# Idempotent: each stage is skipped if its output already exists. Each stage
# pre-checks disk free > 10 GB before starting and aborts with a clear error
# otherwise. All stages log to logs/trt/p7-p12_<timestamp>.log.
#
# DO NOT run while SFT training is in progress — TRT load takes the whole GPU.
# DO NOT modify the live training files (scripts/train_lora.py,
# accelerate_configs/fsdp_8gpu.yaml, the 1cam_qwen3vl_multimodal yaml).
#
# Usage:
#   bash deploy/trt_b5ppp/run_full_pipeline.sh
#   bash deploy/trt_b5ppp/run_full_pipeline.sh --skip-l2     # skip P11 (saves ~2h)
#   bash deploy/trt_b5ppp/run_full_pipeline.sh --dry-run     # echo commands only
#
# Exit codes:
#   0 = full pipeline OK
#   1 = generic error
#   2 = disk pre-check failure
#   3 = required ckpt missing
#   10+ = stage-specific failure (10=P7, 11=P8, 12=P9, 13=P10, 14=P11, 15=P12)

set -uo pipefail
# NOTE: do NOT use `set -e` here — we want each stage's failure to surface
# with a descriptive error AND let later stages skip cleanly. Each stage
# captures its own exit code and decides whether to continue.

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
REPO_ROOT="/workspace/DriveLM_VLM_Project"
DEPLOY_DIR="${REPO_ROOT}/deploy/trt_b5ppp"
LOG_DIR="${REPO_ROOT}/logs/trt"
TS="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="${LOG_DIR}/p7-p12_${TS}.log"

CKPT_ROOT="${REPO_ROOT}/checkpoints_qwen25/nusc_planning_b5pp_1cam_qwen3vl_multimodal"
CKPT_BF16="${CKPT_ROOT}/final"
CKPT_FP8="${CKPT_ROOT}/quant_fp8"
CKPT_NVFP4="${CKPT_ROOT}/quant_nvfp4"

ENGINE_DIR="${DEPLOY_DIR}/engines"
BENCH_OUT_DIR="${REPO_ROOT}/deploy/trt_bench"
RESULTS_DIR="${DEPLOY_DIR}/results"
REPORT_MD="${DEPLOY_DIR}/REPORT.md"

PY_TRT="/venv/trt_llm/bin/python"

# Stage flags
DO_P7=1; DO_P8=1; DO_P9=1; DO_P10=1; DO_P11=1; DO_P12=1
DRY_RUN=0
SKIP_L2=0

for arg in "$@"; do
    case "$arg" in
        --skip-l2)   SKIP_L2=1 ;;
        --dry-run)   DRY_RUN=1 ;;
        --only-p7)   DO_P8=0; DO_P9=0; DO_P10=0; DO_P11=0; DO_P12=0 ;;
        --only-p10)  DO_P7=0; DO_P8=0; DO_P9=0; DO_P11=0; DO_P12=0 ;;
        --skip-p8)   DO_P8=0 ;;
        --skip-p9)   DO_P9=0 ;;
        -h|--help)
            grep '^#' "$0" | head -40
            exit 0 ;;
        *) echo "[ERR] unknown arg: $arg" >&2; exit 1 ;;
    esac
done

mkdir -p "${LOG_DIR}" "${ENGINE_DIR}" "${BENCH_OUT_DIR}" "${RESULTS_DIR}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log() {
    local msg="[$(date '+%H:%M:%S')] $*"
    echo "${msg}" | tee -a "${LOG_FILE}"
}

die() {
    log "FATAL: $*"
    exit "${2:-1}"
}

run() {
    log "RUN: $*"
    if (( DRY_RUN )); then
        return 0
    fi
    "$@" 2>&1 | tee -a "${LOG_FILE}"
    return "${PIPESTATUS[0]}"
}

check_disk() {
    local need_gb="${1:-10}"
    local free_gb
    free_gb=$(df -BG /workspace | awk 'NR==2 {gsub("G",""); print $4}')
    if (( free_gb < need_gb )); then
        die "disk free ${free_gb}G < required ${need_gb}G; stop ALL experiments per disk-panic protocol" 2
    fi
    log "DISK OK: ${free_gb}G free (>= ${need_gb}G needed)"
}

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------
log "============================================================"
log "B.5'' v2 TRT-LLM pipeline orchestrator"
log "  ts        = ${TS}"
log "  ckpt_bf16 = ${CKPT_BF16}"
log "  log       = ${LOG_FILE}"
log "  dry_run   = ${DRY_RUN}"
log "  skip_l2   = ${SKIP_L2}"
log "============================================================"

if [[ ! -d "${CKPT_BF16}" ]]; then
    die "BF16 ckpt missing: ${CKPT_BF16}" 3
fi
if [[ ! -f "${CKPT_BF16}/model.safetensors" ]]; then
    die "BF16 ckpt looks unsaved (no model.safetensors): ${CKPT_BF16}" 3
fi

check_disk 10

# Verify TRT venv imports (dry, no CUDA)
log "PRECHECK: TRT-LLM venv import"
if ! (( DRY_RUN )); then
    "${PY_TRT}" -c "import tensorrt_llm; from tensorrt_llm._torch.models import modeling_qwen3vl; print('OK', tensorrt_llm.__version__)" \
        2>&1 | tee -a "${LOG_FILE}" \
        || die "TRT-LLM venv import failed — fix before running pipeline" 1
fi

# ---------------------------------------------------------------------------
# Stage P7: BF16 engine load probe
# ---------------------------------------------------------------------------
stage_p7() {
    local marker="${ENGINE_DIR}/b5ppp_bf16/build_config.json"
    if [[ -f "${marker}" ]]; then
        log "P7 SKIP: ${marker} exists"
        return 0
    fi
    log "==== STAGE P7: BF16 engine load probe ===="
    check_disk 5
    run "${PY_TRT}" "${DEPLOY_DIR}/build_engine.py" \
        --ckpt "${CKPT_BF16}" \
        --precision bf16 \
        --out "${ENGINE_DIR}/b5ppp_bf16" \
        --max-seq-len 12288 --max-batch-size 8 --free-gpu-mem-frac 0.6
    local rc=$?
    if (( rc != 0 )); then
        log "P7 FAIL (rc=${rc})"
        return 10
    fi
    log "P7 DONE"
    return 0
}

# ---------------------------------------------------------------------------
# Stage P8: FP8 PTQ + engine
# ---------------------------------------------------------------------------
stage_p8() {
    local marker="${ENGINE_DIR}/b5ppp_fp8/build_config.json"
    if [[ -f "${marker}" ]]; then
        log "P8 SKIP: ${marker} exists"
        return 0
    fi
    log "==== STAGE P8: FP8 PTQ + engine ===="
    check_disk 25  # 8 GB bf16 model + 5 GB FP8 ckpt out + headroom

    if [[ ! -d "${CKPT_FP8}" ]] || [[ -z "$(ls -A ${CKPT_FP8} 2>/dev/null)" ]]; then
        run "${PY_TRT}" "${DEPLOY_DIR}/quant_fp8.py" \
            --ckpt "${CKPT_BF16}" \
            --calib-n 128 \
            --out "${CKPT_FP8}"
        local rc=$?
        if (( rc != 0 )); then
            log "P8 quant FAIL (rc=${rc})"
            return 11
        fi
    else
        log "P8: ${CKPT_FP8} already populated; skipping PTQ, building engine only"
    fi

    run "${PY_TRT}" "${DEPLOY_DIR}/build_engine.py" \
        --ckpt "${CKPT_FP8}" \
        --precision fp8 \
        --out "${ENGINE_DIR}/b5ppp_fp8" \
        --max-seq-len 12288 --max-batch-size 8 --free-gpu-mem-frac 0.6
    local rc=$?
    if (( rc != 0 )); then
        log "P8 engine FAIL (rc=${rc})"
        return 11
    fi
    log "P8 DONE"
    return 0
}

# ---------------------------------------------------------------------------
# Stage P9: NVFP4 PTQ + engine
# ---------------------------------------------------------------------------
stage_p9() {
    local marker="${ENGINE_DIR}/b5ppp_nvfp4/build_config.json"
    if [[ -f "${marker}" ]]; then
        log "P9 SKIP: ${marker} exists"
        return 0
    fi
    log "==== STAGE P9: NVFP4 PTQ + engine ===="
    check_disk 20  # NVFP4 ckpt ~3 GB + bf16 source already on disk

    if [[ ! -d "${CKPT_NVFP4}" ]] || [[ -z "$(ls -A ${CKPT_NVFP4} 2>/dev/null)" ]]; then
        run "${PY_TRT}" "${DEPLOY_DIR}/quant_nvfp4.py" \
            --ckpt "${CKPT_BF16}" \
            --calib-n 128 \
            --out "${CKPT_NVFP4}"
        local rc=$?
        if (( rc != 0 )); then
            log "P9 quant FAIL (rc=${rc})"
            return 12
        fi
    else
        log "P9: ${CKPT_NVFP4} already populated; skipping PTQ, building engine only"
    fi

    run "${PY_TRT}" "${DEPLOY_DIR}/build_engine.py" \
        --ckpt "${CKPT_NVFP4}" \
        --precision nvfp4 \
        --out "${ENGINE_DIR}/b5ppp_nvfp4" \
        --max-seq-len 12288 --max-batch-size 8 --free-gpu-mem-frac 0.6
    local rc=$?
    if (( rc != 0 )); then
        log "P9 engine FAIL (rc=${rc})"
        return 12
    fi
    log "P9 DONE"
    return 0
}

# ---------------------------------------------------------------------------
# Stage P10: latency 3 precision x 3 batch sizes
# ---------------------------------------------------------------------------
stage_p10() {
    log "==== STAGE P10: latency bench 3 precision x 3 batch ===="
    check_disk 5

    local rc_any=0
    for prec in bf16 fp8 nvfp4; do
        local ckpt
        case "${prec}" in
            bf16) ckpt="${CKPT_BF16}" ;;
            fp8)  ckpt="${CKPT_FP8}"  ;;
            nvfp4) ckpt="${CKPT_NVFP4}" ;;
        esac
        if [[ ! -d "${ckpt}" ]]; then
            log "P10 SKIP ${prec}: ckpt missing ${ckpt}"
            continue
        fi

        for bs in 1 2 4; do
            local out="${BENCH_OUT_DIR}/B5pp_v2_trt_${prec}_bs${bs}.json"
            if [[ -f "${out}" ]]; then
                log "P10 SKIP: ${out} exists"
                continue
            fi
            run "${PY_TRT}" "${DEPLOY_DIR}/bench_trt.py" \
                --ckpt "${ckpt}" \
                --precision "${prec}" \
                --max-batch-size "${bs}" \
                --n-warmup 3 --n-runs 20 \
                --max-new-tokens 14 \
                --mm-payload real \
                --l2-n 0 \
                --out "${out}"
            local rc=$?
            if (( rc != 0 )); then
                log "P10 FAIL ${prec} bs=${bs} (rc=${rc})"
                rc_any=13
            fi
        done
    done
    if (( rc_any != 0 )); then return ${rc_any}; fi
    log "P10 DONE"
    return 0
}

# ---------------------------------------------------------------------------
# Stage P11: accuracy bench (full 5119 val L2 per precision)
#
# Uses scripts/planning_eval.py (HF path, NOT in-process L2 — see F6/bench_trt
# in-proc L2 disabled by default because reloading HF after TRT OOMs 32GB).
# For precision != bf16 we evaluate on the quantized HF ckpt that modelopt
# wrote (it is loadable by transformers in mixed-precision).
# ---------------------------------------------------------------------------
stage_p11() {
    if (( SKIP_L2 )); then
        log "P11 SKIP: --skip-l2 set"
        return 0
    fi
    log "==== STAGE P11: full 5119 val L2 per precision ===="
    check_disk 5

    local rc_any=0
    for prec in bf16 fp8 nvfp4; do
        local ckpt
        case "${prec}" in
            bf16) ckpt="${CKPT_BF16}" ;;
            fp8)  ckpt="${CKPT_FP8}"  ;;
            nvfp4) ckpt="${CKPT_NVFP4}" ;;
        esac
        if [[ ! -d "${ckpt}" ]]; then
            log "P11 SKIP ${prec}: ckpt missing ${ckpt}"
            continue
        fi
        local out="${RESULTS_DIR}/L2_full_${prec}.json"
        if [[ -f "${out}" ]]; then
            log "P11 SKIP: ${out} exists"
            continue
        fi
        run "${PY_TRT}" "${REPO_ROOT}/scripts/planning_eval.py" \
            --ckpt "${ckpt}" \
            --output "${out}"
            # planning_eval defaults read from configs/nuscenes_planning_full.yaml
            # base + the experiment yaml; no --max-samples => full 5119
        local rc=$?
        if (( rc != 0 )); then
            log "P11 FAIL ${prec} (rc=${rc})"
            rc_any=14
        fi
    done
    if (( rc_any != 0 )); then return ${rc_any}; fi
    log "P11 DONE"
    return 0
}

# ---------------------------------------------------------------------------
# Stage P12: aggregate REPORT.md
# ---------------------------------------------------------------------------
stage_p12() {
    log "==== STAGE P12: aggregate REPORT.md ===="
    check_disk 1

    if (( DRY_RUN )); then
        log "P12 dry-run: would aggregate JSONs into ${REPORT_MD}"
        return 0
    fi

    # The aggregator is intentionally embedded here (no extra script file) — it
    # only stitches JSON summaries the prior stages produced. Lives at REPO
    # root so it can be re-run standalone after a partial pipeline.
    "${PY_TRT}" - <<PYEOF | tee -a "${LOG_FILE}"
import json, os, sys
from pathlib import Path

BENCH = Path("${BENCH_OUT_DIR}")
RESULTS = Path("${RESULTS_DIR}")
OUT = Path("${REPORT_MD}")

lines = ["# B.5'' v2 Qwen3-VL-4B 1-cam multimodal — TRT-LLM deployment report",
         "",
         "Pipeline run timestamp: ${TS}",
         "",
         "## P10 Latency (3 precision x 3 batch sizes)",
         "",
         "| precision | batch | TTFT ms (mean/p50/p99) | full ${} ms | toks/s | mem GB |".replace('\${}', '14 tok'),
         "|-----------|------:|-----------------------:|------------:|-------:|-------:|"]
for prec in ("bf16","fp8","nvfp4"):
    for bs in (1,2,4):
        p = BENCH / f"B5pp_v2_trt_{prec}_bs{bs}.json"
        if not p.exists():
            lines.append(f"| {prec} | {bs} | (missing) | | | |")
            continue
        d = json.load(open(p))
        ttft = d['TTFT_ms']
        full = d['full_traj_ms']
        tps = d['throughput_toks_per_s']
        mem = d['gpu_mem_gb']
        lines.append(f"| {prec} | {bs} | {ttft['mean']:.1f}/{ttft['p50']:.1f}/{ttft['p99']:.1f} | "
                     f"{full['mean']:.1f} | {tps['mean']:.1f} | {mem['bench_peak']:.2f} |")

lines += ["", "## P11 Accuracy (full 5119 val L2)", "",
          "| precision | n_samples | L2_avg | L2_1s | L2_2s | L2_3s |",
          "|-----------|----------:|-------:|------:|------:|------:|"]
for prec in ("bf16","fp8","nvfp4"):
    p = RESULTS / f"L2_full_{prec}.json"
    if not p.exists():
        lines.append(f"| {prec} | (missing) | | | | |")
        continue
    d = json.load(open(p))
    ta = d.get("TemAvg", {})
    n = d.get("n_samples", "?")
    def _f(k):
        v = ta.get(k)
        return f"{v:.4f}" if isinstance(v,(int,float)) else "nan"
    lines.append(f"| {prec} | {n} | {_f('L2_avg')} | {_f('L2_1s')} | {_f('L2_2s')} | {_f('L2_3s')} |")

lines += ["", "## Notes", "",
          "- Stage logs: \`${LOG_FILE}\`",
          "- Engine wrappers: \`${ENGINE_DIR}/b5ppp_{bf16,fp8,nvfp4}/build_config.json\`",
          "- Bench JSONs: \`${BENCH_OUT_DIR}/B5pp_v2_trt_*.json\`",
          "- Calib subset rationale: \`deploy/trt_b5ppp/CALIB_README.md\`",
          ""]

OUT.write_text("\n".join(lines))
print(f"[p12] wrote {OUT} ({len(lines)} lines)")
PYEOF
    local rc=$?
    if (( rc != 0 )); then
        log "P12 FAIL (rc=${rc})"
        return 15
    fi
    log "P12 DONE → ${REPORT_MD}"
    return 0
}

# ---------------------------------------------------------------------------
# Drive
# ---------------------------------------------------------------------------
rc=0
if (( DO_P7 )); then stage_p7;  rc=$?; (( rc != 0 )) && { log "STOP: P7 failed"; exit ${rc}; }; fi
if (( DO_P8 )); then stage_p8;  rc=$?; (( rc != 0 )) && { log "STOP: P8 failed"; exit ${rc}; }; fi
if (( DO_P9 )); then stage_p9;  rc=$?; (( rc != 0 )) && { log "STOP: P9 failed"; exit ${rc}; }; fi
if (( DO_P10 )); then stage_p10; rc=$?; (( rc != 0 )) && { log "P10 had failures; continuing"; }; fi
if (( DO_P11 )); then stage_p11; rc=$?; (( rc != 0 )) && { log "P11 had failures; continuing"; }; fi
if (( DO_P12 )); then stage_p12; rc=$?; (( rc != 0 )) && { log "P12 failed"; exit ${rc}; }; fi

log "============================================================"
log "PIPELINE DONE (overall_rc=${rc})"
log "============================================================"
exit ${rc}
