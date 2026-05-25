#!/usr/bin/env bash
# ============================================================================
# MASTER overnight chain for B.5'' v2 (Qwen3-VL-4B, 1-cam, full-modal planning).
#
# Stages (each resumable via env flag; disk-checked <15G abort before heavy work):
#   S0  poll  : block until training final/ exists AND training procs are gone
#   S1  clean : delete ONLY this run's own checkpoint-<N>/ intermediates;
#               KEEP final/; NEVER touch *.preserved_* or anything else
#   SA  bench : Stage A 9-cell compression benchmark on B.5' Qwen2.5-VL-3B 3-cam
#               (run_compress_benchmark_b5prime.sh; training-free, existing ckpt)
#   SB  trt   : Stage B = build+bench {BF16, FP8, NVFP4}, each with FasterVLM 4x
#               applied to VIDEO tokens at deploy. Logs to logs/trt/.
#   SR  report: aggregate REPORT.md (Stage-A Pareto table + Stage-B 3-precision
#               latency/L2 table).
#
# DO NOT modify the live training files. This script only READS final/ and
# writes new dirs. shellcheck-clean; NOT auto-executed by this prep agent.
#
# Env flags (set to 0 to skip a stage):
#   DO_POLL DO_CLEAN DO_BENCH DO_TRT DO_REPORT   (default all 1)
#   POLL_TIMEOUT_MIN (default 720)  POLL_INTERVAL_S (default 120)
#   COMPRESS_METHOD (default fastervlm)  COMPRESS_RATIO (default 4)
#   DRY_RUN=1 to echo only.
#
# Usage:
#   bash deploy/trt_b5ppp/run_overnight_chain.sh
#   DO_POLL=0 DO_CLEAN=0 bash deploy/trt_b5ppp/run_overnight_chain.sh   # resume at bench
# ============================================================================
set -euo pipefail

REPO_ROOT="/workspace/DriveLM_VLM_Project"
DEPLOY_DIR="${REPO_ROOT}/deploy/trt_b5ppp"
CKPT_ROOT="${REPO_ROOT}/checkpoints_qwen25/nusc_planning_b5pp_1cam_qwen3vl_multimodal"
CKPT_BF16="${CKPT_ROOT}/final"
CKPT_FP8="${CKPT_ROOT}/quant_fp8"
CKPT_NVFP4="${CKPT_ROOT}/quant_nvfp4"

ENGINE_DIR="${DEPLOY_DIR}/engines"
BENCH_OUT_DIR="${REPO_ROOT}/deploy/trt_bench"
RESULTS_DIR="${DEPLOY_DIR}/results"
COMPRESS_OUT_DIR="${REPO_ROOT}/eval_results/compress_bench_b5prime"
TRT_LOG_DIR="${REPO_ROOT}/logs/trt"
REPORT_MD="${DEPLOY_DIR}/REPORT.md"

PY_TRT="/venv/trt_llm/bin/python"
PY_SYS="/usr/bin/python3"

DO_POLL="${DO_POLL:-1}"; DO_CLEAN="${DO_CLEAN:-1}"; DO_SMOKE="${DO_SMOKE:-1}"
DO_BENCH="${DO_BENCH:-1}"; DO_TRT="${DO_TRT:-1}"; DO_REPORT="${DO_REPORT:-1}"
POLL_TIMEOUT_MIN="${POLL_TIMEOUT_MIN:-720}"
POLL_INTERVAL_S="${POLL_INTERVAL_S:-120}"
COMPRESS_METHOD="${COMPRESS_METHOD:-fastervlm}"
COMPRESS_RATIO="${COMPRESS_RATIO:-4}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${ENGINE_DIR}" "${BENCH_OUT_DIR}" "${RESULTS_DIR}" "${TRT_LOG_DIR}"
TS="$(date +%Y%m%d-%H%M%S)"
MASTER_LOG="${TRT_LOG_DIR}/overnight_chain_${TS}.log"

log()  { echo "[$(date '+%F %T')] $*" | tee -a "${MASTER_LOG}"; }
die()  { log "FATAL: $*"; exit 1; }
run()  { log "RUN: $*"; (( DRY_RUN )) && return 0; "$@"; }

check_disk() {
    local need_gb="${1:-15}" free_gb
    free_gb=$(df -BG /workspace | awk 'NR==2 {gsub("G","",$4); print $4}')
    (( free_gb < need_gb )) && die "disk free ${free_gb}G < ${need_gb}G floor; abort per disk-panic protocol"
    log "DISK OK: ${free_gb}G free (>= ${need_gb}G)"
}

# Detect live training: torchrun/accelerate procs touching the experiment, OR
# any process holding a GPU at high util. We check process cmdline for the
# experiment name and for train_lora.py.
training_running() {
    if pgrep -f "train_lora.py" >/dev/null 2>&1; then return 0; fi
    if pgrep -f "nusc_planning_b5pp_1cam_qwen3vl_multimodal" >/dev/null 2>&1; then return 0; fi
    return 1
}

# ---------------------------------------------------------------------------
# S0 — poll until final/ exists AND training procs gone
# ---------------------------------------------------------------------------
stage_poll() {
    log "==== S0 POLL: wait for training final/ + procs gone ===="
    local waited=0 timeout_s=$(( POLL_TIMEOUT_MIN * 60 ))
    while true; do
        local final_ready=0
        if [[ -d "${CKPT_BF16}" ]] && \
           { [[ -f "${CKPT_BF16}/model.safetensors" ]] || ls "${CKPT_BF16}"/model*.safetensors >/dev/null 2>&1; }; then
            final_ready=1
        fi
        if (( final_ready )) && ! training_running; then
            log "S0: final/ present AND no training procs -> proceed"
            return 0
        fi
        (( waited >= timeout_s )) && die "S0 timeout after ${POLL_TIMEOUT_MIN} min (final_ready=${final_ready}, training_running=$(training_running && echo yes || echo no))"
        log "S0: waiting (final_ready=${final_ready}); sleep ${POLL_INTERVAL_S}s (waited ${waited}s)"
        (( DRY_RUN )) && { log "S0 dry-run: skip wait"; return 0; }
        sleep "${POLL_INTERVAL_S}"; waited=$(( waited + POLL_INTERVAL_S ))
    done
}

# ---------------------------------------------------------------------------
# S1 — cleanup ONLY this run's checkpoint-<N>/ intermediates
# ---------------------------------------------------------------------------
stage_clean() {
    log "==== S1 CLEAN: trim checkpoint-<N>/ intermediates (keep final/, *.preserved_*) ===="
    [[ -d "${CKPT_ROOT}" ]] || { log "S1: ckpt root missing; nothing to clean"; return 0; }
    # ONLY match the orchestrated-run pattern checkpoint-<digits>. Never touch
    # final/, final.preserved_*, quant_*, or anything else.
    local found=0
    while IFS= read -r d; do
        [[ -z "${d}" ]] && continue
        found=1
        log "S1: removing intermediate ${d}"
        run rm -rf "${d}"
    done < <(find "${CKPT_ROOT}" -maxdepth 1 -type d -regextype posix-extended \
                  -regex '.*/checkpoint-[0-9]+$' 2>/dev/null)
    (( found == 0 )) && log "S1: no checkpoint-<N>/ intermediates found (already clean)"
    log "S1: kept -> $(ls -1d "${CKPT_ROOT}"/final "${CKPT_ROOT}"/*.preserved_* 2>/dev/null | tr '\n' ' ')"
}

# ---------------------------------------------------------------------------
# SMOKE — cheap GPU sanity before full-scale Stage A/B (NONE of this chain has
# ever executed on GPU; training held the cards through all CPU-only prep). Run
# ONE 50-sample cell (B.5' FasterVLM 4x). If its JSON lacks a finite numeric L2,
# ABORT the whole chain rather than burn hours on a broken full run.
# ---------------------------------------------------------------------------
stage_smoke() {
    log "==== SMOKE: 1 cell x 50 samples (B.5' fastervlm 4x) before full chain ===="
    check_disk 15
    local smoke_json="${COMPRESS_OUT_DIR}/_smoke_fastervlm_r4.json"
    rm -f "${smoke_json}"
    if (( DRY_RUN )); then log "SMOKE: dry-run, skipping"; return 0; fi
    CUDA_VISIBLE_DEVICES=0 MAX_SAMPLES=50 \
        "/usr/bin/python3" "${REPO_ROOT}/scripts/planning_eval_compress_mm.py" \
        --ckpt "${REPO_ROOT}/checkpoints_qwen25/nusc_planning_b5prime_3cam_multimodal/final" \
        --planning-cams "CAM_FRONT,CAM_FRONT_LEFT,CAM_FRONT_RIGHT" --max-length 12288 \
        --spatial-method fastervlm --spatial-ratio 4 --max-samples 50 \
        --batch-size 4 --max-new-tokens 20 --output "${smoke_json}" \
        > "${TRT_LOG_DIR}/smoke_fastervlm_r4.log" 2>&1 || {
            die "SMOKE FAILED: eval crashed; see ${TRT_LOG_DIR}/smoke_fastervlm_r4.log. Chain aborted, no GPU hours wasted on full run."
        }
    # Validate the JSON has a finite numeric L2 (TemAvg or NoAvg).
    "/usr/bin/python3" - "$smoke_json" <<'PYEOF' || die "SMOKE FAILED: ${smoke_json} has no finite L2; chain aborted."
import json, math, sys
d = json.load(open(sys.argv[1]))
def finite(x):
    try: return math.isfinite(float(x))
    except Exception: return False
cands = []
def walk(o):
    if isinstance(o, dict):
        for k,v in o.items():
            if "l2" in k.lower() and finite(v): cands.append(float(v))
            walk(v)
    elif isinstance(o, list):
        for v in o: walk(v)
walk(d)
sys.exit(0 if cands else 1)
PYEOF
    log "SMOKE PASS: finite L2 present in ${smoke_json}; proceeding to full chain."
    rm -f "${smoke_json}"
}

# SA — Stage A 9-cell compression benchmark
# ---------------------------------------------------------------------------
stage_bench() {
    # P3 = compression sweep on B.5' (Qwen2.5-VL-3B, 3-cam) — EXISTING ckpt,
    # training-free. NOT the Qwen3-VL-4B 1-cam model (that only gets FasterVLM
    # 4x inside Stage B TRT). B.5' has no deepstack; the mm eval auto-skips it.
    log "==== SA BENCH: 9-cell spatial-compression benchmark (B.5' Qwen2.5-VL-3B 3-cam) ===="
    check_disk 15
    run bash "${DEPLOY_DIR}/run_compress_benchmark_b5prime.sh"
    log "SA: JSONs -> ${COMPRESS_OUT_DIR}"
}

# ---------------------------------------------------------------------------
# SB — Stage B TRT: build + bench {bf16, fp8, nvfp4} each w/ FasterVLM 4x
# ---------------------------------------------------------------------------
trt_import_check() {
    log "SB: TRT-LLM venv import check"
    run "${PY_TRT}" -c "import tensorrt_llm; from tensorrt_llm._torch.models import modeling_qwen3vl; print('TRT-LLM', tensorrt_llm.__version__, 'qwen3vl OK')"
}

build_and_bench_precision() {
    local prec="$1" ckpt="$2"
    local eng="${ENGINE_DIR}/b5ppp_${prec}"
    local blog="${TRT_LOG_DIR}/build_${prec}_${TS}.log"
    local benchlog="${TRT_LOG_DIR}/bench_${prec}_${TS}.log"
    local out="${BENCH_OUT_DIR}/B5pp_v2_trt_${prec}_compress_${COMPRESS_METHOD}${COMPRESS_RATIO}.json"

    if [[ -f "${eng}/build_config.json" ]]; then
        log "SB: ${prec} engine wrapper exists -> skip build"
    else
        log "SB: build ${prec}"
        run "${PY_TRT}" "${DEPLOY_DIR}/build_engine.py" \
            --ckpt "${ckpt}" --precision "${prec}" --out "${eng}" \
            --max-seq-len 12288 --max-batch-size 8 --free-gpu-mem-frac 0.6 \
            2>&1 | tee -a "${blog}"
    fi

    if [[ -f "${out}" ]]; then
        log "SB: ${prec} bench output exists -> skip"
        return 0
    fi
    log "SB: bench ${prec} with ${COMPRESS_METHOD} x${COMPRESS_RATIO} (video-only compression)"
    run "${PY_TRT}" "${DEPLOY_DIR}/bench_trt.py" \
        --ckpt "${ckpt}" --precision "${prec}" \
        --max-batch-size 1 --n-warmup 3 --n-runs 20 --max-new-tokens 14 \
        --mm-payload real --l2-n 50 \
        --compress-method "${COMPRESS_METHOD}" --compress-ratio "${COMPRESS_RATIO}" \
        --out "${out}" \
        2>&1 | tee -a "${benchlog}"
}

stage_trt() {
    log "==== SB TRT: build+bench bf16/fp8/nvfp4 with ${COMPRESS_METHOD} x${COMPRESS_RATIO} ===="
    trt_import_check

    # bf16: directly from final/
    check_disk 15
    build_and_bench_precision bf16 "${CKPT_BF16}"

    # fp8: PTQ if not present, then build+bench
    check_disk 25
    if [[ ! -d "${CKPT_FP8}" ]] || [[ -z "$(ls -A "${CKPT_FP8}" 2>/dev/null)" ]]; then
        log "SB: FP8 PTQ (calib-n 128)"
        run "${PY_TRT}" "${DEPLOY_DIR}/quant_fp8.py" \
            --ckpt "${CKPT_BF16}" --calib-n 128 --out "${CKPT_FP8}" \
            2>&1 | tee -a "${TRT_LOG_DIR}/quant_fp8_${TS}.log"
    fi
    build_and_bench_precision fp8 "${CKPT_FP8}"

    # nvfp4: PTQ if not present, then build+bench
    check_disk 20
    if [[ ! -d "${CKPT_NVFP4}" ]] || [[ -z "$(ls -A "${CKPT_NVFP4}" 2>/dev/null)" ]]; then
        log "SB: NVFP4 PTQ (calib-n 128)"
        run "${PY_TRT}" "${DEPLOY_DIR}/quant_nvfp4.py" \
            --ckpt "${CKPT_BF16}" --calib-n 128 --out "${CKPT_NVFP4}" \
            2>&1 | tee -a "${TRT_LOG_DIR}/quant_nvfp4_${TS}.log"
    fi
    build_and_bench_precision nvfp4 "${CKPT_NVFP4}"
}

# ---------------------------------------------------------------------------
# SR — aggregate REPORT.md
# ---------------------------------------------------------------------------
stage_report() {
    log "==== SR REPORT: aggregate REPORT.md ===="
    check_disk 1
    (( DRY_RUN )) && { log "SR dry-run: would write ${REPORT_MD}"; return 0; }
    COMPRESS_OUT_DIR="${COMPRESS_OUT_DIR}" BENCH_OUT_DIR="${BENCH_OUT_DIR}" \
    REPORT_MD="${REPORT_MD}" TS="${TS}" CM="${COMPRESS_METHOD}" CR="${COMPRESS_RATIO}" \
    "${PY_SYS}" - <<'PYEOF' | tee -a "${MASTER_LOG}"
import json, os
from pathlib import Path
COMP = Path(os.environ["COMPRESS_OUT_DIR"]); BENCH = Path(os.environ["BENCH_OUT_DIR"])
OUT = Path(os.environ["REPORT_MD"]); TS = os.environ["TS"]
CM = os.environ["CM"]; CR = os.environ["CR"]
L = ["# B.5'' v2 Qwen3-VL-4B 1-cam multimodal — overnight chain report",
     "", f"Run: {TS}", "",
     "## Stage A — spatial-compression Pareto (full val, full-modal, video-only pruning)",
     "",
     "| cell | method | ratio | vis_in | vis_out | L2_avg (TemAvg) | L2_avg (NoAvg) | collision_avg |",
     "|------|--------|------:|-------:|--------:|----------------:|---------------:|--------------:|"]
order = ["baseline","fastervlm_r2","fastervlm_r4","fastervlm_r8","fastervlm_r16",
         "prumerge_r2","prumerge_r4","prumerge_r8","prumerge_r16"]
def f(v): return f"{v:.4f}" if isinstance(v,(int,float)) and v==v else "nan"
for name in order:
    p = COMP / f"{name}.json"
    if not p.exists():
        L.append(f"| {name} | | | | | (missing) | | |"); continue
    d = json.load(open(p)); c = d.get("compression",{})
    L.append(f"| {name} | {d.get('method')} | {d.get('ratio')} | "
             f"{c.get('visual_tokens_in',0):.0f} | {c.get('visual_tokens_out',0):.0f} | "
             f"{f(d.get('TemAvg',{}).get('L2_avg'))} | {f(d.get('NoAvg',{}).get('L2_avg'))} | "
             f"{f(d.get('collision',{}).get('collision_avg'))} |")
L += ["", f"## Stage B — TRT 3-precision (each with {CM} x{CR} video compression)",
      "",
      "| precision | TTFT ms (mean/p50/p99) | full ms | toks/s | mem GB | vis_in | vis_out | L2_avg |",
      "|-----------|-----------------------:|--------:|-------:|-------:|-------:|--------:|-------:|"]
for prec in ("bf16","fp8","nvfp4"):
    p = BENCH / f"B5pp_v2_trt_{prec}_compress_{CM}{CR}.json"
    if not p.exists():
        L.append(f"| {prec} | (missing) | | | | | | |"); continue
    d = json.load(open(p))
    ttft = d.get("TTFT_ms",{}); full = d.get("full_traj_ms",{})
    tps = d.get("throughput_toks_per_s",{}); mem = d.get("gpu_mem_gb",{})
    l2s = d.get("l2_summary") or {}
    L.append(f"| {prec} | {ttft.get('mean',float('nan')):.1f}/{ttft.get('p50',float('nan')):.1f}/{ttft.get('p99',float('nan')):.1f} | "
             f"{full.get('mean',float('nan')):.1f} | {tps.get('mean',float('nan')):.1f} | "
             f"{mem.get('bench_peak',float('nan')):.2f} | {d.get('visual_tokens_in','?')} | "
             f"{d.get('visual_tokens_out','?')} | {f(l2s.get('L2_avg_mean'))} |")
L += ["", "## Notes",
      f"- Stage A baseline L2 expected ~0.658; deviations flag a compression regression.",
      f"- Compression applied to VIDEO tokens only; HD-map+bbox+ego untouched.",
      ""]
OUT.write_text("\n".join(L))
print(f"[report] wrote {OUT}")
PYEOF
    log "SR: -> ${REPORT_MD}"
}

# ---------------------------------------------------------------------------
# Drive
# ---------------------------------------------------------------------------
log "============================================================"
log "OVERNIGHT CHAIN start ts=${TS} dry_run=${DRY_RUN}"
log "  poll=${DO_POLL} clean=${DO_CLEAN} smoke=${DO_SMOKE} bench=${DO_BENCH} trt=${DO_TRT} report=${DO_REPORT}"
log "  compress=${COMPRESS_METHOD} x${COMPRESS_RATIO}"
log "============================================================"
(( DO_POLL ))   && stage_poll
(( DO_CLEAN ))  && stage_clean
(( DO_SMOKE ))  && stage_smoke
(( DO_BENCH ))  && stage_bench
(( DO_TRT ))    && stage_trt
(( DO_REPORT )) && stage_report
log "OVERNIGHT CHAIN DONE -> ${REPORT_MD}"
