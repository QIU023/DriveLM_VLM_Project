#!/usr/bin/env bash
# Sweep spatial × temporal compression on B.5' (3-cam multi-modal顶配) ckpt.
# Single GPU per cell (planning_eval_compress.py limitation); 200 samples by
# default for Pareto-shape signal; bump MAX_SAMPLES=5119 for paper-grade.
#
# OUTPUT: eval_results/sweep_compress_pareto_b5prime/planning_<tag>.json
#
# NOTE: B.5' ckpt was trained with multimodal inputs (HD-map + bbox + 3-cam).
# This script evaluates with --planning-cams 3-cam but WITHOUT --multimodal
# (planning_eval_compress.py does not yet support --multimodal — see TODO in
# scripts/planning_eval_compress.py). For comparable-to-training eval, see
# the parallel agent task adding multimodal support to that script.
set -uo pipefail  # NOT -e: per-cell failures must not kill the sweep
cd "$(dirname "$0")/.."

CKPT="${CKPT:-checkpoints_qwen25/nusc_planning_b5prime_3cam_multimodal/final}"
INFOS_VAL="${INFOS_VAL:-data/uniad_infos/nuscenes_infos_temporal_val.pkl}"
MAX_SAMPLES="${MAX_SAMPLES:-500}"
OUT_DIR="${OUT_DIR:-eval_results/sweep_compress_pareto_b5prime}"
PLANNING_CAMS="${PLANNING_CAMS:-CAM_FRONT,CAM_FRONT_LEFT,CAM_FRONT_RIGHT}"
mkdir -p "${OUT_DIR}"

LOG_DIR="logs/sweep_b5prime"
mkdir -p "$LOG_DIR"

run() {  # $1=spatial_method $2=spatial_ratio $3=temporal_method $4=temporal_ratio
  local tag="s_${1}${2}_t_${3}${4}"
  echo "=== $(date -Iseconds)  ${tag} ==="
  /usr/bin/python3 scripts/planning_eval_compress.py \
    --ckpt "${CKPT}" --infos-val "${INFOS_VAL}" --max-samples "${MAX_SAMPLES}" \
    --planning-cams "${PLANNING_CAMS}" \
    --spatial-method "$1" --spatial-ratio "$2" \
    --temporal-method "$3" --temporal-ratio "$4" \
    --output "${OUT_DIR}/planning_${tag}.json" \
    > "${LOG_DIR}/${tag}.log" 2>&1
  if [[ -f "${OUT_DIR}/planning_${tag}.json" ]]; then
    L2=$(/usr/bin/python3 -c "import json; d=json.load(open('${OUT_DIR}/planning_${tag}.json')); v=d.get('TemAvg',{}).get('L2_avg', d.get('L2_avg', 'nan')); print(f'{v:.4f}' if isinstance(v,float) else str(v))" 2>/dev/null || echo "?")
    echo "  ${tag}: L2=${L2}"
  else
    echo "  ${tag}: FAILED (no JSON — see logs/sweep_b5prime/${tag}.log)"
  fi
}

# baseline (no compression) — keep, even if already done from prior incomplete run
run none 1 none 1

# 2D spatial axis (training-free) — RUN FIRST, no placeholder mismatch issues
for sm in fastervlm prumerge pyramiddrop; do
  for sr in 2 4 8; do
    run "$sm" "$sr" none 1
  done
done

# 1D temporal axis — KNOWN BUG (placeholder count mismatch on 3-cam, agent task to fix).
# Run anyway; cells will fail with [no JSON] and we'll patch + re-run later.
for tm in temporal_pool vtm longvu; do
  for tr in 2 4; do
    run none 1 "$tm" "$tr"
  done
done

# combined (joint compression Pareto) — depends on temporal fix
run fastervlm 4 temporal_pool 2
run fastervlm 4 temporal_pool 4

echo
echo "=== aggregated Pareto ==="
/usr/bin/python3 -c "
import json, os, glob
rows = []
for fp in sorted(glob.glob('${OUT_DIR}/planning_*.json')):
    d = json.load(open(fp))
    tag = os.path.basename(fp).replace('planning_', '').replace('.json', '')
    rows.append((tag, d.get('L2_avg', float('nan')), d.get('collision_avg', float('nan'))*100, d.get('n_samples', 0)))
print(f'{'TAG':<30s} {'L2_avg':>8s}  {'coll%':>6s}  {'n':>5s}')
for tag, l2, coll, n in rows:
    print(f'{tag:<30s} {l2:>8.4f}  {coll:>5.2f}%  {n:>5d}')
"
echo "Done. Pareto points in ${OUT_DIR}/planning_*.json"
