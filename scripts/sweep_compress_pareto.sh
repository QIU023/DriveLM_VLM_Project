#!/usr/bin/env bash
# Sweep the spatial × temporal compression grid -> unified Pareto frontier on the
# DRIVING metric (L2 vs. visual-token budget). Training-free (re-eval only).
# See docs/interview_prep_qwen25vl_architecture.md Part 2.5.
#
# Each run writes eval_results/planning_<tag>.json with the achieved token budget
# + L2, so the points are directly comparable. Plot ratio (x) vs L2_avg (y).
set -euo pipefail

CKPT="${CKPT:-checkpoints_qwen25/nuscenes_planning_3b_full_sft/final}"
INFOS_VAL="${INFOS_VAL:-data/uniad_infos/nuscenes_infos_temporal_val.pkl}"
MAX_SAMPLES="${MAX_SAMPLES:-200}"
OUT_DIR="${OUT_DIR:-eval_results}"
mkdir -p "${OUT_DIR}"

run() {  # $1=spatial_method $2=spatial_ratio $3=temporal_method $4=temporal_ratio
  local tag="s_${1}${2}_t_${3}${4}"
  echo "=== ${tag} ==="
  python scripts/planning_eval_compress.py \
    --ckpt "${CKPT}" --infos-val "${INFOS_VAL}" --max-samples "${MAX_SAMPLES}" \
    --spatial-method "$1" --spatial-ratio "$2" \
    --temporal-method "$3" --temporal-ratio "$4" \
    --output "${OUT_DIR}/planning_${tag}.json"
}

# baseline (no compression)
run none 1 none 1

# spatial-only axis
for r in 2 4 8 16; do run fastervlm "$r" none 1; done

# temporal-only axis
for tr in 2 4; do run none 1 temporal_pool "$tr"; done

# combined (the stacking story): spatial 4x × temporal {2,4}
run fastervlm 4 temporal_pool 2
run fastervlm 4 temporal_pool 4

echo "Done. Pareto points in ${OUT_DIR}/planning_*.json (ratio vs L2_avg)."
