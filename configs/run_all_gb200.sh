#!/bin/bash
# Run SATS experiments on GB200 (Direction 2: CRP compression + Direction 2.5: Distillation)
# Usage: bash configs/run_all_gb200.sh
#
# Estimated time on GB200: ~4-6 hours total

set -e
cd "$(dirname "$0")/.."

echo "=========================================="
echo "  GB200 SATS Experiment Suite"
echo "  $(date)"
echo "=========================================="

# Step 0: Precompute CRP importance + patch labels (if not done)
if [ ! -f precomputed/crp_importance.pt ]; then
    echo ""
    echo "========== Step 0: Precompute CRP data (~15min) =========="
    python scripts/precompute_crp_data.py --config configs/gb200.yaml
fi

# Direction 2: CRP visual token compression
echo ""
echo "========== 1/4 CRP 4x (ours) =========="
python scripts/train_lora.py --config configs/crp_c4.yaml

echo ""
echo "========== 2/4 CRP 8x (ours) =========="
python scripts/train_lora.py --config configs/crp_c8.yaml

echo ""
echo "========== 3/4 CRP Merge 4x (ours) =========="
python scripts/train_lora.py --config configs/crp_merge_c4.yaml

# Direction 2.5: Region-Aware Relation Distillation (7B → 3B)
echo ""
echo "========== 4/4 Distillation 7B→3B (RRD) =========="
python scripts/train_distill.py --config configs/distill_7b_3b_rrd.yaml

echo ""
echo "=========================================="
echo "  All experiments done! $(date)"
echo "=========================================="
