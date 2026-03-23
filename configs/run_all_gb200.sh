#!/bin/bash
# Run all experiments on GB200 (Direction 2 + 2.5)
# Usage: bash configs/run_all_gb200.sh
#
# Estimated time on GB200: ~8-12 hours total

set -e
cd "$(dirname "$0")/.."

echo "=========================================="
echo "  GB200 Full Experiment Suite"
echo "  $(date)"
echo "=========================================="

# Step 0: Precompute CRP importance (if not done)
if [ ! -f precomputed/crp_importance.pt ]; then
    echo ""
    echo "========== Step 0: Precompute CRP data (~15min) =========="
    python scripts/precompute_crp_data.py --config configs/gb200.yaml
fi

# Direction 2: Visual Token Compression (existing baselines + CRP)
echo ""
echo "========== 1/9 Baseline (no compression) =========="
python scripts/train_lora.py --config configs/baseline.yaml

echo ""
echo "========== 2/9 Avg Pool 4x =========="
python scripts/train_lora.py --config configs/avg_pool_c4.yaml

echo ""
echo "========== 3/9 FasterVLM 4x =========="
python scripts/train_lora.py --config configs/fastervlm_c4.yaml

echo ""
echo "========== 4/9 PruMerge 4x =========="
python scripts/train_lora.py --config configs/prumerge_c4.yaml

echo ""
echo "========== 5/9 PyramidDrop 4x =========="
python scripts/train_lora.py --config configs/pyramiddrop_c4.yaml

echo ""
echo "========== 6/9 CRP 4x (ours) =========="
python scripts/train_lora.py --config configs/crp_c4.yaml

echo ""
echo "========== 7/9 CRP 8x (ours) =========="
python scripts/train_lora.py --config configs/crp_c8.yaml

echo ""
echo "========== 8/9 CRP Merge 4x (ours) =========="
python scripts/train_lora.py --config configs/crp_merge_c4.yaml

# Direction 2.5: Knowledge Distillation
echo ""
echo "========== 9/9 Distillation 7B→3B =========="
python scripts/train_distill.py --config configs/distill_7b_3b.yaml

echo ""
echo "=========================================="
echo "  All experiments done! $(date)"
echo "=========================================="
