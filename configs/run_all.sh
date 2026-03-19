#!/bin/bash
# Run all Layer 2 experiments on GH200
# Usage: bash configs/run_all.sh

SCRIPT="python scripts/train_lora.py --epochs 3 --val-every 200 --val-batches 50"

echo "========== 1/5 Baseline (no compression) =========="
$SCRIPT --compress-method none

echo "========== 2/5 Avg Pool 4x =========="
$SCRIPT --compress-method avg_pool --compress-ratio 4

echo "========== 3/5 FasterVLM 4x =========="
$SCRIPT --compress-method fastervlm --compress-ratio 4

echo "========== 4/5 PruMerge 4x =========="
$SCRIPT --compress-method prumerge --compress-ratio 4

echo "========== 5/5 PyramidDrop 4x =========="
$SCRIPT --compress-method pyramiddrop --compress-ratio 4

echo "All experiments done!"
