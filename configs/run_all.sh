#!/bin/bash
# Run all Layer 2 visual token compression experiments on GH200
# Usage: bash configs/run_all.sh

set -e

echo "========== 1/5 Baseline (no compression) =========="
python scripts/train_lora.py --config configs/baseline.yaml

echo "========== 2/5 Avg Pool 4x =========="
python scripts/train_lora.py --config configs/avg_pool_c4.yaml

echo "========== 3/5 FasterVLM 4x =========="
python scripts/train_lora.py --config configs/fastervlm_c4.yaml

echo "========== 4/5 PruMerge 4x =========="
python scripts/train_lora.py --config configs/prumerge_c4.yaml

echo "========== 5/5 PyramidDrop 4x =========="
python scripts/train_lora.py --config configs/pyramiddrop_c4.yaml

echo "All experiments done!"
