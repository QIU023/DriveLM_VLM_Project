#!/bin/bash
# Sequential full-val evaluation for baseline + fastervlm
# Usage: bash results/run_eval_all.sh

set -e
export PATH="/home/ubuntu/miniconda3/bin:$PATH"
source activate qwen25vl 2>/dev/null

echo "=== Waiting for baseline eval (PID 375158) to finish ==="
while kill -0 375158 2>/dev/null; do sleep 60; done
echo "=== Baseline done ==="

echo ""
echo "=== Starting FasterVLM full eval ==="
python -u scripts/eval_full.py \
    --config configs/fastervlm_c4.yaml \
    --lora checkpoints_qwen25/fastervlm_c4/final \
    --output results/eval_fastervlm_c4_full.json \
    --save-every 200 \
    2>&1 | tee results/eval_fastervlm_c4_full.log

echo "=== All evaluations complete ==="
