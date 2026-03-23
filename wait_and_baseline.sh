#!/bin/bash
# Wait for CRP experiments to finish, then retrain baseline with gb200 config (bs=16)
cd /root/DriveLM_VLM_Project

echo "Waiting for CRP experiments to finish (checking crp_merge_c4/final)..."
while [ ! -d "checkpoints_qwen25/crp_merge_c4/final" ]; do
    echo -n "."
    sleep 120
done

echo ""
echo "CRP done. Starting baseline retrain (bs=16, gb200 config)..."
nohup /venv/main/bin/python -u scripts/train_lora.py --config configs/baseline.yaml 2>&1 | tee logs/baseline_b200.log
echo "Baseline retrain done."
