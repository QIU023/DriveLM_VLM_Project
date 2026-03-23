#!/bin/bash
# 蒸馏完成后，自动接 CRP 压缩实验 2x/4x/8x/16x
cd /root/DriveLM_VLM_Project

echo "=========================================="
echo "  启动蒸馏实验 7B→3B RRD"
echo "  $(date)"
echo "=========================================="
nohup /venv/main/bin/python -u scripts/train_distill.py --config configs/distill_7b_3b.yaml 2>&1 | tee logs/distill_7b_3b.log

echo ""
echo "=========================================="
echo "  蒸馏完成，开始 CRP 压缩实验"
echo "  $(date)"
echo "=========================================="

echo ""
echo "=== 1/4  CRP 2x ==="
/venv/main/bin/python -u scripts/train_lora.py --config configs/crp_c2.yaml 2>&1 | tee logs/crp_c2.log

echo ""
echo "=== 2/4  CRP 4x ==="
/venv/main/bin/python -u scripts/train_lora.py --config configs/crp_c4.yaml 2>&1 | tee logs/crp_c4.log

echo ""
echo "=== 3/4  CRP 8x ==="
/venv/main/bin/python -u scripts/train_lora.py --config configs/crp_c8.yaml 2>&1 | tee logs/crp_c8.log

echo ""
echo "=== 4/4  CRP 16x ==="
/venv/main/bin/python -u scripts/train_lora.py --config configs/crp_c16.yaml 2>&1 | tee logs/crp_c16.log

echo ""
echo "=========================================="
echo "  所有实验完成！$(date)"
echo "=========================================="
