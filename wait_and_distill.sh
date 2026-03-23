#!/bin/bash
# Wait for full precompute to finish, then run distillation
cd /root/DriveLM_VLM_Project

echo "Waiting for full precompute (need 4000+ images in crp_importance.pt)..."
while true; do
    count=$(/venv/main/bin/python -c "
import torch, sys
try:
    d = torch.load('precomputed/crp_importance.pt', weights_only=True)
    print(len(d))
except:
    print(0)
" 2>/dev/null)
    if [ "$count" -ge 4000 ] 2>/dev/null; then
        echo "Precompute ready! ($count images)"
        break
    fi
    echo -n "."
    sleep 60
done

echo ""
echo "Starting 7B->3B distillation..."
nohup /venv/main/bin/python -u scripts/train_distill.py --config configs/distill_7b_3b.yaml 2>&1 | tee logs/distill_7b_3b.log
