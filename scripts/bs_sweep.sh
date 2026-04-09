#!/bin/bash
# Sweep batch_size for teacher_32b mini run with grad_checkpointing
# Reports wall-clock time + OOM count for each
set -u
cd /root/DriveLM_VLM_Project
source /venv/main/bin/activate

for bs in 2 4 8; do
  ga=$((8 / bs))
  log="logs/teacher_32b_mini_bs${bs}_ckpt.log"
  echo "================================================================"
  echo "  bs=${bs}  grad_accum=${ga}  grad_ckpt=true"
  echo "================================================================"
  start=$(date +%s)
  python -u scripts/train_lora.py --config configs/teacher_32b.yaml --mini --bs $bs 2>&1 | tee "$log"
  end=$(date +%s)
  elapsed=$((end - start))
  oom=$(grep -c "OOM" "$log" || echo 0)
  echo ""
  echo ">>> bs=${bs}: ${elapsed}s, OOMs=${oom}"
  echo ""
done

echo "================================================================"
echo "  Summary"
echo "================================================================"
for bs in 2 4 8; do
  log="logs/teacher_32b_mini_bs${bs}_ckpt.log"
  oom=$(grep -c "OOM" "$log" 2>/dev/null || echo 0)
  echo "bs=${bs}: OOMs=${oom}"
done
