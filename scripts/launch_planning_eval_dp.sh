#!/usr/bin/env bash
# Launch scripts/planning_eval.py under torchrun with data-parallel sharding.
#
# Usage:
#   bash scripts/launch_planning_eval_dp.sh <ckpt_dir> [extra args ...]
#
# Env knobs:
#   NPROC          - number of GPUs (default: 8)
#   BATCH_SIZE     - per-rank batch size (default: 4)
#   CUDA_VISIBLE_DEVICES - restrict GPUs (e.g. "1,2,3,4,5,6,7" to avoid GPU 0
#                   if another eval is running). NPROC must match the count.
#
# Example:
#   CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 NPROC=7 \
#     bash scripts/launch_planning_eval_dp.sh \
#       checkpoints_qwen25/nuscenes_planning_3b_full_sft/final \
#       --infos-val data/uniad_infos/nuscenes_infos_temporal_val.pkl \
#       --nusc-root data/nuscenes \
#       --output /tmp/eval_results_dp.json

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <ckpt_dir> [extra planning_eval args]" >&2
  exit 2
fi

CKPT="$1"; shift

NPROC="${NPROC:-8}"
BATCH_SIZE="${BATCH_SIZE:-4}"

# Pick a free TCP port so multiple eval runs don't collide.
MASTER_PORT="${MASTER_PORT:-$(/usr/bin/python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[launch_planning_eval_dp] NPROC=${NPROC} BATCH_SIZE=${BATCH_SIZE} MASTER_PORT=${MASTER_PORT}"
echo "[launch_planning_eval_dp] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[launch_planning_eval_dp] ckpt=${CKPT}"

TORCHRUN="${TORCHRUN:-$(command -v torchrun || true)}"
if [[ -z "${TORCHRUN}" ]]; then
  TORCHRUN="/usr/bin/python3 -m torch.distributed.run"
fi

exec ${TORCHRUN} \
  --standalone \
  --nproc_per_node="${NPROC}" \
  --master_port="${MASTER_PORT}" \
  "${SCRIPT_DIR}/planning_eval.py" \
  --ckpt "${CKPT}" \
  --batch-size "${BATCH_SIZE}" \
  "$@"
