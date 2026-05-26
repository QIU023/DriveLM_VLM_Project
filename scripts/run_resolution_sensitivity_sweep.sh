#!/usr/bin/env bash
# Resolution-sensitivity sweep on the FIXED B.5''' 3-cam AutoVLA-res ckpt.
#
# Question: the 3-cam retrain at 240 tok/cam scored L2 0.626, while the older
# 36-tok/cam thumbnail run scored 0.549. Is that gap (a) camera RESOLUTION, or
# (b) just a different training run (seed/data order)?
#
# We cannot rebuild the 2x2 (the thumbnail CKPT was overwritten), so we measure
# how much THIS ckpt's L2 depends on camera resolution at EVAL time. The ckpt's
# saved video cap is 524288 (240 tok/cam). We override it to several caps:
#
#   109760  -> [2,6,12]   -> 36  tok/cam  (== the old thumbnail resolution)
#   262144  -> ~[2,12,20] -> ~120 tok/cam
#   524288  -> [2,16,30]  -> 240 tok/cam  (== TRAINING res; sanity must ~= 0.626)
#   1048576 -> ~[2,22,42] -> ~460 tok/cam (above training)
#
# Reading:
#   * L2 ~flat across caps  -> this model barely uses camera resolution; the
#     0.549<->0.626 training gap is dominated by seed/run noise + ego/map priors
#     (the "ego status is all you need" regime). Higher pixels !-> better L2.
#   * L2 degrades sharply at 36 tok/cam -> camera resolution DOES matter at
#     inference (note: there is a train(240)/eval(36) MISMATCH caveat here, so
#     a drop at 36 partly reflects mismatch, not "low-res is intrinsically bad").
#
# Each eval is full 5119-val, DP-8. ~7 min/cap. Output JSON per cap.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export HF_HOME=/workspace/.hf_home; unset HF_HUB_OFFLINE || true

CKPT=checkpoints_qwen25/nusc_planning_b5ppp_3cam_qwen3vl_multimodal/final
OUTDIR=docs/eval_results/resolution_sweep
mkdir -p "$OUTDIR"

for CAP in 109760 262144 524288 1048576; do
  echo "===== RES SWEEP cap=${CAP} ($(date +%H:%M)) ====="
  NPROC=8 BATCH_SIZE=4 bash scripts/launch_planning_eval_dp.sh \
    "$CKPT" \
    --infos-val data/uniad_infos/nuscenes_infos_temporal_val.pkl \
    --nusc-root data/nuscenes \
    --planning-cams CAM_FRONT,CAM_FRONT_LEFT,CAM_FRONT_RIGHT \
    --multimodal \
    --hdmap-dir data/preproc/hdmap_bev \
    --bbox-jsonl 'data/preproc/bbox_egostate_{split}.jsonl' \
    --num-past-frames 4 \
    --eval-max-length 12288 \
    --video-max-pixels "${CAP}" \
    --output "${OUTDIR}/b5ppp_3cam_rescap_${CAP}.json"
  echo "===== done cap=${CAP} ($(date +%H:%M)) ====="
done
echo "ALL_RES_SWEEP_DONE"
