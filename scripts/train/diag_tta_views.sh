#!/bin/bash
# Diagnostic round 3: sanity-check the WBF merge on orig before the full sweep.
# Reference: official no-TTA evaluate on orig = 0.503 ; harness 800/no-flip = 0.4986
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2

PY=/home/constant/miniconda3/envs/deimv2/bin/python
S=/home/constant/CodeBuddy/Dev/AIC/scripts/tta_eval.py
C=configs/deimv2/deimv2_dinov3_l_aic_vitl800_rdttrack_deepir_aug_val_orig.yml
R=outputs/deimv2_dinov3_l_aic_vitl800_rdttrack_deepir_aug/best_stg1.pth

run() {
  echo "### $* $(date +%H:%M:%S)"
  $PY "$S" -c "$C" -r "$R" --use-amp "$@" 2>&1 \
    | grep -E 'RESULT_AP|Traceback|Error' || echo "(no result)"
}

run --scales 640,800 --flip
echo "DIAG3 DONE $(date +%H:%M:%S)"
