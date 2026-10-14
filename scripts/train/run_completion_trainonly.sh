#!/usr/bin/env bash
# Resume the depth-completion RGB-D ablation: skip prep/gen (already done),
# run train (68ep) + eval (orig, night005). Logs use absolute paths.
set -e
cd /home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
PY=/home/constant/miniconda3/envs/deimv2/bin/python
LOG=/home/constant/CodeBuddy/Dev/AIC/work_dir

echo "[3/4] train rgd_completion 68ep"
$PY train.py -c configs/deimv2/deimv2_dinov3_l_aic_rgd_completion.yml --use-amp \
    > "$LOG/completion_train.log" 2>&1

CK=outputs/deimv2_dinov3_l_aic_rgd_completion/best_stg2.pth
echo "[4/4] eval orig + night005"
$PY train.py -c configs/deimv2/deimv2_dinov3_l_aic_rgd_completion__orig.yml \
    -r "$CK" --use-amp --test-only >> "$LOG/completion_eval.log" 2>&1
$PY train.py -c configs/deimv2/deimv2_dinov3_l_aic_rgd_completion__night005.yml \
    -r "$CK" --use-amp --test-only >> "$LOG/completion_eval.log" 2>&1

echo "COMPLETION_DONE" >> "$LOG/completion_pipeline.log"
