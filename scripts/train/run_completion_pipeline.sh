#!/usr/bin/env bash
# Depth-completion RGB-D ablation pipeline (single image). Steps:
#   1) materialise completed depth maps (prep_depth_complete.py, GPU/DAv2)
#   2) generate rgd config for the `completion` variant
#   3) train DINOv3STAsRDT 4ch (68 ep)
#   4) eval on orig + night005
set -e
cd /home/constant/CodeBuddy/Dev/AIC
PY=/home/constant/miniconda3/envs/deimv2/bin/python

PYTHONIOENCODING=utf-8 $PY -m py_compile /home/constant/CodeBuddy/Dev/AIC/scripts/prep/prep_depth_complete.py
ln -sfn /home/constant/CodeBuddy/Dev/AIC/work_dir/depth_variants/completion \
       /home/constant/CodeBuddy/Dev/AIC/work_dir/depth_by_method/1_prep_variants/completion

echo "[1/4] prep completion depth maps (DAv2 on IR)"
$PY /home/constant/CodeBuddy/Dev/AIC/scripts/prep/prep_depth_complete.py --splits train test > /home/constant/CodeBuddy/Dev/AIC/logs/completion_prep.log 2>&1

echo "[2/4] verify rgd config (generator gen_rgd_configs.py is gone; configs are checked in)"
RGD_CFG=/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/configs/deimv2/deimv2_dinov3_l_aic_rgd_completion.yml
if [ ! -f "$RGD_CFG" ]; then
  echo "ERROR: missing $RGD_CFG"
  echo "       create it manually, or restore scripts/gen/gen_rgd_configs.py"
  exit 1
fi

cd /home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
echo "[3/4] train rgd_completion 68ep"
$PY train.py -c configs/deimv2/deimv2_dinov3_l_aic_rgd_completion.yml --use-amp \
    > ../work_dir/completion_train.log 2>&1

CK=outputs/deimv2_dinov3_l_aic_rgd_completion/best_stg2.pth
echo "[4/4] eval orig + night005"
$PY train.py -c configs/deimv2/deimv2_dinov3_l_aic_rgd_completion_val_orig.yml \
    -r "$CK" --use-amp --test-only >> /home/constant/CodeBuddy/Dev/AIC/logs/completion_eval.log 2>&1
$PY train.py -c configs/deimv2/deimv2_dinov3_l_aic_rgd_completion_val_night005.yml \
    -r "$CK" --use-amp --test-only >> /home/constant/CodeBuddy/Dev/AIC/logs/completion_eval.log 2>&1

echo "COMPLETION_DONE" >> /home/constant/CodeBuddy/Dev/AIC/logs/completion_pipeline.log
