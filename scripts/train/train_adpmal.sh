#!/usr/bin/env bash
# train_adpmal.sh - 启动 Class-Adaptive MAL 50-Epoch 定向补训
set -e
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2:$PYTHONPATH"
cd /home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2

echo "[train_adpmal] Starting Class-Adaptive MAL training on RTX 5060 Ti..."
/home/constant/miniconda3/envs/deimv2/bin/python train.py \
    -c configs/deimv2/deim_hgnetv2_x_aic_obj2coco_ft_adpmal.yml \
    -t ckpts/dfine_x_obj2coco.pth \
    --use-amp \
    --seed 0 \
    2>&1 | tee /home/constant/CodeBuddy/Dev/AIC/logs/dfine_x_adpmal_train.log
