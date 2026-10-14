#!/usr/bin/env bash
# G=8 (DCNv3-style) 重跑种子实验: 唯一差异 --seed 1 (原 G=8 用默认 seed 0)
# 目的: 验证 v3 早期 +12.5pt@ep2 跳变是真实效应还是种子运气; 计划只跑前 12 epoch 后人工裁决
set -u
cd /home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2:/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/constant/miniconda3/envs/deimv2/bin/python
W=/home/constant/CodeBuddy/Dev/AIC/logs
CFG=configs/deimv2/deim_hgnetv2_x_aic_obj2coco_ft_dcnv3_s1.yml

echo "===== START G8_SEED1 $(date) =====" >> "$W/dfine_x_dcnv3_g8s1_run.log"
$PY train.py -c "$CFG" -t ckpts/dfine_x_obj2coco.pth --use-amp --seed 1 > "$W/dfine_x_dcnv3_g8s1_train.log" 2>&1
TRC=$?
echo "train rc=$TRC" >> "$W/dfine_x_dcnv3_g8s1_run.log"
echo "===== DONE G8_SEED1 $(date) =====" >> "$W/dfine_x_dcnv3_g8s1_run.log"
