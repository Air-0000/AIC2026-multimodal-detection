#!/usr/bin/env bash
# DEIM-D-FINE-X (HGNetv2-B5) AIC 单阶段微调(离线CP) + day/9档night 评测
# 贴图与标签已离线生成 (train_cpoff, build_cp_dataset.py), 配置无在线 CP 算子
# 配置: deimv2/deim_hgnetv2_x_aic_obj2coco_ft_cpoff.yml
set -u
cd /home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
# ancdata 崩溃对策(已 3 次): PyTorch fd 共享策略在 ulimit -n=1024 下传句柄失败
ulimit -n 65535
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2:/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/constant/miniconda3/envs/deimv2/bin/python
W=/home/constant/CodeBuddy/Dev/AIC/logs
CFG=configs/deimv2/deim_hgnetv2_x_aic_obj2coco_ft_cpoff.yml
OUT=./outputs/deim_hgnetv2_x_aic_obj2coco_ft_cpoff

echo "===== START DFINE_X_FT_CPOFF $(date) =====" >> "$W/dfine_x_cpoff_run.log"
$PY train.py -c "$CFG" -t ckpts/dfine_x_obj2coco.pth --use-amp > "$W/dfine_x_cpoff_train.log" 2>&1
TRC=$?
echo "train rc=$TRC" >> "$W/dfine_x_cpoff_run.log"
if [ "$TRC" -ne 0 ]; then
  echo "TRAIN FAILED (rc=$TRC), skip evals" >> "$W/dfine_x_cpoff_run.log"
  exit 1
fi

CKPT=$(ls -t "$OUT"/*.pth 2>/dev/null | head -1)
echo "DFINE_X CKPT=$CKPT" >> "$W/dfine_x_cpoff_run.log"
[ -z "$CKPT" ] && { echo "NO CKPT, abort eval"; exit 1; }

PREF=configs/deimv2/deim_hgnetv2_x_aic_obj2coco_ft_cpoff_val_
for lv in day night50 night25 night12 night05 night005 night01 night002 night001 night0005; do
  cfg=${PREF}${lv}.yml; [ "$lv" = "day" ] && cfg=${PREF}orig.yml
  echo "=== dfine_x eval $lv $(date) ===" >> "$W/dfine_x_cpoff_run.log"
  $PY train.py -c "$cfg" -r "$CKPT" --use-amp --test-only > "$W/dfine_x_cpoff_eval_${lv}.log" 2>&1
  echo "rc=$? $lv" >> "$W/dfine_x_cpoff_run.log"
done
echo "ALL DONE DFINE_X_FT_CPOFF $(date)" >> "$W/dfine_x_cpoff_run.log"
