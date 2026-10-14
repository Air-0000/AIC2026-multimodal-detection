#!/usr/bin/env bash
# 补跑 cpoff 的 day + 9 档 night 评测
# 首版 val yml 漏了 val_dataloader dataset transforms ops 覆写 => PIL 直进 collate 崩
# (rc=1, "Image object is not subscriptable"); 生成脚本已修并用本脚本补跑。
set -u
cd /home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
ulimit -n 65535
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2:/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
PY=/home/constant/miniconda3/envs/deimv2/bin/python
OUT=./outputs/deimv2_outputs_none  # placeholder, unused
CKPT=./outputs/deim_hgnetv2_x_aic_obj2coco_ft_cpoff/best_stg2.pth
W=/home/constant/CodeBuddy/Dev/AIC/logs
PREF=configs/deimv2/deim_hgnetv2_x_aic_obj2coco_ft_cpoff_val_

echo "===== EVAL-REDO START $(date) =====" >> "$W/dfine_x_cpoff_run.log"
for lv in orig night50 night25 night12 night05 night005 night01 night002 night001 night0005; do
  cfg=${PREF}${lv}.yml
  $PY train.py -c "$cfg" -r "$CKPT" --use-amp --test-only > "$W/dfine_x_cpoff_eval_${lv}.log" 2>&1
  echo "rc=$? $lv" >> "$W/dfine_x_cpoff_run.log"
done
echo "===== EVAL-REDO DONE $(date) =====" >> "$W/dfine_x_cpoff_run.log"
