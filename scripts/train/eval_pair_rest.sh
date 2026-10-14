#!/usr/bin/env bash
# 公平对照(续): dcn 最优 ckpt(ep47) 与 cpoff best, 在剩余 8 档 night 上用同一条 eval 路径
set -u
cd /home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
ulimit -n 65535
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2:/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
PY=/home/constant/miniconda3/envs/deimv2/bin/python
W=/home/constant/CodeBuddy/Dev/AIC/logs
O=./outputs
D=$O/deim_hgnetv2_x_aic_obj2coco_ft_dcn/best_stg1_ep47_AP5613.pth
C=$O/deim_hgnetv2_x_aic_obj2coco_ft_cpoff/best_stg2.pth

echo "===== PAIR EVAL REST START $(date) =====" >> "$W/dfine_x_pair_eval.log"
for lv in night50 night25 night05 night005 night01 night002 night001 night0005; do
  $PY train.py -c configs/deimv2/deim_hgnetv2_x_aic_obj2coco_ft_dcn_val_${lv}.yml \
      -r "$D" --use-amp --test-only > "$W/pair_dcn47_${lv}.log" 2>&1
  echo "rc=$? dcn47_$lv" >> "$W/dfine_x_pair_eval.log"
  $PY train.py -c configs/deimv2/deim_hgnetv2_x_aic_obj2coco_ft_cpoff_val_${lv}.yml \
      -r "$C" --use-amp --test-only > "$W/pair_cpoff_${lv}.log" 2>&1
  echo "rc=$? cpoff_$lv" >> "$W/dfine_x_pair_eval.log"
done
echo "===== PAIR EVAL REST DONE $(date) =====" >> "$W/dfine_x_pair_eval.log"
