#!/usr/bin/env bash
# 公平对照: 同一条 eval 路径, 双方各自 best ckpt, 跑 day(orig) 档
set -u
cd /home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
ulimit -n 65535
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2:/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
PY=/home/constant/miniconda3/envs/deimv2/bin/python
W=/home/constant/CodeBuddy/Dev/AIC/logs
O=./outputs

run () {  # $1=tag $2=val_yml $3=ckpt $4=outlog
  $PY train.py -c "$2" -r "$3" --use-amp --test-only > "$W/$4" 2>&1
  echo "rc=$? $1 ($3)" >> "$W/dfine_x_pair_eval.log"
}

echo "===== PAIR EVAL START $(date) =====" >> "$W/dfine_x_pair_eval.log"
run dcn_best47 configs/deimv2/deim_hgnetv2_x_aic_obj2coco_ft_dcn_val_orig.yml \
    $O/deim_hgnetv2_x_aic_obj2coco_ft_dcn/best_stg1_ep47_AP5613.pth pair_dcn47_orig.log
run cpoff_best configs/deimv2/deim_hgnetv2_x_aic_obj2coco_ft_cpoff_val_orig.yml \
    $O/deim_hgnetv2_x_aic_obj2coco_ft_cpoff/best_stg2.pth pair_cpoff_orig.log
run dcn_best47_n12 configs/deimv2/deim_hgnetv2_x_aic_obj2coco_ft_dcn_val_night12.yml \
    $O/deim_hgnetv2_x_aic_obj2coco_ft_dcn/best_stg1_ep47_AP5613.pth pair_dcn47_night12.log
run cpoff_best_n12 configs/deimv2/deim_hgnetv2_x_aic_obj2coco_ft_cpoff_val_night12.yml \
    $O/deim_hgnetv2_x_aic_obj2coco_ft_cpoff/best_stg2.pth pair_cpoff_night12.log
echo "===== PAIR EVAL DONE $(date) =====" >> "$W/dfine_x_pair_eval.log"
