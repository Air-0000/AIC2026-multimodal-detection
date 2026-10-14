#!/usr/bin/env bash
# DCN 残差实验续训: 从 last.pth (ep38) 续到 ep49 + day/9档night 评测
set -u
cd /home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2:/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/constant/miniconda3/envs/deimv2/bin/python
W=/home/constant/CodeBuddy/Dev/AIC/logs
CFG=configs/deimv2/deim_hgnetv2_x_aic_obj2coco_ft_dcn.yml
OUT=./outputs/deim_hgnetv2_x_aic_obj2coco_ft_dcn

echo "===== RESUME DFINE_X_DCN $(date) =====" >> "$W/dfine_x_dcn_run.log"
$PY train.py -c "$CFG" -r "$OUT/last.pth" --use-amp >> "$W/dfine_x_dcn_train.log" 2>&1
TRC=$?
echo "train rc=$TRC" >> "$W/dfine_x_dcn_run.log"
[ "$TRC" -ne 0 ] && { echo "TRAIN FAILED (rc=$TRC), skip evals" >> "$W/dfine_x_dcn_run.log"; exit 1; }

# best 优先 (resume 会重置 best_stat, 训完后手动挑最优 ckpt 评测)
CKPT=$(ls "$OUT"/best_stg1*.pth 2>/dev/null | head -1)
[ -z "$CKPT" ] && CKPT=$(ls -t "$OUT"/*.pth 2>/dev/null | head -1)
echo "DFINE_X_DCN CKPT=$CKPT" >> "$W/dfine_x_dcn_run.log"

PREF=configs/deimv2/deim_hgnetv2_x_aic_obj2coco_ft_dcn_val_
for lv in day night50 night25 night12 night05 night005 night01 night002 night001 night0005; do
  cfg=${PREF}${lv}.yml; [ "$lv" = "day" ] && cfg=${PREF}orig.yml
  echo "=== dfine_x_dcn eval $lv $(date) ===" >> "$W/dfine_x_dcn_run.log"
  $PY train.py -c "$cfg" -r "$CKPT" --use-amp --test-only > "$W/dfine_x_dcn_eval_${lv}.log" 2>&1
  echo "rc=$? $lv" >> "$W/dfine_x_dcn_run.log"
done
echo "ALL DONE DFINE_X_DCN $(date)" >> "$W/dfine_x_dcn_run.log"
