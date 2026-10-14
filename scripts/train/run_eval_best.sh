#!/bin/bash
# 杠杆④ 公平对照: 两模型均用 best_stg2.pth (ufall 的 0.2111 来自此 ckpt)。
# 5 协议 × 2 模型, 提取 bbox AP@0.50:0.95 汇总。
set -u
cd /home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
PY=/home/constant/miniconda3/envs/deimv2/bin/python
OD=/home/constant/CodeBuddy/Dev/AIC/work_dir
CFGD=configs/deimv2
SUMMARY=$OD/logs_text/eval_best_summary.txt
: > "$SUMMARY"

ufall_ckpt=outputs/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall/best_stg2.pth
envgate_ckpt=outputs/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_envgate/best_stg2.pth

run() {
  local tag=$1 cfg=$2 ckpt=$3
  local log=$OD/evalbest_${tag}.log
  echo "===== $tag =====" | tee -a "$SUMMARY"
  $PY train.py -c "$CFGD/$cfg" --test-only -r "$ckpt" > "$log" 2>&1
  local ap
  ap=$(grep -E "IoU=0.50:0.95 \| area=   all \| maxDets=100" "$log" | head -1 | grep -oE "= [0-9.]+" | grep -oE "[0-9.]+")
  echo "$tag  AP=$ap" | tee -a "$SUMMARY"
}

run ufall_real        deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_val_night005.yml          "$ufall_ckpt"
run ufall_smoke       deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_val_night005_smoke.yml       "$ufall_ckpt"
run ufall_abbothblk   deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_val_night005_abbothblk.yml   "$ufall_ckpt"
run ufall_abirblk     deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_val_night005_abirblk.yml       "$ufall_ckpt"
run ufall_abdepblk    deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_val_night005_abdepblk.yml      "$ufall_ckpt"

run envgate_real       deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_envgate_val_night005.yml          "$envgate_ckpt"
run envgate_smoke      deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_envgate_val_night005_smoke.yml       "$envgate_ckpt"
run envgate_abbothblk  deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_envgate_val_night005_abbothblk.yml   "$envgate_ckpt"
run envgate_abirblk    deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_envgate_val_night005_abirblk.yml       "$envgate_ckpt"
run envgate_abdepblk   deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_envgate_val_night005_abdepblk.yml      "$envgate_ckpt"

echo "DONE" | tee -a "$SUMMARY"
