#!/bin/bash
# 杠杆④ 评测矩阵: ufall(base) vs envgate, 5 协议, 均用各自 last.pth。
# 提取 bbox AP@0.50:0.95 (area=all, maxDets=100) 汇总到 eval_matrix_summary.txt
set -u
cd /home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
PY=/home/constant/miniconda3/envs/deimv2/bin/python
OD=/home/constant/CodeBuddy/Dev/AIC/work_dir
CFGD=configs/deimv2
SUMMARY=$OD/logs_text/eval_matrix_summary.txt
: > "$SUMMARY"

ufall_ckpt=outputs/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall/last.pth
envgate_ckpt=outputs/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_envgate/last.pth

run() {
  local tag=$1 cfg=$2 ckpt=$3
  local log=$OD/eval_${tag}.log
  echo "===== $tag =====" | tee -a "$SUMMARY"
  $PY train.py -c "$CFGD/$cfg" --test-only -r "$ckpt" > "$log" 2>&1
  local ap
  ap=$(grep -E "IoU=0.50:0.95 \| area=   all \| maxDets=100" "$log" | head -1 | grep -oE "= [0-9.]+" | grep -oE "[0-9.]+")
  echo "$tag  AP=$ap" | tee -a "$SUMMARY"
}

# ufall baselines (real 已知 0.2111, 这里补齐其余 4 个)
run ufall_smoke       deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_val_night005_smoke.yml       "$ufall_ckpt"
run ufall_abbothblk   deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_val_night005_abbothblk.yml   "$ufall_ckpt"
run ufall_abirblk     deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_val_night005_abirblk.yml       "$ufall_ckpt"
run ufall_abdepblk    deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_val_night005_abdepblk.yml      "$ufall_ckpt"

# envgate (real 已测=0.104, 补齐其余 4 个)
run envgate_smoke       deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_envgate_val_night005_smoke.yml       "$envgate_ckpt"
run envgate_abbothblk   deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_envgate_val_night005_abbothblk.yml   "$envgate_ckpt"
run envgate_abirblk     deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_envgate_val_night005_abirblk.yml       "$envgate_ckpt"
run envgate_abdepblk    deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_envgate_val_night005_abdepblk.yml      "$envgate_ckpt"

echo "DONE" | tee -a "$SUMMARY"
