#!/bin/bash
# ViT-B @800 + 增强: RGB 主干升级到 DINOv3 ViT-B/16(768-dim), 分辨率 640->800,
# 增强加几何扰动 + 暗光退化。基线是 ..._ufall_x_deepir_midboost_l4u3 (AP=0.5372)。
#
# 前置(只需做一次, 已完成):
#   1) 下载官方 ViT-B 权重 -> vendor/DEIMv2/ckpts/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
#   2) 在 vision_transformer.py 注册 'dinov3_vitb16'
#   3) 生成热启动权重:
#        cd vendor/DEIMv2 && python prepare_dinov3_for_rdttrack.py \
#            --dinov3 ckpts/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
#            --out    ckpts/dinov3/vitb16_rdttrack_init.pth \
#            --inherit-aux     outputs/.../best_stg1.pth \
#            --inherit-encoder outputs/.../best_stg1.pth
#
# 显存兜底(若 OOM, 按顺序试):
#   a) accum_steps 10 + total_batch_size 2  (micro=1)
#   b) 配置里 DINOv3STAsRDTTrack 加 use_modality_stem: True (辅助流降为 3-conv)
set -u
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2

PY=/home/constant/miniconda3/envs/deimv2/bin/python
W=/home/constant/CodeBuddy/Dev/AIC/logs
PREF=configs/deimv2/deimv2_dinov3_l_aic_vitb800_rdttrack_deepir_aug_val_
CFG=configs/deimv2/deimv2_dinov3_l_aic_vitb800_rdttrack_deepir_aug.yml
SRC=ckpts/dinov3/vitb16_rdttrack_init.pth
OUT=outputs/deimv2_dinov3_l_aic_vitb800_rdttrack_deepir_aug
LV="day night50 night25 night12 night05 night01 night005 night002 night001 night0005"

echo "===== START VITB800 $(date) =====" >> "$W/vitb800_run.log"

# 排队: 等 GPU 空闲(可用 >= 12GB)。本实验比基线重得多, 阈值不要调低。
while true; do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
  FREE=$((TOTAL - USED))
  if [ "$FREE" -ge 12000 ]; then break; fi
  echo "queued: free=${FREE}MiB $(date)" >> "$W/vitb800_run.log"
  sleep 60
done
echo "GPU free, launching train $(date)" >> "$W/vitb800_run.log"

# 注意用 -t (tuning) 而不是 -r: -t 走 _matched_state, 按 shape 过滤,
# 新增/变维的模块(如 1x1 aux proj)会保持随机初始化而不会报错。
$PY train.py -c "$CFG" -t "$SRC" --use-amp > "$W/vitb800_train.log" 2>&1
echo "TRAIN DONE exit=$? $(date)" >> "$W/vitb800_run.log"

CKPT=$(ls -t "$OUT"/*.pth 2>/dev/null | head -1)
echo "VITB800 CKPT=$CKPT" >> "$W/vitb800_run.log"
if [ -z "$CKPT" ]; then echo "NO CKPT, abort eval"; exit 1; fi

for lv in $LV; do
  cfg=${PREF}${lv}.yml; [ "$lv" = "day" ] && cfg=${PREF}orig.yml
  echo "=== vitb800 eval $lv $(date) ===" >> "$W/vitb800_run.log"
  $PY train.py -c "$cfg" -r "$CKPT" --use-amp --test-only > "$W/vitb800_eval_${lv}.log" 2>&1
done
echo "EVAL DONE $(date)" >> "$W/vitb800_run.log"
echo "ALL DONE VITB800 $(date)" >> "$W/vitb800_run.log"
