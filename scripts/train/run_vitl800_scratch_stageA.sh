#!/bin/bash
# =============================================================================
# Stage A / RGB-only detector —— 一次性训练三阶段流程的第一步
#
# 起点: ckpts/dinov3/vitl16_rdttrack_scratch_full.pth
#   rgb   = 官方 DINOv3 ViT-L/16 预训练 (303.2M, 3ch 原样)
#   ir    = 官方 DINOv3 ViT-S+/16 预训练 (28.6M, patch_embed 3->2ch mean-repeat)
#   depth = 官方 DINOv3 ViT-S+/16 预训练 (28.5M, patch_embed 3->1ch mean-repeat)
#   encoder / decoder / convs / aux_proj / sta / prompt / dt_ort = 无此键 -> 随机
# 不使用任何在 AIC 上训练过的权重。
#
# Stage A 为什么等价于"纯 RGB 训练"(零模型改动):
#   forward 里 `ir_feats = [torch.zeros_like(f) for f in ir_feats]`,
#   zeros_like 造的是全新张量 -> 与 aux 主干计算图断开 -> aux 收不到梯度,
#   网络此时只能依赖 RGB。同时 ir/depth_drop_prob=1.0 保证丢弃恒发生。
# =============================================================================
set -u
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2

PY=/home/constant/miniconda3/envs/deimv2/bin/python
W=/home/constant/CodeBuddy/Dev/AIC/logs
CFG=configs/deimv2/deimv2_dinov3_l_aic_vitl800_scratch_stageA_rgb.yml
SRC=ckpts/dinov3/vitl16_rdttrack_scratch_full.pth
OUT=outputs/vitl800_scratch_stageA_rgb
# Stage A 必须用专属 val 配置 (ir/depth_drop_prob=1.0): 通用 val 配置是 0.15,
# eval 时 _zero_stream(training=False, 0.15) 恒 False -> 真实 IR 特征会撞进
# running_var≈0 的 BatchNorm2d -> 数值爆炸, 10 档评测全是垃圾。
PREF=configs/deimv2/deimv2_dinov3_l_aic_vitl800_scratch_stageA_rgb_val_
LV="day night50 night25 night12 night05 night01 night005 night002 night001 night0005"

echo "===== START STAGE_A $(date) =====" >> "$W/stageA_rgb_run.log"

# 排队: 等 GPU 可用 >= 12GB
while true; do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
  FREE=$((TOTAL - USED))
  if [ "$FREE" -ge 12000 ]; then break; fi
  echo "queued: free=${FREE}MiB $(date)" >> "$W/stageA_rgb_run.log"
  sleep 60
done
echo "GPU free, launching train $(date)" >> "$W/stageA_rgb_run.log"

if [ -f "$OUT/last.pth" ]; then
  RESUME_ARG="-r $OUT/last.pth"
  echo "RESUMING from $OUT/last.pth ($(date))" >> "$W/stageA_rgb_run.log"
else
  RESUME_ARG="-t $SRC"
  echo "FRESH from clean official weights $SRC ($(date))" >> "$W/stageA_rgb_run.log"
fi

$PY train.py -c "$CFG" $RESUME_ARG --use-amp > "$W/stageA_rgb_train.log" 2>&1
TRAIN_RC=$?
echo "TRAIN DONE exit=$TRAIN_RC $(date)" >> "$W/stageA_rgb_run.log"
if [ "$TRAIN_RC" -ne 0 ]; then
  echo "TRAIN FAILED rc=$TRAIN_RC (重跑本脚本会从 last.pth 续训)" >> "$W/stageA_rgb_run.log"
  exit "$TRAIN_RC"
fi

CKPT=$(ls -t "$OUT"/*.pth 2>/dev/null | head -1)
echo "STAGE_A CKPT=$CKPT" >> "$W/stageA_rgb_run.log"
if [ -z "$CKPT" ]; then echo "NO CKPT, abort eval"; exit 1; fi

for lv in $LV; do
  cfg=${PREF}${lv}.yml; [ "$lv" = "day" ] && cfg=${PREF}orig.yml
  echo "=== stageA eval $lv $(date) ===" >> "$W/stageA_rgb_run.log"
  $PY train.py -c "$cfg" -r "$CKPT" --use-amp --test-only > "$W/stageA_rgb_eval_${lv}.log" 2>&1
done
echo "ALL DONE STAGE_A $(date)" >> "$W/stageA_rgb_run.log"
