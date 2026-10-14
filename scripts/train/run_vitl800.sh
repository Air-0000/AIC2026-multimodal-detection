#!/bin/bash
# ViT-L @800 + 增强: RGB 主干升级到 DINOv3 ViT-L/16(1024-dim, 24 层), 分辨率 640->800,
# 增强加几何扰动 + 暗光退化。基线是 ..._ufall_x_deepir_midboost_l4u3 (AP=0.5372)。
# 与 vitb800 实验的唯一差别是 backbone 规模(768 -> 1024), 其余保持一致以便归因。
#
# 前置(只需做一次, 已完成):
#   1) 下载官方 ViT-L 权重 -> vendor/DEIMv2/ckpts/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
#      (官方 facebook 仓 gated/401, 走镜像; 已 sha256 校验与官方逐字节一致)
#   2) 在 vision_transformer.py 注册 'dinov3_vitl16'
#   3) 生成热启动权重:
#        cd vendor/DEIMv2 && python prepare_dinov3_for_rdttrack.py \
#            --dinov3 ckpts/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
#            --out    ckpts/dinov3/vitl16_rdttrack_init.pth \
#            --inherit-aux     outputs/.../best_stg1.pth \
#            --inherit-encoder outputs/.../best_stg1.pth
#
# 注意: ViT-L 是 24 层, interaction_indexes=[11,17,23] / unfreeze_rgb_stages=6
#       是按 12 层 ViT 的 [5,8,11]/3 等比放大来的, 已写进配置。
#
# 显存兜底(若 OOM, 按顺序试; 当前 micro=3, 峰值 ~11.3GiB, 余量仅 ~2.9GiB,
#   已逼近上限 —— 降峰值的唯一旋钮是减 micro):
#   a) accum_steps 调大(micro 减半) —— 唯一能降峰值的办法
#   b) 配置里 DINOv3STAsRDTTrack 加 use_modality_stem: True (辅助流降为 3-conv)
#   c) 分辨率降到 640 (python3 scripts/gen/gen_vit800_configs.py --backbone vitl16 --size 640)
#   d) 退回 vitb800
set -u
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2

PY=/home/constant/miniconda3/envs/deimv2/bin/python
W=/home/constant/CodeBuddy/Dev/AIC/logs
PREF=configs/deimv2/deimv2_dinov3_l_aic_vitl800_rdttrack_deepir_aug_val_
CFG=configs/deimv2/deimv2_dinov3_l_aic_vitl800_rdttrack_deepir_aug.yml
SRC=ckpts/dinov3/vitl16_rdttrack_init.pth
OUT=outputs/deimv2_dinov3_l_aic_vitl800_rdttrack_deepir_aug
LV="day night50 night25 night12 night05 night01 night005 night002 night001 night0005"

echo "===== START VITL800 $(date) =====" >> "$W/vitl800_run.log"

# 排队: 等 GPU 空闲(可用 >= 12GB)。本实验比基线重得多, 阈值不要调低。
while true; do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
  FREE=$((TOTAL - USED))
  if [ "$FREE" -ge 12000 ]; then break; fi
  echo "queued: free=${FREE}MiB $(date)" >> "$W/vitl800_run.log"
  sleep 60
done
echo "GPU free, launching train $(date)" >> "$W/vitl800_run.log"

# 优先断点续训: 若上次训练崩在中间(留下 last.pth), 用 -r 接着跑; 否则 -t 热启动。
# -t (tuning) 走 _matched_state 按 shape 过滤, 变维模块(如 1x1 aux proj)保持随机初始化不报错;
# -r (resume) 严格恢复 optimizer/epoch, 用于崩后续训(模型结构须一致)。
if [ -f "$OUT/last.pth" ]; then
  RESUME_ARG="-r $OUT/last.pth"
  echo "RESUMING from $OUT/last.pth ($(date))" >> "$W/vitl800_run.log"
else
  RESUME_ARG="-t $SRC"
  echo "FRESH tuning from $SRC ($(date))" >> "$W/vitl800_run.log"
fi
$PY train.py -c "$CFG" $RESUME_ARG --use-amp > "$W/vitl800_train.log" 2>&1
TRAIN_RC=$?
echo "TRAIN DONE exit=$TRAIN_RC $(date)" >> "$W/vitl800_run.log"
if [ "$TRAIN_RC" -ne 0 ]; then
  echo "TRAIN FAILED rc=$TRAIN_RC —— 不跑 eval (模型不完整, 重跑本脚本会从 last.pth 续训)" >> "$W/vitl800_run.log"
  exit "$TRAIN_RC"
fi

CKPT=$(ls -t "$OUT"/*.pth 2>/dev/null | head -1)
echo "VITL800 CKPT=$CKPT" >> "$W/vitl800_run.log"
if [ -z "$CKPT" ]; then echo "NO CKPT, abort eval"; exit 1; fi

for lv in $LV; do
  cfg=${PREF}${lv}.yml; [ "$lv" = "day" ] && cfg=${PREF}orig.yml
  echo "=== vitl800 eval $lv $(date) ===" >> "$W/vitl800_run.log"
  $PY train.py -c "$cfg" -r "$CKPT" --use-amp --test-only > "$W/vitl800_eval_${lv}.log" 2>&1
done
echo "EVAL DONE $(date)" >> "$W/vitl800_run.log"
echo "ALL DONE VITL800 $(date)" >> "$W/vitl800_run.log"
