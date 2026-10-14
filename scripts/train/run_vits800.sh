#!/bin/bash
# ViT-S @800 + 增强: vitl800 的受控 ViT-S 对照实验。
#
# 与 vitl800 的唯一变量 = backbone 规模:
#   vitl800: RGB 主干 dinov3_vitl16   (1024-dim, 24 层)
#   vits800: RGB 主干 dinov3_vits16plus ( 384-dim, 12 层)
# 其余 (800 分辨率 / 同光度+几何+暗光增强 / 解冻比例 25% / 同 LLRD / 同 batch) 完全照搬,
# 以便把 "换大 backbone" 隔离成单变量。
#
# 热启动: 继承 l4u3 的 best_stg1.pth (已是微调好的 ViT-S 检测器, 含 vits16plus 主干
#   + ir/depth + decoder)。-t 走 _matched_state 全量继承, 起点 = 已训 ViT-S 检测器。
# 这与 vitl800 (官方 ViT-L 起点 + best_stg1 的 head) 构成对称对照:
#   两者同设置训 50 epoch, 唯一变量是 backbone 容量。
#
# 对比含义: 同设置下 ViT-S(从已训检测器起点) vs ViT-L(从官方预训练起点) 各自能到哪,
#   最终 mAP + 11 档暗光评测直接比 backbone 容量带来的增益。
#
# 注意: ViT-S 是 12 层, interaction_indexes=[5,8,11] / unfreeze_rgb_stages=3
#   (按 24 层 [11,17,23]/6 等比缩放, 保证解冻比例同为 25%)。
set -u
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2

PY=/home/constant/miniconda3/envs/deimv2/bin/python
W=/home/constant/CodeBuddy/Dev/AIC/logs
PREF=configs/deimv2/deimv2_dinov3_l_aic_vits800_rdttrack_deepir_aug_val_
CFG=configs/deimv2/deimv2_dinov3_l_aic_vits800_rdttrack_deepir_aug.yml
SRC=/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/outputs/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir_midboost_l4u3/best_stg1.pth
OUT=outputs/deimv2_dinov3_l_aic_vits800_rdttrack_deepir_aug
LV="day night50 night25 night12 night05 night01 night005 night002 night001 night0005"

echo "===== START VITS800 $(date) =====" >> "$W/vits800_run.log"

# 排队: 严格等 vitl800 完全释放(含 11 档 eval)后再接力, 避免两实验争抢 GPU。
#   阈值设 14GB: vitl800 训练+eval 全程占用, 只有它全部结束 free 才回到 ~14GiB。
#   vits800 本身显存小(micro=3 余量充足), 不需要这么多空闲, 高阈值只为"等前序结束"。
while true; do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
  FREE=$((TOTAL - USED))
  if [ "$FREE" -ge 14000 ]; then break; fi
  echo "queued: free=${FREE}MiB $(date)" >> "$W/vits800_run.log"
  sleep 60
done
echo "GPU free, launching train $(date)" >> "$W/vits800_run.log"

# 优先断点续训: 若上次崩在中间(留下 last.pth), 用 -r 接着跑; 否则 -t 热启动。
if [ -f "$OUT/last.pth" ]; then
  RESUME_ARG="-r $OUT/last.pth"
  echo "RESUMING from $OUT/last.pth ($(date))" >> "$W/vits800_run.log"
else
  RESUME_ARG="-t $SRC"
  echo "FRESH tuning from $SRC ($(date))" >> "$W/vits800_run.log"
fi
$PY train.py -c "$CFG" $RESUME_ARG --use-amp > "$W/vits800_train.log" 2>&1
TRAIN_RC=$?
echo "TRAIN DONE exit=$TRAIN_RC $(date)" >> "$W/vits800_run.log"
if [ "$TRAIN_RC" -ne 0 ]; then
  echo "TRAIN FAILED rc=$TRAIN_RC —— 不跑 eval (重跑本脚本会从 last.pth 续训)" >> "$W/vits800_run.log"
  exit "$TRAIN_RC"
fi

CKPT=$(ls -t "$OUT"/*.pth 2>/dev/null | head -1)
echo "VITS800 CKPT=$CKPT" >> "$W/vits800_run.log"
if [ -z "$CKPT" ]; then echo "NO CKPT, abort eval"; exit 1; fi

for lv in $LV; do
  cfg=${PREF}${lv}.yml; [ "$lv" = "day" ] && cfg=${PREF}orig.yml
  echo "=== vits800 eval $lv $(date) ===" >> "$W/vits800_run.log"
  $PY train.py -c "$cfg" -r "$CKPT" --use-amp --test-only > "$W/vits800_eval_${lv}.log" 2>&1
done
echo "EVAL DONE $(date)" >> "$W/vits800_run.log"
echo "ALL DONE VITS800 $(date)" >> "$W/vits800_run.log"
