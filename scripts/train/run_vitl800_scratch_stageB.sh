#!/bin/bash
# =============================================================================
# Stage B / 模态注入 (RGB 全冻结) —— 三阶段流程第二步
#
# 起点: 由 scripts/gen/prepare_stageB_init.py 从 Stage A 产物生成
#   ckpts/dinov3/stageB_from_stageA.pth
#     - 保留: RGB 主干 / IR 主干 / Depth 主干 / encoder / decoder / sta / convs
#             (即 Stage A 训出来的"纯 RGB 检测器")
#     - 重置: ir_norms, depth_norms, ir_aux_proj, depth_aux_proj, dt_ort, prompt
#             (Stage A 里这些是在零输入上训的, 权重是废的, 见脚本顶部注释)
#
# 本阶段机制:
#   unfreeze_rgb_stages: 0 -> rgb_dinov3.requires_grad_(False), RGB 主干完全冻结。
#   aux 两条流代码里本来就没有冻结过 (只有 rgb_dinov3 被 requires_grad_(False)),
#   所以 IR/Depth + 注入通路天然可训练, 无需额外改动。
#   ir/depth_drop_prob = 0.15 -> 随机整条置零, 逼模型在缺模态时也能靠 RGB 兜底。
# =============================================================================
set -u
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2

PY=/home/constant/miniconda3/envs/deimv2/bin/python
W=/home/constant/CodeBuddy/Dev/AIC/logs
R=/home/constant/CodeBuddy/Dev/AIC
CFG=configs/deimv2/deimv2_dinov3_l_aic_vitl800_scratch_stageB_mod.yml
OUT=outputs/vitl800_scratch_stageB_mod
STAGE_A_OUT=outputs/vitl800_scratch_stageA_rgb
INIT=ckpts/dinov3/stageB_from_stageA.pth
# Stage B/C 用通用 val 配置: eval 时 _zero_stream(training=False, 0.15) 恒 False,
# aux 走真实特征, 与训练期 BN running 统计一致。
PREF=configs/deimv2/deimv2_dinov3_l_aic_vitl800_rdttrack_deepir_aug_val_
LV="day night50 night25 night12 night05 night01 night005 night002 night001 night0005"

echo "===== START STAGE_B $(date) =====" >> "$W/stageB_mod_run.log"

# 排队: 等 GPU 可用 >= 12GB
while true; do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
  FREE=$((TOTAL - USED))
  if [ "$FREE" -ge 12000 ]; then break; fi
  echo "queued: free=${FREE}MiB $(date)" >> "$W/stageB_mod_run.log"
  sleep 60
done
echo "GPU free, launching train $(date)" >> "$W/stageB_mod_run.log"

if [ -f "$OUT/last.pth" ]; then
  RESUME_ARG="-r $OUT/last.pth"
  echo "RESUMING from $OUT/last.pth ($(date))" >> "$W/stageB_mod_run.log"
else
  # 首次起训: 先由 Stage A 产物生成起点 (重置融合通路)
  if [ ! -f "$INIT" ]; then
    echo "building Stage B init from Stage A ($(date))" >> "$W/stageB_mod_run.log"
    SRC_A=$(ls -t "$STAGE_A_OUT"/last.pth "$STAGE_A_OUT"/best_stg1.pth 2>/dev/null | head -1)
    if [ -z "$SRC_A" ]; then
      echo "NO STAGE_A CKPT, abort" | tee -a "$W/stageB_mod_run.log"
      exit 1
    fi
    echo "  Stage A source: $SRC_A" >> "$W/stageB_mod_run.log"
    $PY "$R/scripts/gen/prepare_stageB_init.py" --src "$SRC_A" --dst "$INIT" \
      >> "$W/stageB_mod_run.log" 2>&1
    if [ $? -ne 0 ]; then
      echo "prepare_stageB_init FAILED" | tee -a "$W/stageB_mod_run.log"
      exit 1
    fi
  fi
  RESUME_ARG="-t $INIT"
  echo "FRESH from $INIT ($(date))" >> "$W/stageB_mod_run.log"
fi

$PY train.py -c "$CFG" $RESUME_ARG --use-amp > "$W/stageB_mod_train.log" 2>&1
TRAIN_RC=$?
echo "TRAIN DONE exit=$TRAIN_RC $(date)" >> "$W/stageB_mod_run.log"
if [ "$TRAIN_RC" -ne 0 ]; then
  echo "TRAIN FAILED rc=$TRAIN_RC (重跑本脚本会从 last.pth 续训)" >> "$W/stageB_mod_run.log"
  exit "$TRAIN_RC"
fi

CKPT=$(ls -t "$OUT"/*.pth 2>/dev/null | head -1)
echo "STAGE_B CKPT=$CKPT" >> "$W/stageB_mod_run.log"
if [ -z "$CKPT" ]; then echo "NO CKPT, abort eval"; exit 1; fi

for lv in $LV; do
  cfg=${PREF}${lv}.yml; [ "$lv" = "day" ] && cfg=${PREF}orig.yml
  echo "=== stageB eval $lv $(date) ===" >> "$W/stageB_mod_run.log"
  $PY train.py -c "$cfg" -r "$CKPT" --use-amp --test-only > "$W/stageB_mod_eval_${lv}.log" 2>&1
done
echo "ALL DONE STAGE_B $(date)" >> "$W/stageB_mod_run.log"
