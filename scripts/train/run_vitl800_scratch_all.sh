#!/bin/bash
# =============================================================================
# 一次性训练总控: Stage A -> Stage B -> Stage C, 顺序执行, 任一阶段失败即停。
#
#   A  RGB-only 40ep   : 只用 RGB 学会检测            (aux 恒零, RGB 末 6 层解冻)
#   B  模态注入 20ep   : RGB 全冻结, 放开 IR/Depth 与注入通路
#   C  联合微调 40ep   : 全开低 lr (主干 2e-5), 端到端收敛
#
# 每个阶段脚本自身都有:
#   * GPU >= 12GB 排队
#   * last.pth 续训兜底 (被 kill 后重跑本脚本即可接着跑)
#   * 训完自动跑 10 档 night 评测
#
# 用法 (在宿主机上后台跑, 沙箱看不到 GPU):
#   nohup bash scripts/train/run_vitl800_scratch_all.sh > /dev/null 2>&1 &
# 只跑单阶段:
#   bash scripts/train/run_vitl800_scratch_stageA.sh
# =============================================================================
set -u
S=/home/constant/CodeBuddy/Dev/AIC/scripts/train
W=/home/constant/CodeBuddy/Dev/AIC/logs

echo "===== PIPELINE START $(date) =====" >> "$W/scratch_pipeline.log"

for stage in A B C; do
  case $stage in
    A) SH=$S/run_vitl800_scratch_stageA.sh ; LOG=$W/stageA_rgb_run.log ;;
    B) SH=$S/run_vitl800_scratch_stageB.sh ; LOG=$W/stageB_mod_run.log ;;
    C) SH=$S/run_vitl800_scratch_stageC.sh ; LOG=$W/stageC_fuse_run.log ;;
  esac
  echo "--- entering stage $stage $(date) ---" >> "$W/scratch_pipeline.log"
  bash "$SH"
  RC=$?
  if [ "$RC" -ne 0 ]; then
    echo "STAGE $stage FAILED rc=$RC $(date)  (see $LOG)" >> "$W/scratch_pipeline.log"
    echo "PIPELINE ABORTED at stage $stage" >> "$W/scratch_pipeline.log"
    exit "$RC"
  fi
  echo "--- stage $stage OK $(date) ---" >> "$W/scratch_pipeline.log"
done

echo "===== PIPELINE ALL DONE $(date) =====" >> "$W/scratch_pipeline.log"
