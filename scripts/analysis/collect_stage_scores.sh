#!/bin/bash
# =============================================================================
# 汇总三阶段(scratch curriculum)各亮度档的评测 mAP。
# 每个 eval 日志里 COCO 摘要会打印 12 行指标, 我们只取 area=all/maxDets=100 的
#   IoU=0.50:0.95 -> 总 mAP   (比赛指标)
#   IoU=0.50      -> AP50
# 若日志里有多次运行(脚本被反复重跑), 取第一次出现的一组, 与 MODEL_RECORDS 口径一致。
# =============================================================================
cd /home/constant/CodeBuddy/Dev/AIC || exit 1

LV="day night50 night25 night12 night05 night01 night005 night002 night001 night0005"

report() {
  local tag="$1" title="$2"
  echo "###### $title ######"
  printf '%-10s %-9s %-9s\n' "档位" "mAP" "AP50"
  for lv in $LV; do
    local f="logs/${tag}_${lv}.log"
    if [ ! -f "$f" ]; then printf '%-10s %-9s %-9s\n' "$lv" "n/a" "n/a"; continue; fi
    local ap ap50
    ap=$(grep -m1 'IoU=0.50:0.95 | area=   all | maxDets=100' "$f" | grep -oE '= [0-9.]+$' | tr -d '= ')
    ap50=$(grep -m1 'IoU=0.50      | area=   all | maxDets=100' "$f" | grep -oE '= [0-9.]+$' | tr -d '= ')
    printf '%-10s %-9s %-9s\n' "$lv" "${ap:-n/a}" "${ap50:-n/a}"
  done
  echo
}

report "stageA_rgb_eval"  "Stage A / RGB-only (scratch, 40ep)"
report "stageB_mod_eval"  "Stage B / 模态注入 (20ep)"
report "stageC_fuse_eval" "Stage C / 联合微调 (若有)"
