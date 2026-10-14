#!/bin/bash
# 只读盘点: 找出可安全删除的中间 checkpoint, 并检查是否被任何 config/脚本引用。
# 不删除任何东西 —— 只生成清单供人工确认。
set -u
R=/home/constant/CodeBuddy/Dev/AIC
O=$R/vendor/DEIMv2/outputs
MAN=$R/outputs/metrics/disk_cleanup_manifest.txt

echo "###### A) 旧实验中间 checkpoint 清单 (排除本次 scratch) ######"
find "$O" -maxdepth 2 -name 'checkpoint*.pth' -size +100M \
     ! -path "$O/vitl800_scratch_*" -printf '%s\t%p\n' | sort -rn > "$MAN"
echo "manifest -> $MAN"
awk '{s+=$1; n++} END {printf "  总计 %d 个文件, %.1f GB\n", n, s/1073741824}' "$MAN"

echo
echo "###### B) 按实验目录汇总 (前 20) ######"
awk -F'\t' '{p=$2; sub(/\/[^/]*$/, "", p); s[p]+=$1; c[p]++}
     END {for (k in s) printf "%.1f GB\t%d files\t%s\n", s[k]/1073741824, c[k], k}' "$MAN" \
  | sort -rn | head -20

echo
echo "###### C) 引用安全检查: configs / scripts 是否引用 checkpoint*.pth ######"
n1=$(grep -rl 'checkpoint0' "$R/vendor/DEIMv2/configs" 2>/dev/null | wc -l)
n2=$(grep -rl 'checkpoint0' "$R/scripts" 2>/dev/null | wc -l)
echo "  configs 中引用的文件数: $n1"
echo "  scripts 中引用的文件数: $n2"
if [ "$n1" -gt 0 ]; then grep -rn 'checkpoint0' "$R/vendor/DEIMv2/configs" 2>/dev/null | head -5; fi

echo
echo "###### D) 每个旧实验都会保留的权重 ######"
find "$O" -maxdepth 2 -name 'best_stg*.pth' ! -path "$O/vitl800_scratch_*" | wc -l \
  | awk '{print "  保留 best_stg*.pth: "$1" 个"}'
find "$O" -maxdepth 2 -name 'last.pth' ! -path "$O/vitl800_scratch_*" | wc -l \
  | awk '{print "  保留 last.pth     : "$1" 个"}'
