#!/bin/bash
# =============================================================================
# 清理旧实验的中间 checkpoint。用户已确认 (2026-09-26)。
#
# 删除对象: vendor/DEIMv2/outputs/*/checkpoint*.pth   (maxdepth 2, >100MB)
#   - 排除本次 scratch 三阶段 (vitl800_scratch_*)
#   - 排除唯一被引用的那个 (run_midboost_ablate.sh 用)
#   - 每个实验的 best_stg*.pth / last.pth 一律保留
# 另删: 本次 Stage C 那个被磁盘写满截断的损坏文件 checkpoint0019.pth
#
# 安全守卫: 每个路径必须 (a) 落在 $O/ 之下 (b) 文件名匹配 checkpoint*.pth,
#           否则跳过并报警 —— 防止清单损坏导致误删。
# =============================================================================
set -u
R=/home/constant/CodeBuddy/Dev/AIC
O=$R/vendor/DEIMv2/outputs
MAN=$R/outputs/metrics/disk_cleanup_manifest.txt
KEEP="$O/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir_midboost_combo/checkpoint0049.pth"
CORRUPT="$O/vitl800_scratch_stageC_fuse/checkpoint0019.pth"

echo "### 磁盘 (删除前)"; df -h / | tail -1

# 重新生成清单, 避免用过期数据
find "$O" -maxdepth 2 -name 'checkpoint*.pth' -size +100M \
     ! -path "$O/vitl800_scratch_*" -printf '%s\t%p\n' | sort -rn > "$MAN"
TOTAL=$(awk '{s+=$1} END {printf "%d", s}' "$MAN")
NFILES=$(wc -l < "$MAN")
echo "### 清单: $NFILES 个文件, $(awk -v t="$TOTAL" 'BEGIN{printf "%.1f", t/1073741824}') GB"

deleted=0; skipped=0; bytes=0
while IFS=$'\t' read -r sz path; do
  case "$path" in
    "$O"/*) ;;                                  # 必须在 outputs 之下
    *) echo "SKIP (outside outputs): $path"; skipped=$((skipped+1)); continue ;;
  esac
  case "$(basename "$path")" in
    checkpoint*.pth) ;;                         # 必须是中间 checkpoint
    *) echo "SKIP (name): $path"; skipped=$((skipped+1)); continue ;;
  esac
  if [ "$path" = "$KEEP" ]; then
    echo "KEEP (referenced by run_midboost_ablate.sh): $path"
    skipped=$((skipped+1)); continue
  fi
  if rm -f -- "$path"; then
    deleted=$((deleted+1)); bytes=$((bytes+sz))
  else
    echo "FAIL rm: $path"; skipped=$((skipped+1))
  fi
done < "$MAN"

echo "### 旧实验中间 ckpt: 删除 $deleted 个, 跳过 $skipped 个, 释放 $(awk -v b="$bytes" 'BEGIN{printf "%.1f", b/1073741824}') GB"

# 损坏的截断文件 (同样带存在性检查)
if [ -f "$CORRUPT" ]; then
  cs=$(stat -c %s "$CORRUPT")
  rm -f -- "$CORRUPT" && echo "### 已删除损坏文件 checkpoint0019.pth ($(awk -v b="$cs" 'BEGIN{printf "%.2f", b/1073741824}') GB)"
fi

echo "### 保留项核对"
echo "  best_stg*.pth : $(find "$O" -maxdepth 2 -name 'best_stg*.pth' ! -path "$O/vitl800_scratch_*" | wc -l) 个"
echo "  last.pth      : $(find "$O" -maxdepth 2 -name 'last.pth' ! -path "$O/vitl800_scratch_*" | wc -l) 个"
echo "  被引用文件    : $([ -f "$KEEP" ] && echo 在 || echo '!!! 丢失 !!!')"

echo "### 本次 scratch 三阶段 (须完好)"
for d in vitl800_scratch_stageA_rgb vitl800_scratch_stageB_mod vitl800_scratch_stageC_fuse; do
  printf "  %-28s %s\n" "$d" "$(ls "$O/$d"/*.pth 2>/dev/null | wc -l) 个 .pth"
done

echo "### 磁盘 (删除后)"; df -h / | tail -1
