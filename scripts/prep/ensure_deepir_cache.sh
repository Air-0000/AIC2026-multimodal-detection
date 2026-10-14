#!/usr/bin/env bash
# =============================================================================
# 确保 6 通道离线缓存 (work_dir/deepir_cache/train) 完整。
#
# 背景: 2026-09-26 凌晨发现该目录整个消失, 导致 Stage C 续训连续 5 次
#   FileNotFoundError: work_dir/deepir_cache/train/003788.pt
#   -> 每次都在 1 分钟内 rc=1, retry wrapper 5 次全败后放弃。
#   该缓存由 vendor/DEIMv2/prep_deepir_cache.py 生成 (1800 样本 / 约 20 GB /
#   重建约 4 分钟, 支持断点续跑)。它此前是手工执行的, 没有任何脚本保证其存在,
#   也没有任何流程在训练前检查 —— 于是缺失时表现为"训练启动即死"。
#
# 用法:
#   bash scripts/prep/ensure_deepir_cache.sh          # 检查并按需重建
#   bash scripts/prep/ensure_deepir_cache.sh --check  # 只检查, 不重建
#
# 退出码: 0 = 缓存就绪; 1 = 缺失且未修复 (或 --check 下缺失)
# =============================================================================
set -u

R=/home/constant/CodeBuddy/Dev/AIC
CACHE="$R/work_dir/deepir_cache/train"
ANN="$R/work_dir/deimv2_dataset/train/annotations/instances_train.json"
PY=/home/constant/miniconda3/envs/deimv2/bin/python
CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

want=$("$PY" -c "import json;print(len(json.load(open('$ANN'))['images']))" 2>/dev/null || echo 0)
have=$(ls "$CACHE" 2>/dev/null | wc -l)

echo "[cache] dir=$CACHE"
echo "[cache] have=$have  want=$want"

if [ "$want" -eq 0 ]; then
  echo "[cache] ERROR: 读不到标注文件 $ANN"
  exit 1
fi

if [ "$have" -ge "$want" ]; then
  echo "[cache] OK"
  exit 0
fi

if [ "$CHECK_ONLY" -eq 1 ]; then
  echo "[cache] MISSING ($((want - have)) 个) —— --check 模式不重建"
  exit 1
fi

echo "[cache] 不完整 -> 重建 (断点续跑, 约 4 分钟)"
cd "$R/vendor/DEIMv2" || exit 1
env PYTHONPATH="$R/scripts/lib" "$PY" prep_deepir_cache.py
rc=$?
have=$(ls "$CACHE" 2>/dev/null | wc -l)
echo "[cache] rebuild rc=$rc  after: have=$have want=$want"

if [ "$have" -ge "$want" ]; then
  echo "[cache] OK (重建完成)"
  exit 0
fi
echo "[cache] FAILED: 重建后仍不完整"
exit 1
