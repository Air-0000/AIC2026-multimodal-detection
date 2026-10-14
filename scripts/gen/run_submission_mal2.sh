#!/bin/bash
# run_submission_mal2.sh — 在宿主机前台跑 mal2 提交包生成（val 黄金校验 + test 推理 + 双版打包）
set -u
REPO=/home/constant/CodeBuddy/Dev/AIC
PY=/home/constant/miniconda3/envs/deimv2/bin/python
[ -x "$PY" ] || PY=$(command -v python3)
echo "PY=$PY"
cd "$REPO/vendor/DEIMv2" || exit 1
mkdir -p "$REPO/work_dir/submissions/submission_mal2"
# 输出目录由脚本内部常量 OUTD 决定，不接收 --out 参数
exec "$PY" "$REPO/scripts/gen/make_submission_mal2.py"
