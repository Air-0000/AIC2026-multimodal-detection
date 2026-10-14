#!/bin/bash
# 诊断: Stage C 为什么在重试时 1 秒内退出 (rc=120, 日志为空)
# 手动跑一次训练, 限时 60s, 把前 25 行输出原样打出来。
set -u
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2 || exit 1

PY=/home/constant/miniconda3/envs/deimv2/bin/python
CFG=configs/deimv2/deimv2_dinov3_l_aic_vitl800_scratch_stageC_fuse.yml
OUT=outputs/vitl800_scratch_stageC_fuse
LOG=/home/constant/.cache/stageC_probe.log

echo "--- disk before ---"
df -h / | tail -1

timeout 60 "$PY" train.py -c "$CFG" -r "$OUT/last.pth" --use-amp > "$LOG" 2>&1
rc=$?
echo "--- train.py exit code = $rc ---"

echo "--- log size: $(stat -c %s "$LOG" 2>/dev/null) bytes ---"
echo "--- first 25 non-empty lines ---"
grep -v '^[[:space:]]*$' "$LOG" 2>/dev/null | head -25
echo "--- last 25 lines ---"
tail -25 "$LOG" 2>/dev/null

echo "--- disk after ---"
df -h / | tail -1
