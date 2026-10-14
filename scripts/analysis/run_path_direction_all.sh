#!/usr/bin/env bash
# 对三阶段各跑一次轨迹方向一致性分析。
set -u
REPO=/home/constant/CodeBuddy/Dev/AIC
PY=/home/constant/miniconda3/envs/deimv2/bin/python
cd "$REPO" || exit 1
for d in vitl800_scratch_stageA_rgb vitl800_scratch_stageB_mod vitl800_scratch_stageC_fuse; do
    echo "########## $d ##########"
    $PY scripts/analysis/path_direction_analysis.py "vendor/DEIMv2/outputs/$d" \
        --json "outputs/path_direction_$d.json" 2>&1 | tail -48
    echo
done
