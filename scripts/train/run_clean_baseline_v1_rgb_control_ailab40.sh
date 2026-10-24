#!/usr/bin/env bash
# =============================================================================
# RGB-only control for clean baseline v1: 50 epochs on ailab40 (Tesla V100 32GB).
# Same split, resolution, schedule, optimizer, teacher, Co-ATSS and evaluation as v1; only the IR/depth
# inputs and the multimodal backbone wrapper are removed (configs/deimv2/aic_clean_baseline_v1_rgb_control.yml).
# Runs from the isolated tree built by deploy_rgb_control_ailab40.sh; pure FP32, cuDNN disabled by the
# clean-baseline guard exactly as in v1.
# =============================================================================

set -euo pipefail

export CUDA_VISIBLE_DEVICES=${GPU:-1}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

ROOT=${ROOT:?set ROOT to the deployed RGB control tree}
PY="/home/ailab/ssd/bigdata/hejianzhong/_infra/envs/aic2026/bin/python"
LOG="${ROOT}/logs/aic_clean_baseline_v1_rgb_control_train.log"

export PYTHONPATH="${ROOT}/scripts/lib:${ROOT}/vendor/DEIMv2:${PYTHONPATH:-}"
cd "${ROOT}/vendor/DEIMv2"
[ -e outputs/aic_clean_baseline_v1_rgb_control ] && { echo "refusing: output dir exists"; exit 1; }

echo "Start: $(date)  GPU: ${CUDA_VISIBLE_DEVICES}  Root: ${ROOT}" | tee -a "${LOG}"
exec $PY -u train.py \
  -c configs/deimv2/aic_clean_baseline_v1_rgb_control.yml \
  -u use_amp=False \
  2>&1 | tee -a "${LOG}"
