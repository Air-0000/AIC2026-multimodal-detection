#!/usr/bin/env bash
# =============================================================================
# Start the RGB-only control on ailab40 as soon as clean baseline v1 has finished, so the GPU does not sit
# idle overnight. Launches only if v1's process is gone, v1 wrote final.pth, and the GPU has no compute
# process; otherwise it logs why and exits without starting anything.
# Usage (on ailab40): ROOT=<deployed RGB control tree> nohup bash queue_rgb_control_after_v1_ailab40.sh &
# =============================================================================
set -euo pipefail

ROOT=${ROOT:?set ROOT to the deployed RGB control tree}
WAIT_PID=${WAIT_PID:-18924}
GPU=${GPU:-1}
V1_OUT=${V1_OUT:-/home/ailab40/hejianzhong/AIC_CLEAN_BASELINE_20261021/vendor/DEIMv2/outputs/aic_clean_baseline_v1}
LOG="${ROOT}/logs/queue_rgb_control.log"

echo "$(date '+%F %T') waiting for PID ${WAIT_PID}" >> "${LOG}"
while kill -0 "${WAIT_PID}" 2>/dev/null; do sleep 300; done
sleep 120
if [ ! -f "${V1_OUT}/final.pth" ]; then
  echo "$(date '+%F %T') v1 ended without ${V1_OUT}/final.pth; not launching" >> "${LOG}"; exit 1
fi
busy=$(nvidia-smi -i "${GPU}" --query-compute-apps=pid --format=csv,noheader)
if [ -n "${busy}" ]; then
  echo "$(date '+%F %T') GPU ${GPU} busy (${busy}); not launching" >> "${LOG}"; exit 1
fi
echo "$(date '+%F %T') launching RGB control on GPU ${GPU}" >> "${LOG}"
ROOT="${ROOT}" GPU="${GPU}" exec bash "${ROOT}/scripts/train/run_clean_baseline_v1_rgb_control_ailab40.sh"
