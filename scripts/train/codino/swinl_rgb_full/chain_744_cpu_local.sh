#!/usr/bin/env bash
set -euo pipefail
AIC_CPU_REPO=$(cd "$(dirname "$0")/../../../.." && pwd)
if [ -n "${PILOT_PID:-}" ]; then
 while kill -0 "$PILOT_PID" 2>/dev/null; do sleep 5; done
fi
test -f "$AIC_CPU_REPO/outputs/analysis/aic744_cpu_pilot/metrics.json"
exec caffeinate -i bash "$AIC_CPU_REPO/scripts/train/codino/swinl_rgb_full/run_744_cpu_local.sh" --limit 300 --night-only --out "$AIC_CPU_REPO/outputs/analysis/aic744_cpu_val300_night01"
