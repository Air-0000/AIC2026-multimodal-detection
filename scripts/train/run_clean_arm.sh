#!/usr/bin/env bash
# =============================================================================
# Launch one clean-baseline arm on any machine (cloud or lab), from a repo checkout laid out like the
# repository (inputs staged with scripts/prep/pack_training_inputs.py).
#   ARM=v1     configs/deimv2/aic_clean_baseline_v1.yml
#   ARM=rgb    configs/deimv2/aic_clean_baseline_v1_rgb_control.yml
#   ARM=night  configs/deimv2/aic_clean_baseline_v1_night.yml
# Precision follows the configs' clean-baseline guard (FP32, TF32 off); cuDNN follows train.py (off only
# on Volta). Refuses to overwrite an existing output folder unless RESUME points at a checkpoint.
# Usage: ROOT=/path/to/repo ARM=v1 GPU=0 [PY=python] [RESUME=.../last.pth] bash scripts/train/run_clean_arm.sh
# Run scripts/train/preflight_cloud.sh first.
# =============================================================================
set -euo pipefail

ROOT=${ROOT:?set ROOT to the repo checkout}
ARM=${ARM:?set ARM to v1, rgb or night}
GPU=${GPU:?set GPU to the physical GPU index}
PY=${PY:-python}
case "$ARM" in
  v1) CFG=aic_clean_baseline_v1 ;;
  rgb) CFG=aic_clean_baseline_v1_rgb_control ;;
  night) CFG=aic_clean_baseline_v1_night ;;
  *) echo "ARM must be v1, rgb or night"; exit 1 ;;
esac

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT/scripts/lib:$ROOT/vendor/DEIMv2:${PYTHONPATH:-}"
mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/${CFG}_train.log"
cd "$ROOT/vendor/DEIMv2"
OUT=$(awk '/^output_dir:/{print $2}' "configs/deimv2/$CFG.yml")

ARGS=()
if [ -n "${RESUME:-}" ]; then
  [ -f "$RESUME" ] || { echo "RESUME not found: $RESUME"; exit 1; }
  ARGS=(-r "$RESUME" -u tuning=null)
elif [ -e "$OUT" ]; then
  echo "refusing: $ROOT/vendor/DEIMv2/$OUT exists (set RESUME=.../last.pth to continue)"; exit 1
fi

echo "$(date '+%F %T') start ARM=$ARM CFG=$CFG GPU=$GPU RESUME=${RESUME:-none} commit=$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo n/a)" | tee -a "$LOG"
"$PY" -u train.py -c "configs/deimv2/$CFG.yml" "${ARGS[@]+"${ARGS[@]}"}" 2>&1 | tee -a "$LOG"
