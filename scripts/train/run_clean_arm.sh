#!/usr/bin/env bash
# =============================================================================
# Launch one clean-baseline arm on any machine (cloud or lab), from a repo checkout laid out like the
# repository (inputs staged with scripts/prep/pack_training_inputs.py).
#   ARM=v1     configs/deimv2/aic_clean_baseline_v1.yml
#   ARM=rgb    configs/deimv2/aic_clean_baseline_v1_rgb_control.yml
#   ARM=night  configs/deimv2/aic_clean_baseline_v1_night.yml
#   ARM=v2     configs/deimv2/aic_clean_baseline_v2_aug_all2000.yml (needs the merged all2000 annotation file)
# Precision follows train.py's clean-baseline guard: on SM >= 8.0 the configs' use_amp: true trains in BF16
# autocast with TF32 on; on Volta TF32 and cuDNN are off and train.py rejects use_amp: true, so set USE_AMP=0
# there (pure FP32). Refuses to overwrite an existing output folder unless RESUME
# points at a checkpoint. With AUTO_SHUTDOWN=1 the machine powers off after train.py exits successfully.
# Usage: ROOT=/path/to/repo ARM=v1 GPU=0 [PY=python] [RESUME=.../last.pth] [USE_AMP=0] [AUTO_SHUTDOWN=1] \
#        bash scripts/train/run_clean_arm.sh
# Run scripts/train/preflight_cloud.sh first.
# =============================================================================
set -euo pipefail

ROOT=${ROOT:?set ROOT to the repo checkout}
ARM=${ARM:?set ARM to v1, rgb, night or v2}
GPU=${GPU:?set GPU to the physical GPU index}
PY=${PY:-python}
case "$ARM" in
  v1) CFG=aic_clean_baseline_v1 ;;
  rgb) CFG=aic_clean_baseline_v1_rgb_control ;;
  night) CFG=aic_clean_baseline_v1_night ;;
  v2) CFG=aic_clean_baseline_v2_aug_all2000 ;;
  *) echo "ARM must be v1, rgb, night or v2"; exit 1 ;;
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
UPDATES=()
if [ -n "${RESUME:-}" ]; then
  [ -f "$RESUME" ] || { echo "RESUME not found: $RESUME"; exit 1; }
  ARGS=(-r "$RESUME")
  UPDATES+=(tuning=null)
elif [ -e "$OUT" ]; then
  echo "refusing: $ROOT/vendor/DEIMv2/$OUT exists (set RESUME=.../last.pth to continue)"; exit 1
fi
# train.py keeps only the last -u, so every YAML override goes into one -u.
if [ "${USE_AMP:-1}" = 0 ]; then
  UPDATES+=(use_amp=False)
fi
if [ ${#UPDATES[@]} -gt 0 ]; then
  ARGS+=(-u "${UPDATES[@]}")
fi

echo "$(date '+%F %T') start ARM=$ARM CFG=$CFG GPU=$GPU RESUME=${RESUME:-none} USE_AMP=${USE_AMP:-1} commit=$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo n/a)" | tee -a "$LOG"
"$PY" -u train.py -c "configs/deimv2/$CFG.yml" "${ARGS[@]+"${ARGS[@]}"}" 2>&1 | tee -a "$LOG"

if [ "${AUTO_SHUTDOWN:-0}" = 1 ]; then
  echo "$(date '+%F %T') training finished, auto-shutdown triggered by AUTO_SHUTDOWN=1" | tee -a "$LOG"
  sync
  if [ -x /usr/bin/shutdown ]; then
    /usr/bin/shutdown
  else
    poweroff
  fi
fi
