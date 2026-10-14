#!/usr/bin/env bash
# Dump val200 detections for the four top-3 models on the global leaderboard, so
# that a paired bootstrap can say whether their 0.4-pt mAP gaps are real.
set -u
ROOT=/home/constant/CodeBuddy/Dev/AIC
PY=/home/constant/miniconda3/envs/deimv2/bin/python
export PYTHONPATH="$ROOT/scripts/lib:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT"

O="$ROOT/outputs/analysis/_dets"
mkdir -p "$O"

run () {  # tag cfg_dir
  local tag="$1" dir="$2"
  local cfg="$ROOT/vendor/DEIMv2/configs/deimv2/${dir}.yml"
  local ckpt="$ROOT/vendor/DEIMv2/outputs/${dir}/best_stg1.pth"
  if [ -f "$O/$tag.pkl" ]; then echo "[skip] $tag"; return; fi
  echo "=== dump $tag ==="
  $PY "$ROOT/scripts/analysis/dump_val_dets.py" \
      --config "$cfg" --ckpt "$ckpt" --out "$O/$tag.pkl" --tag "$tag" \
      > "$ROOT/logs/dumpval_$tag.log" 2>&1
  echo "   exit=$? -> $(tail -2 "$ROOT/logs/dumpval_$tag.log" | tr '\n' ' ')"
}

run rgb_vits  deimv2_dinov3_l_aic_nightsim_inorm_rnight
run fuse_l4   deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir_midboost_l4
run fuse_l4u3 deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir_midboost_l4u3
run fuse_combo deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir_midboost_combo

echo "ALL_DONE"
ls -la "$O"
