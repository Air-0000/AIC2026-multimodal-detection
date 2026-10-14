#!/usr/bin/env bash
# Host-side driver: dump val200 detections for the top models, then run the
# paired bootstrap that turns their mAP gaps into confidence intervals.
# Must run on the GPU host (the sandbox has no CUDA and only 10 MB /dev/shm).
set -u
ROOT=/home/constant/CodeBuddy/Dev/AIC
PY=/home/constant/miniconda3/envs/deimv2/bin/python
export PYTHONPATH="$ROOT/scripts/lib:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT"
mkdir -p outputs/analysis/_dets logs

D="$ROOT/vendor/DEIMv2/outputs"
C="$ROOT/vendor/DEIMv2/configs/deimv2"

dump () {  # tag  config_basename  outputs_dirname
  local tag="$1" cfg="$2" dir="$3"
  if [ -s "$ROOT/outputs/analysis/_dets/$tag.pkl" ]; then echo "[skip] $tag"; return 0; fi
  echo "=== dump $tag ==="
  $PY "$ROOT/scripts/analysis/dump_val_dets.py" \
      --config "$C/$cfg.yml" --ckpt "$D/$dir/best_stg1.pth" \
      --out "$ROOT/outputs/analysis/_dets/$tag.pkl" --tag "$tag" \
      > "$ROOT/logs/dumpval_$tag.log" 2>&1
  local rc=$?
  echo "   exit=$rc  $(tail -1 "$ROOT/logs/dumpval_$tag.log")"
  return $rc
}

dump rgb_vits   deimv2_dinov3_l_aic_nightsim_inorm_rnight \
                deimv2_dinov3_l_aic_nightsim_inorm_rnight
dump fuse_l4    deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir_midboost_l4 \
                deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir_midboost_l4
dump fuse_l4u3  deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir_midboost_l4u3 \
                deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir_midboost_l4u3
dump fuse_combo deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir_midboost_combo \
                deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir_midboost_combo

echo
echo "=== bootstrap ==="
$PY "$ROOT/scripts/analysis/bootstrap_val_ci.py" --iters 200 \
    > "$ROOT/logs/val_ci_bootstrap.log" 2>&1
echo "bootstrap exit=$?"
tail -40 "$ROOT/logs/val_ci_bootstrap.log"
echo "ALL_DONE"
