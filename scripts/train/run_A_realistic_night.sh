#!/usr/bin/env bash
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
# =============================================================================
# Experiment A: realistic-night isolation chain on AIC val.
#
# The OLD darkXX protocol (RGB x const) is gameable by instance-norm and
# OVER-states RGB recoverability at deep night. This runs the isolation chain on
# the REALISTIC nightXX protocol (prep_night_real.py: global log-attenuation +
# non-uniform lamp/vignette field + colour shift + Poisson-Gaussian noise, RGB
# only; IR/Depth untouched) so we learn how much aux actually contributes in a
# domain that looks like the competition's real night frames.
#
#   rgb3ch   (3ch, no IR/Depth)  -> single-modal baseline at realistic night
#   ufall    (5ch, E11 full-unfreeze, best 5ch recipe) -> multimodal
#   rnight   (5ch, predecessor)  -> multimodal (consistency check)
#   + ufall aux ablation: irblk / depblk / bothblk  (isolates IR / Depth / aux)
#
# Logs: /tmp/night_real/<tag>_<lv>.log  (real)
#       /tmp/night_real/ab_ufall_<lv>_<v>.log  (ablation)
# Parse with: python night_curve2.py /tmp/night_real
# =============================================================================
set -u
DEIM=/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
PY=/home/constant/miniconda3/envs/deimv2/bin/python
mkdir -p /tmp/night_real
cd "$DEIM" || exit 1
ts() { date '+%F %T'; }

ck() { for f in best_stg2.pth best_stg1.pth last.pth; do [ -f "$1/$f" ] && { echo "$1/$f"; return; }; done; echo ""; }

run_real() {
  local tag="$1" stem="$2" ckptdir="$3" lv="$4"
  local log="/tmp/night_real/${tag}_${lv}.log"
  [ -f "$log" ] && { echo "[$(ts)] skip $tag $lv (exists)"; return; }
  local C=$(ck "$ckptdir")
  [ -z "$C" ] && { echo "$tag: NO CKPT"; return; }
  echo "[$(ts)] ---- $tag $lv  ($C)"
  $PY train.py -c "configs/deimv2/${stem}_val_${lv}.yml" -r "$C" --use-amp --test-only > "$log" 2>&1
  echo "[$(ts)]      done $tag $lv"
}

run_ab() {
  local lv="$1" v="$2"
  local log="/tmp/night_real/ab_ufall_${lv}_${v}.log"
  [ -f "$log" ] && { echo "[$(ts)] skip ab ufall $lv $v (exists)"; return; }
  local C=$(ck "outputs/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall")
  [ -z "$C" ] && { echo "ufall: NO CKPT"; return; }
  echo "[$(ts)] ---- ab ufall $lv $v  ($C)"
  $PY train.py -c "configs/deimv2/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_val_${lv}_ab${v}.yml" \
    -r "$C" --use-amp --test-only > "$log" 2>&1
  echo "[$(ts)]      done ab ufall $lv $v"
}

LV="night50 night25 night12 night05"
for lv in $LV; do
  run_real rgb3ch deimv2_dinov3_l_aic                                        outputs/deimv2_dinov3_l_aic $lv
  run_real ufall  deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall  outputs/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall $lv
  run_real rnight deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight        outputs/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight $lv
done
for lv in night50 night05; do
  for v in irblk depblk bothblk; do
    run_ab $lv $v
  done
done
echo "[$(ts)] ===== A DONE ====="
