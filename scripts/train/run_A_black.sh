#!/usr/bin/env bash
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
# Near-black realistic-night eval (experiment A, decisive): night01 (x0.01) and
# night005 (x0.005), ambient ~0, only local light + heavy Poisson-Gaussian noise.
# RGB only degraded; IR/Depth untouched. Logs -> /tmp/night_real.
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
  [ -f "$log" ] && { echo "[$(ts)] skip $tag $lv"; return; }
  local C=$(ck "$ckptdir")
  [ -z "$C" ] && { echo "$tag: NO CKPT"; return; }
  echo "[$(ts)] ---- $tag $lv"
  $PY train.py -c "configs/deimv2/${stem}_val_${lv}.yml" -r "$C" --use-amp --test-only > "$log" 2>&1
}
run_ab() {
  local lv="$1" v="$2"
  local log="/tmp/night_real/ab_ufall_${lv}_${v}.log"
  [ -f "$log" ] && { echo "[$(ts)] skip ab ufall $lv $v"; return; }
  local C=$(ck "outputs/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall")
  [ -z "$C" ] && { echo "ufall: NO CKPT"; return; }
  echo "[$(ts)] ---- ab ufall $lv $v"
  $PY train.py -c "configs/deimv2/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_val_${lv}_ab${v}.yml" \
    -r "$C" --use-amp --test-only > "$log" 2>&1
}

for lv in night01 night005; do
  run_real rgb3ch deimv2_dinov3_l_aic                                       outputs/deimv2_dinov3_l_aic $lv
  run_real ufall  deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall  outputs/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall $lv
  run_real rnight deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight        outputs/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight $lv
  for v in irblk depblk bothblk; do run_ab $lv $v; done
done
echo "[$(ts)] ===== A BLACK DONE ====="
