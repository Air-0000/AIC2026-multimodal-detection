#!/bin/bash
set -u
cd /home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
PY=/home/constant/miniconda3/envs/deimv2/bin/python
CKPT_DEEPIR=outputs/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir/best_stg2.pth
CKPT_UFALL=outputs/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall/best_stg2.pth
PREF_DEEPIR=configs/deimv2/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir_val_
PREF_UFALL=configs/deimv2/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_val_
W=/home/constant/CodeBuddy/Dev/AIC/logs
LV="night50 night25 night12 night05 night01 night005 night002 night001 night0005"

echo "START ALL $(date)" >> "$W/bfusion_run.log"

# --- ufall fair baseline (9 levels, same smoke protocol) ---
for lv in $LV; do
  cfg=${PREF_UFALL}${lv}.yml
  [ -f "$cfg" ] || { echo "skip missing $cfg"; continue; }
  echo "=== START ufall $lv $(date) ===" >> "$W/bfusion_run.log"
  $PY train.py -c "$cfg" -r "$CKPT_UFALL" --use-amp --test-only > "$W/ufall_eval_${lv}.log" 2>&1
  echo "=== DONE ufall $lv exit=$? $(date) ===" >> "$W/bfusion_run.log"
done

# --- deepir brightness-adaptive aux gate sweep ---
# tag -> "GAIN BIAS FLOOR CAP"
declare -A G=(
  [g1]="10 0.12 0.0 1.4"
  [g2]="10 0.25 0.0 1.4"
  [g3]="16 0.12 0.0 1.6"
  [g4]="6 0.18 0.0 1.3"
)
ALLV="day night50 night25 night12 night05 night01 night005 night002 night001 night0005"
for tag in g1 g2 g3 g4; do
  set -- ${G[$tag]}
  GAIN=$1; BIAS=$2; FLOOR=$3; CAP=$4
  for lv in $ALLV; do
    cfg=${PREF_DEEPIR}${lv}.yml
    [ "$lv" = "day" ] && cfg=${PREF_DEEPIR}orig.yml
    echo "=== START deepir $tag $lv $(date) ===" >> "$W/bfusion_run.log"
    AUX_GATE=brightness AUX_GATE_GAIN=$GAIN AUX_GATE_BIAS=$BIAS \
    AUX_GATE_FLOOR=$FLOOR AUX_GATE_CAP=$CAP \
    $PY train.py -c "$cfg" -r "$CKPT_DEEPIR" --use-amp --test-only > "$W/deepir_gate_${tag}_${lv}.log" 2>&1
    echo "=== DONE deepir $tag $lv exit=$? $(date) ===" >> "$W/bfusion_run.log"
  done
done

echo "=== ALL DONE $(date) ===" >> "$W/bfusion_run.log"
cd "$W" && python3 /home/constant/CodeBuddy/Dev/AIC/scripts/analysis/plot_deepir_auxgate.py >> "$W/bfusion_run.log" 2>&1
echo "=== FIGURE REGENERATED $(date) ===" >> "$W/bfusion_run.log"
