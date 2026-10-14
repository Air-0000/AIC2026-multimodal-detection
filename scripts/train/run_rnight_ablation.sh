#!/usr/bin/env bash
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
# rnight ablation: prompt (5ch, no RandomNight5ch, backbone frozen) vs
# rnight (5ch, +RandomNight5ch, unfreeze 2 blocks) on night levels.
# NOTE: the two differ in BOTH rnight-aug AND unfreeze_rgb_stages (0 vs 2),
# so this is NOT a clean single-variable test -- it shows rnight-training's
# overall night-deployment gain.
set -u
DEIM=/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
PY=/home/constant/miniconda3/envs/deimv2/bin/python
OUT=/tmp/night_robust
mkdir -p "$OUT"
cd "$DEIM" || exit 1
run() {
  $PY train.py -c "configs/deimv2/$2" -r "$3" --use-amp --test-only > "$OUT/$1.log" 2>&1
  printf '  %-22s AP=%s AP50=%s\n' "$1" \
    "$(grep -aF 'IoU=0.50:0.95 | area=   all | maxDets=100 ] = ' "$OUT/$1.log" | head -1 | sed 's/.*\] = //;s/\r//')" \
    "$(grep -aF 'IoU=0.50      | area=   all | maxDets=100 ] = ' "$OUT/$1.log" | head -1 | sed 's/.*\] = //;s/\r//')"
}
CFG=deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_val
PROMPT=outputs/deimv2_dinov3_l_aic_rdttrack_heat/best_stg2.pth
RNIGHT=outputs/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight/best_stg2.pth
for lv in night50 night25 night12 night05; do
  echo "===== $lv ====="
  run "prompt_$lv" "${CFG}_${lv}.yml" "$PROMPT"
  run "rnight_$lv" "${CFG}_${lv}.yml" "$RNIGHT"
done
echo "[$(date)] DONE"
