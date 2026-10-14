#!/usr/bin/env bash
set -u
DEIM=/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
PY=/home/constant/miniconda3/envs/deimv2/bin/python
OUT=/tmp/night_full
mkdir -p "$OUT"
cd "$DEIM" || exit 1

declare -A CK=(
  [rgb3ch]=outputs/deimv2_dinov3_l_aic/best_stg2.pth
  [nightsim]=outputs/deimv2_dinov3_l_aic_nightsim_inorm/best_stg1.pth
  [fused]=outputs/deimv2_dinov3_l_aic_fused_tardal/best_stg1.pth
  [ufall]=outputs/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall/best_stg2.pth
)
LEVELS="night50 night25 night12 night05 night01 night005 night002 night001 night0005"
for m in rgb3ch nightsim fused ufall; do
  for lv in $LEVELS; do
    case $m in
      rgb3ch) cfg="deimv2_dinov3_l_aic_val_rgb_${lv}.yml";;
      nightsim) cfg="deimv2_dinov3_l_aic_nightsim_inorm_val_${lv}.yml";;
      fused) cfg="deimv2_dinov3_l_aic_fused_tardal_val_${lv}.yml";;
      ufall) cfg="deimv2_dinov3_l_aic_val_ufall_${lv}.yml";;
    esac
    [ -f "configs/deimv2/$cfg" ] || { echo "NO CFG $cfg"; continue; }
    echo "=== $m $lv ==="
    $PY train.py -c "configs/deimv2/$cfg" -r "${CK[$m]}" --use-amp --test-only > "$OUT/${m}_${lv}.log" 2>&1
    ap=$(grep -aF 'IoU=0.50:0.95 | area=   all | maxDets=100 ] = ' "$OUT/${m}_${lv}.log" | head -1 | sed 's/.*\] = //' | tr -d '\r')
    echo "  $m $lv AP=$ap"
  done
done
echo DONE
