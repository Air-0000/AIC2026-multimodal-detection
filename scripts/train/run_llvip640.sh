#!/usr/bin/env bash
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
# =============================================================================
# Real-night OOD test at the SAME 640 resolution as training.
#
#   LLVIP is a genuine nighttime RGB+IR dataset (low-light person detection).
#   The original LLVIP evals resized to 512 while both training and AIC-val use
#   640 -- a documented protocol mismatch (map_summary section 四) that depresses
#   every model and makes the RGB-only vs multimodal comparison unsafe.
#
#   This re-runs the three night-curve models at 640 on LLVIP so the real-night
#   ranking can be read against the synthetic dark-curve
#   (dark05: prompt 0.295 > diffuse 0.224 >> rgb3ch 0.068).
#
#   Single 15.5GiB GPU -> serialised behind run_night_pipeline.sh.
# =============================================================================
set -u
DEIM=/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
PY=/home/constant/miniconda3/envs/deimv2/bin/python
OUT=/tmp/llvip640
mkdir -p "$OUT"
ts() { date '+%F %T'; }

cd "$DEIM" || exit 1

while pgrep -f "run_night_pipeline.sh" >/dev/null 2>&1; do sleep 60; done
echo "[$(ts)] night pipeline drained -> LLVIP@640 real-night test"

run() {   # run <stem> <ckpt> <tag>
    $PY train.py -c "configs/deimv2/$1.yml" -r "$2" --use-amp --test-only > "$OUT/$3.log" 2>&1
    printf '  %-10s AP=%s  AP50=%s\n' "$3" \
        "$(grep -aF 'IoU=0.50:0.95 | area=   all | maxDets=100 ] = ' "$OUT/$3.log" | head -1 | sed 's/.*\] = //' | tr -d '\r')" \
        "$(grep -aF 'IoU=0.50      | area=   all | maxDets=100 ] = ' "$OUT/$3.log" | head -1 | sed 's/.*\] = //' | tr -d '\r')"
}

echo "=== LLVIP (real night, OOD) @640 ==="
run deimv2_dinov3_l_aic_rgb3ch_llvip640        outputs/deimv2_dinov3_l_aic/best_stg2.pth                 rgb3ch
run deimv2_dinov3_l_aic_rdttrack_heat_llvip640 outputs/deimv2_dinov3_l_aic_rdttrack_heat/best_stg2.pth  prompt
run deimv2_dinov3_l_aic_rdtdiffuse_llvip640    outputs/deimv2_dinov3_l_aic_rdtdiffuse/best_stg2.pth     diffuse
echo "[$(ts)] LLVIP640 COMPLETE"
