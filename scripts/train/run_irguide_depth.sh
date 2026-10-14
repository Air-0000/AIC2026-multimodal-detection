#!/bin/bash
# End-to-end: optimize AIC night depth (IR-guided guided filter) -> feed into
# the fully-integrated midboost detector -> train -> (eval done separately) so we
# can measure whether "optimizing the depth map" actually helps RGB-D detection.
set -u
cd /home/constant/CodeBuddy/Dev/AIC
W=/home/constant/CodeBuddy/Dev/AIC/logs
SYS=/usr/bin/python3
PY=/home/constant/miniconda3/envs/deimv2/bin/python
RD=$W/depth_refined_irheat
CFG=configs/deimv2/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir_midboost_combo_irguide.yml
DEEPIR_CKPT=outputs/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir/best_stg2.pth

echo "=== [1/3] generate IR-guided refined depth (CPU) $(date) ===" >> "$W/irguide_run.log"
$SYS depth_refine.py dataset/train/depth/ --guide-dir $W/deimv2_heat/images --guide-dir2 $W/deimv2_ir/images --out-dir $RD/train >> "$W/irguide_refine.log" 2>&1
echo "    refined png: $(ls $RD/train/*_refined.png 2>/dev/null | wc -l)" >> "$W/irguide_run.log"

echo "=== [2/3] build flat dir (original basenames) $(date) ===" >> "$W/irguide_run.log"
$PY /home/constant/CodeBuddy/Dev/AIC/scripts/analysis/_build_flat.py >> "$W/irguide_run.log" 2>&1
cnt=$(ls "$W/depth_refined_irheat_png" | wc -l)
exp=$(ls dataset/train/depth | wc -l)
echo "    flat=$cnt expected=$exp" >> "$W/irguide_run.log"
if [ "$cnt" -ne "$exp" ]; then echo "FLAT MISMATCH -> abort"; exit 1; fi

echo "=== [3/3] train midboost_combo_irguide (50ep, tune from deepir best_stg2) $(date) ===" >> "$W/irguide_run.log"
cd /home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
$PY train.py -c "$CFG" -t "$DEEPIR_CKPT" --use-amp >> "$W/irguide_train.log" 2>&1
echo "=== TRAIN DONE $(date) ===" >> "$W/irguide_run.log"
