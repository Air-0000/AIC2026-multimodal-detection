#!/bin/bash
# Sovereign V10 CAA-SAHI Submission Package Runner
# Generates per-image .txt predictions for AIC 2026 Phase 2 test set (1000 images)

set -e
cd /root/autodl-tmp/AIC/vendor/DEIMv2

export PYTHONPATH=/root/autodl-tmp/AIC/vendor/DEIMv2:/root/autodl-tmp/AIC/vendor/Depth-Anything-V2:/root/autodl-tmp/AIC/scripts/lib

OUT_DIR=/root/autodl-tmp/AIC/work_dir/submission_v10_caa_sahi
mkdir -p "$OUT_DIR"

echo "========================================"
echo " Sovereign V10 CAA-SAHI Submission"
echo " Test images: 1000"
echo " Output: $OUT_DIR"
echo "========================================"
date

python /root/autodl-tmp/AIC/scripts/eval/submit_v10_caa_sahi.py \
    --config  configs/deimv2/deim_hgnetv2_x_aic_sovereign_v10_p2_res_rtx6000.yml \
    --checkpoint outputs/deim_hgnetv2_x_aic_sovereign_v10_p2_res_rtx6000/best_stg1.pth \
    --test_dir /root/autodl-tmp/AIC/dataset/test/AIC2026_PHASE_2_1000 \
    --out_dir  "$OUT_DIR" \
    --conf_thresh 0.15 \
    --max_boxes   100

echo ""
echo "=== Submission validation ==="
python /root/autodl-tmp/AIC/scripts/eval/check_submission.py \
    --pred_dir "$OUT_DIR" 2>/dev/null || echo "[WARN] check_submission.py not found or failed — manual check needed"

echo ""
echo "=== Packing submission zip ==="
ZIP_PATH=/root/autodl-tmp/AIC/work_dir/submission_v10_caa_sahi.zip
cd "$OUT_DIR"
zip -r "$ZIP_PATH" . -i "*.txt"
echo "Zip created: $ZIP_PATH"
ls -lh "$ZIP_PATH"

date
echo "All done."
