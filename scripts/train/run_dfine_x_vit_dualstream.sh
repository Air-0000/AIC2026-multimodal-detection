#!/usr/bin/env bash
# =============================================================================
# DEIM-D-FINE-X 异构双流 (HGNetv2-B5 CNN + DAv2 ViT) 多模态训练与评测脚本
# -----------------------------------------------------------------------------
# 架构: HGNetv2ViTDualStream (CNN 光热局部流 + DAv2 ViT 3D 几何大模型流)
# 起点: ckpts/dfine_x_obj2coco.pth (官方 COCO 59.3 / 内部 0.5550 AP 最强基座热启动)
# DAv2: work_dir/da2_weights/depth_anything_v2_vits.pth
# 配置: configs/deimv2/deim_hgnetv2_vit_dualstream_aic.yml
# =============================================================================
set -u
cd /home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2:/home/constant/CodeBuddy/Dev/AIC/vendor/Depth-Anything-V2:/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/constant/miniconda3/envs/deimv2/bin/python
W=/home/constant/CodeBuddy/Dev/AIC/logs
CFG=configs/deimv2/deim_hgnetv2_vit_dualstream_aic.yml
OUT=./outputs/deim_hgnetv2_vit_dualstream_aic
TUN=ckpts/dfine_x_obj2coco.pth

mkdir -p "$W"
mkdir -p "$OUT"

echo "===== START DFINE_X_VIT_DUALSTREAM $(date) =====" >> "$W/dfine_x_vit_dualstream_run.log"

# ---- 检查 GPU 显存 ----
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
echo "GPU free memory: ${FREE} MiB" | tee -a "$W/dfine_x_vit_dualstream_run.log"

echo "===== LAUNCHING TRAINING: $CFG =====" | tee -a "$W/dfine_x_vit_dualstream_run.log"
$PY train.py -c "$CFG" -t "$TUN" --use-amp 2>&1 | tee "$W/dfine_x_vit_dualstream_train.log"
TRC=${PIPESTATUS[0]}
echo "Training finished with exit code $TRC $(date)" >> "$W/dfine_x_vit_dualstream_run.log"

if [ "$TRC" -ne 0 ]; then
  echo "TRAIN FAILED (rc=$TRC)" | tee -a "$W/dfine_x_vit_dualstream_run.log"
  exit 1
fi

echo "Training completed successfully! Starting 10-level evaluation (Day + 9 Night)..." | tee -a "$W/dfine_x_vit_dualstream_run.log"

CKPT="$OUT/best_stg1.pth"
if [ ! -f "$CKPT" ]; then
  CKPT="$OUT/last.pth"
fi
echo "Using evaluation checkpoint: $CKPT" | tee -a "$W/dfine_x_vit_dualstream_run.log"

PREF=configs/deimv2/deim_hgnetv2_vit_dualstream_val_
for lv in day night50 night25 night12 night05 night01 night005 night002 night001 night0005; do
  cfg=${PREF}${lv}.yml
  [ "$lv" = "day" ] && cfg=${PREF}orig.yml
  echo "=== Evaluating $lv $(date) ===" | tee -a "$W/dfine_x_vit_dualstream_run.log"
  $PY train.py -c "$cfg" -r "$CKPT" --use-amp --test-only > "$W/dfine_x_vit_dualstream_eval_${lv}.log" 2>&1
  echo "Eval rc=$? for $lv" >> "$W/dfine_x_vit_dualstream_run.log"
done

echo "===== ALL EVALUATIONS COMPLETE =====" | tee -a "$W/dfine_x_vit_dualstream_run.log"
$PY /home/constant/CodeBuddy/Dev/AIC/scripts/analysis/parse_dualstream_eval_matrix.py | tee -a "$W/dfine_x_vit_dualstream_run.log"

