#!/usr/bin/env bash
# DEIM-D-FINE-X 多模态 v1 (HGNetv2-B5 双干 + IR CSSA 注入) AIC 训练 + day/9档night 评测
# 起点: ckpts/dfine_x_obj2coco_mm.pth (obj2coco 成品, backbone.* -> backbone.rgb_net.* 重映射)
# 配置: deim_hgnetv2_x_aic_obj2coco_ft_dcnmm.yml (单变量 vs 无 DCN 的 RGB obj2coco_ft 基线)
# 排队: 等待 GPU 空闲显存 >= 14 GiB 后自动开跑 (dcnv3_s1 系列结束后接力)
set -u
cd /home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2:/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/constant/miniconda3/envs/deimv2/bin/python
W=/home/constant/CodeBuddy/Dev/AIC/logs
CFG=configs/deimv2/deim_hgnetv2_x_aic_obj2coco_ft_dcnmm.yml
OUT=./outputs/deim_hgnetv2_x_aic_obj2coco_ft_dcnmm
TUN=ckpts/dfine_x_obj2coco_mm.pth

echo "===== START DFINE_X_MM_QUEUED $(date) =====" >> "$W/dfine_x_dcnmm_run.log"

# ---- 前置: 干净 _na 6ch 缓存 (缺失则重建, ~2h CPU) ----
CACHE=/home/constant/CodeBuddy/Dev/AIC/work_dir/deepir_cache_na/train
if [ ! -d "$CACHE" ] || [ "$(ls "$CACHE" 2>/dev/null | wc -l)" -lt 1800 ]; then
  echo "building clean _na cache $(date)" >> "$W/dfine_x_dcnmm_run.log"
  PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}" \
    $PY /home/constant/CodeBuddy/Dev/AIC/scripts/prep/prep_deepir_cache.py \
      --out "$CACHE" \
      --img_folder /home/constant/CodeBuddy/Dev/AIC/work_dir/deimv2_dataset/train/images \
      --ann_file /home/constant/CodeBuddy/Dev/AIC/work_dir/deimv2_dataset/train/annotations/instances_train.json \
      --ir_folder /home/constant/CodeBuddy/Dev/AIC/work_dir/deimv2_heat_na/images \
      --raw_ir_folder /home/constant/CodeBuddy/Dev/AIC/work_dir/deimv2_ir_na/images \
      --depth_folder /home/constant/CodeBuddy/Dev/AIC/dataset/train/depth \
      --n_channels 6 --modality rgb_ir > "$W/dfine_x_dcnmm_cache.log" 2>&1
  echo "cache rc=$? ($(ls "$CACHE" 2>/dev/null | wc -l) files) $(date)" >> "$W/dfine_x_dcnmm_run.log"
fi

# ---- 等待 GPU 空闲 >= 14 GiB (轮询 60s) ----
while true; do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "$FREE" -ge 14336 ]; then
    echo "GPU free ${FREE}MiB, launching $(date)" >> "$W/dfine_x_dcnmm_run.log"
    break
  fi
  sleep 60
done

echo "===== TRAIN START DFINE_X_MM $(date) =====" >> "$W/dfine_x_dcnmm_run.log"
$PY train.py -c "$CFG" -t "$TUN" --use-amp > "$W/dfine_x_dcnmm_train.log" 2>&1
TRC=$?
echo "train rc=$TRC" >> "$W/dfine_x_dcnmm_run.log"
if [ "$TRC" -ne 0 ]; then
  echo "TRAIN FAILED (rc=$TRC), skip evals" >> "$W/dfine_x_dcnmm_run.log"
  exit 1
fi

CKPT=$(ls -t "$OUT"/*.pth 2>/dev/null | head -1)
echo "DFINE_X_MM CKPT=$CKPT" >> "$W/dfine_x_dcnmm_run.log"
[ -z "$CKPT" ] && { echo "NO CKPT, abort eval"; exit 1; }

PREF=configs/deimv2/deim_hgnetv2_x_aic_obj2coco_ft_dcnmm_val_
for lv in day night50 night25 night12 night05 night005 night01 night002 night001 night0005; do
  cfg=${PREF}${lv}.yml; [ "$lv" = "day" ] && cfg=${PREF}orig.yml
  echo "=== dfine_x_mm eval $lv $(date) ===" >> "$W/dfine_x_dcnmm_run.log"
  $PY train.py -c "$cfg" -r "$CKPT" --use-amp --test-only > "$W/dfine_x_dcnmm_eval_${lv}.log" 2>&1
  echo "rc=$? $lv" >> "$W/dfine_x_dcnmm_run.log"
done
echo "ALL DONE DFINE_X_MM $(date)" >> "$W/dfine_x_dcnmm_run.log"
