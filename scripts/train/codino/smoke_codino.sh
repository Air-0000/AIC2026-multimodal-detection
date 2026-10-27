#!/usr/bin/env bash
# =============================================================================
# Smoke test of a Co-DINO setup after setup_codino_env.sh. Stops at the first failure:
#   0. (vit_l) converts the released ViT-L checkpoint if that has not been done (convert_codino_vit_ckpt.py).
#   1. check_codino_env.py with the training config (GPU kernels, mmcv ops vs references, label files, images).
#   2. The unchanged 80-class Objects365->COCO model: swin_l on one val image (upstream config, look at the saved
#      image); vit_l on all 200 val images, scored on the COCO classes that map onto AIC classes
#      (check_codino_coco_on_aic_val.py; AP near 0 means the port or conversion is broken).
#   3. 40 training iterations + validation on 20 images (smoke config), on NGPUS GPUs; prints the median seconds
#      per iteration, peak memory and the projected time of the 12-epoch run.
# SCALE=16g (vit_l only) runs step 3 at reduced image size so it fits a 16 GB card; its timing is not the real one.
# Output in work_dir/codino/smoke_<model>_<time>/, log in logs/.
# Usage: ROOT=/root/autodl-tmp/AIC_clean_v1 MODEL=vit_l [NGPUS=2] [SCALE=16g] bash scripts/train/codino/smoke_codino.sh
# Pick the GPUs with CUDA_VISIBLE_DEVICES.
# =============================================================================
set -euo pipefail
source "$(dirname "$0")/codino_env.sh"
MODEL=${MODEL:?set MODEL to swin_l or vit_l}
NGPUS=${NGPUS:-1}
CFG=$(codino_config "$MODEL")
ACCUM=$(codino_accum)
SMOKE_CFG=${CFG%_1800}_smoke
[ "${SCALE:-full}" = 16g ] && SMOKE_CFG=${SMOKE_CFG}_16g
[ -f "$CODINO_DIR/$SMOKE_CFG.py" ] || { echo "no smoke config $SMOKE_CFG.py"; exit 1; }
STAMP=$(date +%Y%m%d_%H%M%S)
OUT=$ROOT/work_dir/codino/smoke_${MODEL}_$STAMP
mkdir -p "$OUT" "$ROOT/logs"
LOG=$ROOT/logs/codino_smoke_${MODEL}_$STAMP.log
exec > >(tee -a "$LOG") 2>&1
cd "$ROOT"
nvidia-smi -L

if [ "$MODEL" = vit_l ] && [ ! -f "$VIT_CKPT" ]; then
  echo "== 0 convert the released ViT-L checkpoint"
  "$PY" "$CODINO_DIR/convert_codino_vit_ckpt.py" "$VIT_CKPT_SRC" "$VIT_CKPT"
fi

echo "== 1 environment and config check"
"$PY" "$CODINO_DIR/check_codino_env.py" --config "$CODINO_DIR/$CFG.py"

echo "== 2 the COCO model on AIC val images"
if [ "$MODEL" = vit_l ]; then
  "$PY" "$CODINO_DIR/check_codino_coco_on_aic_val.py" --checkpoint "$VIT_CKPT" --out "$OUT/coco_check"
else
  IMG=$("$PY" -c "import json; print(json.load(open('work_dir/deimv2_dataset/valid/annotations/instances_val.json'))['images'][0]['file_name'])")
  "$PY" "$MMDET_DIR/demo/image_demo.py" "dataset/train/visible/$IMG" \
    "$MMDET_DIR/projects/CO-DETR/configs/codino/co_dino_5scale_swin_l_16xb1_16e_o365tococo.py" \
    --weights "$SWIN_CKPT" --device cuda:0 --pred-score-thr 0.3 --out-dir "$OUT/demo"
  echo "visualisation: $OUT/demo/vis/$IMG"
fi

echo "== 3 40-iteration training smoke ($SMOKE_CFG, NGPUS=$NGPUS, accumulative_counts=$ACCUM)"
codino_train "$CODINO_DIR/$SMOKE_CFG.py" --work-dir "$OUT/train" \
  --cfg-options optim_wrapper.accumulative_counts="$ACCUM"
"$PY" - "$OUT/train" "$NGPUS" <<'EOF'
import glob, json, statistics, sys
rows = [json.loads(l) for f in glob.glob(f'{sys.argv[1]}/*/vis_data/scalars.json') for l in open(f)]
train = [r for r in rows if 'time' in r and 'loss' in r]
times = [r['time'] for r in train if r['iter'] > 10]  # skip warm-up and cuDNN autotuning
step = statistics.median(times)
mem = max(r.get('memory', 0) for r in train)
iters = 1800 / int(sys.argv[2])
print(f'median {step:.2f} s/iter over iterations 11-40, peak memory {mem} MB (rank 0)')
print(f'projected: {step * iters / 3600:.1f} h per epoch, {step * iters * 12 / 3600:.1f} h for 12 epochs '
      f'(plus validation of 200 images per epoch)')
print('loss at logged iterations:', [round(r['loss'], 2) for r in train])
EOF
echo "smoke done; log $LOG"
