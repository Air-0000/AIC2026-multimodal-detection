#!/usr/bin/env bash
# 后台串联：损失地形(已修 bug) -> 三模型 val det 落盘 -> 配对 bootstrap
# 全部在 GPU 宿主机跑；日志写到 logs/post_training_analysis.log
set -u
REPO=/home/constant/CodeBuddy/Dev/AIC
PY=/home/constant/miniconda3/envs/deimv2/bin/python
VENDOR=$REPO/vendor/DEIMv2
OUT=$VENDOR/outputs/vitl800_scratch_stageC_fuse
CFG=$VENDOR/configs/deimv2/deimv2_dinov3_l_aic_vitl800_scratch_stageC_fuse.yml
[ -f "$CFG" ] || CFG=$VENDOR/configs/deimv2/deimv2_dinov3_l_aic_vitl800_rdttrack_deepir_aug_val_orig.yml
LOG=$REPO/logs/post_training_analysis.log
mkdir -p "$REPO/logs" "$REPO/outputs/analysis/_dets"
exec > "$LOG" 2>&1
echo "=== start $(date) ==="

cd "$REPO" || exit 1
SCR=$REPO/scripts/analysis

# 1) loss landscape (bug fixed: filter_normalized_dir 现在在 cpu 上构造方向)
BASE="$OUT/best_stg1.pth"; [ -f "$BASE" ] || BASE="$OUT/best_stg1_ep35_AP05265.pth"
TRAJ=""
for e in 0004 0009 0014 0019 0024 0029 0034 0039; do
  [ -f "$OUT/checkpoint$e.pth" ] && TRAJ="$TRAJ $OUT/checkpoint$e.pth"
done
[ -f "$OUT/last.pth" ] && TRAJ="$TRAJ $OUT/last.pth"
echo "=== [1/3] loss landscape (base=$BASE) ==="
$PY "$SCR/loss_landscape.py" --config "$CFG" --base "$BASE" --traj $TRAJ \
  --dir-mode both --grid 11 --rel 0.05 --batch 4 --loader val \
  --interp "$OUT/last.pth" \
  --json "$REPO/outputs/metrics/loss_landscape_stageC.json" \
  --out-png "$REPO/outputs/loss_landscape_stageC.png"
echo "landscape rc=$?"

# 2) dump val200 dets: best / last / swa
echo "=== [2/3] dump val dets ==="
declare -A CK=(
  [best_stg1_ep35_AP05265]="best_stg1_ep35_AP05265.pth"
  [last]="last.pth"
  [swa_stgC_ep24_39]="swa_stgC_ep24_39.pth"
)
for name in best_stg1_ep35_AP05265 last swa_stgC_ep24_39; do
  ck=${CK[$name]}
  echo "--- $name ($ck) ---"
  $PY "$SCR/dump_val_dets.py" --config "$CFG" --ckpt "$OUT/$ck" \
    --out "$REPO/outputs/analysis/_dets/$name.pkl" --tag "$name"
done

# 3) paired bootstrap (same draws -> 抵消图像难度方差)
echo "=== [3/3] paired bootstrap ==="
$PY "$SCR/bootstrap_val_ci.py" --iters 200 --dets-dir "$REPO/outputs/analysis/_dets" \
  --out-json "$REPO/outputs/metrics/val_ci_bootstrap.json" \
  --out-npz "$REPO/outputs/metrics/val_ci_bootstrap.npz"
echo "bootstrap rc=$?"
echo "=== ALL DONE $(date) ==="
