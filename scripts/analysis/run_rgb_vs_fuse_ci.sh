#!/bin/bash
# 纯 RGB (Stage A) vs 融合 (Stage C) 的配对 bootstrap。
# 必须在 ARXw4T (run_post_training_analysis.sh) 跑完后、GPU 空闲时执行：
#   - 等 GPU >= 12GB 空闲连续 2 次 + ARXw4T 进程消失 + 其 3 份 dets 已落盘
#   - dump Stage A (纯 RGB) 的 val200 dets
#   - 把 Stage C 的 best/last/swa dets 拷到干净目录，做聚焦配对 bootstrap
# 配对 bootstrap 用「同图同抽取」抵消图像难度方差，给出 fusion-gain 的 95% CI。
set -u
REPO=/home/constant/CodeBuddy/Dev/AIC
VENDOR=$REPO/vendor/DEIMv2
PY=/home/constant/miniconda3/envs/deimv2/bin/python
LOG=$REPO/logs/rgb_vs_fuse_ci.log
STAGEA_CFG=$VENDOR/configs/deimv2/deimv2_dinov3_l_aic_vitl800_scratch_stageA_rgb.yml
STAGEA_CKPT=$VENDOR/outputs/vitl800_scratch_stageA_rgb/best_stg1.pth
SRC=$REPO/outputs/analysis/_dets
OUTDIR=$REPO/outputs/analysis/_dets_rgb_vs_fuse
mkdir -p "$OUTDIR"
echo "===== START rgb_vs_fuse $(date) =====" >> "$LOG"

# ---- 等 ARXw4T 结束 + GPU 空闲 ----
cnt=0; waited=0
while true; do
  BUSY=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  FREE=$((TOTAL - BUSY))
  PREV=$(pgrep -f run_post_training_analysis.sh | wc -l)
  DONE=$(ls "$SRC"/best_stg1_ep35_AP05265.pkl "$SRC"/last.pkl "$SRC"/swa_stgC_ep24_39.pkl 2>/dev/null | wc -l)
  if [ "$FREE" -ge 12000 ] && [ "$PREV" -eq 0 ] && [ "$DONE" -ge 3 ]; then
    cnt=$((cnt+1)); [ "$cnt" -ge 2 ] && break
  else
    cnt=0
  fi
  sleep 30; waited=$((waited+30))
  if [ "$waited" -ge 5400 ]; then echo "TIMEOUT waiting for ARXw4T @ $waited s" >> "$LOG"; exit 1; fi
done
echo "ARXw4T done, GPU free. $(date)" >> "$LOG"

# ---- 拷贝 Stage C dets 到干净目录 ----
cp "$SRC"/best_stg1_ep35_AP05265.pkl "$OUTDIR"/ 2>>"$LOG"
cp "$SRC"/last.pkl "$OUTDIR"/ 2>>"$LOG"
cp "$SRC"/swa_stgC_ep24_39.pkl "$OUTDIR"/ 2>>"$LOG"
echo "copied Stage C dets -> $OUTDIR" >> "$LOG"

# ---- dump 纯 RGB (Stage A) ----
echo "=== [1/2] dump Stage A (pure RGB) ===" >> "$LOG"
$PY $REPO/scripts/analysis/dump_val_dets.py --config "$STAGEA_CFG" --ckpt "$STAGEA_CKPT" \
  --out "$OUTDIR"/stageA_rgb.pkl --tag stageA_rgb >> "$LOG" 2>&1
echo "stageA dump rc=$?" >> "$LOG"

# ---- 聚焦配对 bootstrap ----
echo "=== [2/2] paired bootstrap (rgb vs fuse) ===" >> "$LOG"
$PY $REPO/scripts/analysis/bootstrap_val_ci.py --iters 200 \
  --dets-dir "$OUTDIR" \
  --out-json "$REPO"/outputs/metrics/val_ci_rgb_vs_fuse.json \
  --out-npz "$REPO"/outputs/metrics/val_ci_rgb_vs_fuse.npz >> "$LOG" 2>&1
echo "bootstrap rc=$?" >> "$LOG"
echo "=== ALL DONE $(date) ===" >> "$LOG"
