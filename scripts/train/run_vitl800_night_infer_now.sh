#!/bin/bash
# 对 polluted vitl800 的 best_stg1.pth 立即跑 10 条件推理 (orig + 9档暗光, no-TTA, --test-only)。
# GPU 此时已空闲 (clean 训练被临时暂停), 直接跑, 不排队。
PY=/home/constant/miniconda3/envs/deimv2/bin/python
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
BEST=/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/outputs/deimv2_dinov3_l_aic_vitl800_rdttrack_deepir_aug/best_stg1.pth
W=/home/constant/CodeBuddy/Dev/AIC/logs
PREF=configs/deimv2/deimv2_dinov3_l_aic_vitl800_rdttrack_deepir_aug_val_
LV="orig night0005 night001 night002 night005 night01 night05 night12 night25 night50"
SUMMARY=/home/constant/CodeBuddy/Dev/AIC/outputs/metrics/vitl800_best_night_infer.txt

echo "===== VITL800 NIGHT INFER ON BEST ($BEST) $(date) =====" >> "$W/vitl800_best_night_infer.log"
for lv in $LV; do
  cfg=${PREF}${lv}.yml
  echo "=== infer $lv $(date) ===" >> "$W/vitl800_best_night_infer.log"
  $PY train.py -c "$cfg" -r "$BEST" --use-amp --test-only > "$W/vitl800_best_night_infer_${lv}.log" 2>&1
done
echo "NIGHT INFER DONE $(date)" >> "$W/vitl800_best_night_infer.log"

# 汇总 AP@0.50:0.95(area=all) 与 AP50
$PY - "$W" "$SUMMARY" <<'PY'
import sys, re, os
W, SUMMARY = sys.argv[1], sys.argv[2]
conds = ["orig","night0005","night001","night002","night005","night01","night05","night12","night25","night50"]
lines = ["vitl800 (polluted) best_stg1.pth — night inference (no-TTA, single-scale 800)", ""]
lines.append(f"{'cond':10s} {'AP@.5:.95':>10s} {'AP50':>8s}")
for lv in conds:
    log = os.path.join(W, f"vitl800_best_night_infer_{lv}.log")
    if not os.path.exists(log):
        lines.append(f"{lv:10s} {'MISSING':>10s} {'':>8s}"); continue
    t = open(log).read()
    m = re.search(r'IoU=0\.50:0\.95\s*\|\s*area=\s*all\s*\|\s*maxDets=100\s*\]\s*=\s*([0-9.]+)', t)
    m50 = re.search(r'IoU=0\.50\s*\|\s*area=\s*all\s*\|\s*maxDets=100\s*\]\s*=\s*([0-9.]+)', t)
    a = m.group(1) if m else '?'; a50 = m50.group(1) if m50 else '?'
    lines.append(f"{lv:10s} {a:>10s} {a50:>8s}")
open(SUMMARY, "w").write("\n".join(lines)+"\n")
print("\n".join(lines))
PY
