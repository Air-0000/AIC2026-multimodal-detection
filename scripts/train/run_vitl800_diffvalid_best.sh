#!/bin/bash
# 对当前 polluted run 的 best_stg1.pth (ep33, 21:55 生成) 跑 10 条件差分验证 (orig + 9档暗光)。
# 注意: 这 10 个 val 配置是 no-TTA (--test-only, 单尺度 800), 与训练期跟踪的 TTA 指标(0.5035)不是同一口径。
# GPU 排队: 等可用 >= 5GB 才跑 (clean/vits800 训练占满时不抢卡)。
PY=/home/constant/miniconda3/envs/deimv2/bin/python
cd /home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
BEST=/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/outputs/deimv2_dinov3_l_aic_vitl800_rdttrack_deepir_aug/best_stg1.pth
W=/home/constant/CodeBuddy/Dev/AIC/logs
PREF=configs/deimv2/deimv2_dinov3_l_aic_vitl800_rdttrack_deepir_aug_val_
LV="day night50 night25 night12 night05 night01 night005 night002 night001 night0005"
SUMMARY=/home/constant/CodeBuddy/Dev/AIC/outputs/vitl800_best_diffvalid_summary.txt

echo "===== VITL800 DIFF-VALID ON BEST ($(date)) =====" >> "$W/vitl800_best_diffvalid.log"
echo "BEST=$BEST" >> "$W/vitl800_best_diffvalid.log"

# 排队: 等 GPU 空闲 >= 5GB (训练占满 13.4GB 时不抢卡)
while true; do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
  FREE=$((TOTAL - USED))
  if [ "$FREE" -ge 5000 ]; then break; fi
  echo "queued diffvalid: free=${FREE}MiB $(date)" >> "$W/vitl800_best_diffvalid.log"
  sleep 60
done
echo "GPU free, starting diff-valid $(date)" >> "$W/vitl800_best_diffvalid.log"

for lv in $LV; do
  cfg=${PREF}${lv}.yml; [ "$lv" = "day" ] && cfg=${PREF}orig.yml
  echo "=== diffvalid $lv $(date) ===" >> "$W/vitl800_best_diffvalid.log"
  $PY train.py -c "$cfg" -r "$BEST" --use-amp --test-only > "$W/vitl800_best_diffvalid_${lv}.log" 2>&1
done
echo "DIFFVALID DONE $(date)" >> "$W/vitl800_best_diffvalid.log"

# 汇总 AP@0.50:0.95(area=all) 与 AP50
/usr/bin/python3 - "$W" "$SUMMARY" <<'PY'
import sys, re, os
W, SUMMARY = sys.argv[1], sys.argv[2]
conds = ["day","night50","night25","night12","night05","night01","night005","night002","night001","night0005"]
lines = ["vitl800 (polluted) best_stg1.pth — differential validation (no-TTA, orig + 9 night)", ""]
lines.append(f"{'cond':10s} {'AP@.5:.95':>10s} {'AP50':>8s}")
for lv in conds:
    log = os.path.join(W, f"vitl800_best_diffvalid_{lv}.log")
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
