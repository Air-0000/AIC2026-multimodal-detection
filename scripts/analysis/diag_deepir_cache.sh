#!/usr/bin/env bash
# Diagnose the missing depth/IR feature cache that kills Stage C.
set -u
R=/home/constant/CodeBuddy/Dev/AIC
cd "$R" || exit 1

echo "================ deepir_cache 目录结构 ================"
ls -la work_dir/deepir_cache/ 2>/dev/null || echo "(无 deepir_cache)"

echo
echo "================ 各子目录文件数 ================"
for d in work_dir/deepir_cache/*/; do
  [ -d "$d" ] || continue
  n=$(ls "$d" 2>/dev/null | wc -l)
  printf "%-50s %6d 个文件\n" "$d" "$n"
done

echo
echo "================ 训练集规模 ================"
printf "train images : %s\n" "$(ls work_dir/deimv2_dataset/train/images 2>/dev/null | wc -l)"
printf "train anns   : %s\n" "$(python3 -c "import json;print(len(json.load(open('work_dir/deimv2_dataset/train/annotations/instances_train.json'))['images']))" 2>/dev/null || echo '?')"

echo
echo "================ 目标文件 003788.pt ================"
ls -la work_dir/deepir_cache/train/003788.pt 2>&1 | tail -2

echo
echo "================ train cache 编号范围 / 连续性 ================"
python3 - <<'PY'
import os, re
d = "/home/constant/CodeBuddy/Dev/AIC/work_dir/deepir_cache/train"
if not os.path.isdir(d):
    print("no dir", d); raise SystemExit
names = sorted(os.listdir(d))
ids = sorted(int(m.group(1)) for f in names if (m := re.match(r"^(\d+)\.pt$", f)))
print("文件总数 :", len(names))
print("编号 min :", ids[0] if ids else None)
print("编号 max :", ids[-1] if ids else None)
if ids:
    full = set(range(ids[0], ids[-1] + 1))
    miss = sorted(full - set(ids))
    print("区间内应有 :", ids[-1] - ids[0] + 1)
    print("区间内实有 :", len(ids))
    print("区间内缺失 :", len(miss))
    print("缺失样例   :", miss[:12] if miss else "(无)")
    # 是否有 003788 附近
    near = [i for i in miss if 3780 <= i <= 3800]
    print("3780-3800 区间缺失:", near if near else "(无)")
PY

echo
echo "================ 其他 cache 根目录 ================"
ls -d work_dir/*cache* 2>/dev/null

echo
echo "================ 磁盘 ================"
df -h / | tail -1
