#!/bin/bash
# 磁盘清理前的体检:
#   1) Stage C 各 ckpt 的 mtime/大小 (找出被截断的)
#   2) 验证 last.pth 能否正常 torch.load (决定能否续训)
#   3) 统计"旧实验"的中间 checkpoint 总量 (可回收空间)
set -u
O=/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/outputs

echo "###### 1) Stage C 文件清单 ######"
ls -la --block-size=M --time-style=long-iso "$O/vitl800_scratch_stageC_fuse/"*.pth | awk '{print $5"\t"$6" "$7"\t"$8}'

echo
echo "###### 2) Stage C last.pth 完整性 ######"
/home/constant/miniconda3/envs/deimv2/bin/python - <<'PYEOF'
import torch, os
p = "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/outputs/vitl800_scratch_stageC_fuse/last.pth"
print("size = %.2f GB" % (os.path.getsize(p) / 1024**3))
try:
    ck = torch.load(p, map_location="cpu", weights_only=False)
    print("top-level keys:", sorted(ck.keys()))
    print("last_epoch   :", ck.get("last_epoch"))
    bs = ck.get("best_stat")
    print("best_stat    :", bs)
    sd = ck.get("model")
    if isinstance(sd, dict):
        print("model params : %d keys" % len(sd))
    em = ck.get("ema")
    if isinstance(em, dict) and "module" in em:
        print("ema params   : %d keys" % len(em["module"]))
    print("VERDICT: last.pth LOADABLE")
except Exception as e:
    print("VERDICT: last.pth CORRUPT ->", type(e).__name__, e)
PYEOF

echo
echo "###### 3) 旧实验可回收的中间 checkpoint ######"
find "$O" -maxdepth 2 -name 'checkpoint*.pth' -size +100M \
     ! -path "$O/vitl800_scratch_*" -printf '%s\n' \
  | awk '{s+=$1; n++} END {printf "count=%d  total=%.1f GB\n", n, s/1073741824}'

echo
echo "###### 4) 本次 scratch 三阶段的中间 checkpoint ######"
find "$O" -maxdepth 2 -name 'checkpoint*.pth' -size +100M \
     -path "$O/vitl800_scratch_*" -printf '%s\n' \
  | awk '{s+=$1; n++} END {printf "count=%d  total=%.1f GB\n", n, s/1073741824}'
