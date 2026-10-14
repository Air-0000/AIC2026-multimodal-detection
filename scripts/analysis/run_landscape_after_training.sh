#!/usr/bin/env bash
# 等 GPU 真正空出来，再跑 loss landscape（约 10-20 分钟），然后刷新所有静态图。
# v2 相对 v1 的改动：
#   * base 用**最终最佳** best_stg1.pth（v1 写死 ep18 快照，而 C 段后面已经刷到 ep28）；
#   * 轨迹点按每 5 epoch 取样，存在就投，不存在自动跳过（v1 写死 4/14/24）；
#   * 跑完 landscape 顺手刷新轨迹方向 JSON 与训练路径图。
#
# 为什么必须排队：
#   * Stage C 正在用 ~14.8/16.3 GiB，之后还要跑 10 档评测（同样是 train.py 进程）；
#   * 16GB 卡上再塞一个 384M 模型做前向会直接 OOM，或把训练拖慢到 timeout（之前就因
#     cudaErrorLaunchTimeout 挂过一次）。
# 判据：连续 2 次检查（间隔 60s）都满足
#   ① 没有任何 train.py 进程在跑  ② nvidia-smi 空闲显存 >= 13 GiB
# 才启动。最长等 12 小时。
#
# 用法（在宿主机上）：nohup bash scripts/analysis/run_landscape_after_training.sh >/dev/null 2>&1 &
set -u

REPO=/home/constant/CodeBuddy/Dev/AIC
PY=/home/constant/miniconda3/envs/deimv2/bin/python
LOG="$REPO/logs/landscape_stageC.log"
OUT="$REPO/vendor/DEIMv2/outputs/vitl800_scratch_stageC_fuse"
CFG="$REPO/vendor/DEIMv2/configs/deimv2/deimv2_dinov3_l_aic_vitl800_scratch_stageC_fuse.yml"
mkdir -p "$REPO/logs" "$REPO/outputs"

stamp() { date '+%F %T'; }
say() { echo "[$(stamp)] $*" >>"$LOG"; }

say "waiter v2 启动 pid=$$  (等 GPU 空闲，最长 12h)"

DEADLINE=$(( $(date +%s) + 12*3600 ))
ok=0
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    busy=$(pgrep -fc "train\.py" || true)
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "${busy:-0}" = "0" ] && [ "${free_mib:-0}" -ge 13000 ]; then
        ok=$((ok+1))
        say "空闲候选 ${ok}/2  (busy=$busy free=${free_mib}MiB)"
        [ "$ok" -ge 2 ] && break
    else
        ok=0
        say "忙 (train.py 进程 $busy, 空闲 ${free_mib}MiB) -> 继续等"
    fi
    sleep 60
done

if [ "$ok" -lt 2 ]; then
    say "超时退出，没拿到 GPU"
    exit 1
fi

cd "$REPO" || exit 1

BASE="$OUT/best_stg1.pth"
[ -f "$BASE" ] || BASE="$OUT/best_stg1_ep18_AP05081.pth"
say "base = $BASE"
$PY - "$BASE" >>"$LOG" 2>&1 <<'PY'
import sys, torch
d = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print("base 内含 last_epoch =", d.get("last_epoch"),
      " best_stat =", d.get("best_stat"))
PY

TRAJ=""
for e in 0004 0009 0014 0019 0024 0029 0034 0039; do
    [ -f "$OUT/checkpoint$e.pth" ] && TRAJ="$TRAJ $OUT/checkpoint$e.pth"
done
[ -f "$OUT/last.pth" ] && TRAJ="$TRAJ $OUT/last.pth"
say "轨迹点：$TRAJ"

say "开始跑 loss landscape"
# shellcheck disable=SC2086
$PY scripts/analysis/loss_landscape.py \
    --config "$CFG" --base "$BASE" \
    --traj $TRAJ \
    --dir-mode both --grid 11 --rel 0.05 --batch 4 --loader val \
    --interp "$OUT/last.pth" \
    --json outputs/metrics/loss_landscape_stageC.json \
    --out-png outputs/loss_landscape_stageC.png >>"$LOG" 2>&1
say "landscape rc=$?"

say "刷新轨迹方向 JSON"
bash scripts/analysis/run_path_direction_all.sh >>"$LOG" 2>&1
say "path_direction rc=$?"

say "重画训练路径图（含 C 的最终 epoch）"
$PY scripts/analysis/plot_train_path.py --out outputs/figures/train_path_A_B_C.png \
    --stage A:vitl800_scratch_stageA_rgb B:vitl800_scratch_stageB_mod C:vitl800_scratch_stageC_fuse \
    >>"$LOG" 2>&1
say "全部完成"
