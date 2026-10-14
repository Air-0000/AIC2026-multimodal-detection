#!/bin/bash
# =============================================================================
# 阶段脚本重试封装 —— 不改动 run_vitl800_scratch_stage{A,B,C}.sh 本身。
#
# 动机:
#   2026-09-25 15:10 Stage C 在 ep15 被 `cudaErrorLaunchTimeout` 打死(该机 tty2 有
#     active 图形会话, 显示会话与计算任务争用 GPU 时驱动会掐掉长 kernel)。
#   2026-09-25 17:38 重跑, ep15-19 跑完后写 checkpoint0019.pth 时**磁盘写满**
#     (97%, 仅剩 29GB), 留下 2446M 的截断文件; 之后 4 次重试全在 1 秒内失败
#     (bash rc=120, 训练日志 0 字节 —— 因为阶段脚本用 `>` 写日志, 盘满时立刻死),
#     18:19 放弃, 机器空转 7 小时。
#   裸跑 run_vitl800_scratch_all.sh 会让一次瞬态错误终止整条流水线。
#
# 本封装提供:
#   1) 最多 MAX_TRY 次重试 (阶段脚本自身在 last.pth 存在时走 -r 续训, 是"接着跑")
#   2) 磁盘守卫: 空闲 < MIN_FREE_GB 直接放弃并明确报错, 不再空转
#   3) 崩溃现场快照: 失败后把训练日志另存为 <stage>.attempt<N>.crash.log,
#      避免被下一轮的 `>` 覆盖 (2026-09-25 那次现场就是这样丢的)
#
# 用法:
#   bash scripts/train/run_vitl800_scratch_retry.sh C          # 只续 Stage C
#   bash scripts/train/run_vitl800_scratch_retry.sh A B C      # 全流程(带重试)
# 可选环境变量: MAX_TRY=5 RETRY_WAIT=90 MIN_FREE_GB=60
# =============================================================================
set -u
S=/home/constant/CodeBuddy/Dev/AIC/scripts/train
W=/home/constant/CodeBuddy/Dev/AIC/logs
MAX_TRY=${MAX_TRY:-5}
RETRY_WAIT=${RETRY_WAIT:-90}
MIN_FREE_GB=${MIN_FREE_GB:-60}

STAGES="${*:-A B C}"

echo "===== RETRY-WRAPPER START $(date) stages=[$STAGES] max_try=$MAX_TRY =====" \
  >> "$W/scratch_retry.log"

for stage in $STAGES; do
  case $stage in
    A) SH=$S/run_vitl800_scratch_stageA.sh ; TLOG="$W/stageA_rgb_train.log" ;;
    B) SH=$S/run_vitl800_scratch_stageB.sh ; TLOG="$W/stageB_mod_train.log" ;;
    C) SH=$S/run_vitl800_scratch_stageC.sh ; TLOG="$W/stageC_fuse_train.log" ;;
    *) echo "unknown stage '$stage', skip" >> "$W/scratch_retry.log"; continue ;;
  esac

  ok=0
  for i in $(seq 1 "$MAX_TRY"); do
    # 磁盘守卫: 阶段脚本用 `>` 写训练日志, 盘满会立刻失败且日志留空(2026-09-25 的教训)。
    free_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
    if [ "${free_gb:-0}" -lt "${MIN_FREE_GB:-60}" ]; then
      echo "--- stage $stage attempt $i ABORTED: only ${free_gb}GB free (< ${MIN_FREE_GB:-60}GB) $(date) ---" \
        >> "$W/scratch_retry.log"
      exit 1
    fi
    echo "--- stage $stage attempt $i/$MAX_TRY (free ${free_gb}GB) $(date) ---" >> "$W/scratch_retry.log"
    bash "$SH"
    rc=$?
    if [ "$rc" -eq 0 ]; then
      echo "--- stage $stage OK on attempt $i $(date) ---" >> "$W/scratch_retry.log"
      ok=1
      break
    fi
    # 保住崩溃现场: 下一轮 attempt 会用 `>` 覆盖 TLOG, 先快照一份。
    if [ -f "$TLOG" ]; then
      cp "$TLOG" "$W/$(basename "$stage").attempt${i}.crash.log" 2>/dev/null
    fi
    echo "--- stage $stage FAILED rc=$rc attempt $i/$MAX_TRY, log kept as $(basename "$stage").attempt${i}.crash.log, retry in ${RETRY_WAIT}s $(date) ---" \
      >> "$W/scratch_retry.log"
    sleep "$RETRY_WAIT"
  done

  if [ "$ok" -ne 1 ]; then
    echo "===== GIVING UP on stage $stage after $MAX_TRY attempts $(date) =====" \
      >> "$W/scratch_retry.log"
    exit 1
  fi
done

echo "===== RETRY-WRAPPER ALL DONE $(date) =====" >> "$W/scratch_retry.log"
