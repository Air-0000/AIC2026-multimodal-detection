#!/usr/bin/env bash
# =============================================================================
# IR 对齐实验（报告 §6.1 语义错配靶点）: l4u3_na 基线 vs l4u3_na_sedcn 对齐臂
#
# 设计:
#   * 两臂唯一 diff = aux_se / aux_dcn（RGBGuidedDCN 零初始化可变形对齐 + SE 门控）
#   * 数据 = 干净 _na（heat_na / ir_na, 跳过 align_ir() 假平移）；缓存 deepir_cache_na
#   * 双臂都 from-scratch（weights_path: null, 无 -t）
#   * 训练 50ep 各 ~3.3h（参照 l4u3 09-22 实测 07:36→10:54）
#
# 队列: 等 GPU 空闲 >= 12.5GB（dcng4 评测结束后自然放行），先基线后对齐臂，
#       各自跑完立即接 10 档评测（day + night50..night0005）。
# 日志: logs/l4u3_na_run.log
# =============================================================================
set -u
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
PY=/home/constant/miniconda3/envs/deimv2/bin/python
W=/home/constant/CodeBuddy/Dev/AIC/logs
P=deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir_midboost_l4u3_na
LV="day night50 night25 night12 night05 night01 night005 night002 night001 night0005"

wait_gpu() {
  while true; do
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
    FREE=$((TOTAL - USED))
    [ "$FREE" -ge 12500 ] && return 0
    echo "queued: free=${FREE}MiB $(date)" >> "$W/l4u3_na_run.log"
    sleep 60
  done
}

run_arm() {  # $1 = cfg 后缀 (na / na_sedcn)
  CFG=configs/deimv2/${P}${1}.yml
  OUT=outputs/${P}${1}
  echo "===== TRAIN ARM ${1} $(date) =====" >> "$W/l4u3_na_run.log"
  $PY train.py -c "$CFG" --use-amp > "$W/${P}${1}_train.log" 2>&1
  echo "train rc=$? $(date)" >> "$W/l4u3_na_run.log"
  CKPT=$(ls -t "$OUT"/*.pth 2>/dev/null | head -1)
  echo "ARM ${1} CKPT=$CKPT" >> "$W/l4u3_na_run.log"
  [ -z "$CKPT" ] && { echo "NO CKPT arm ${1}, abort" >> "$W/l4u3_na_run.log"; return 1; }
  for lv in $LV; do
    cfg=configs/deimv2/${P}${1}_val_${lv}.yml
    echo "=== eval ${1} $lv $(date) ===" >> "$W/l4u3_na_run.log"
    $PY train.py -c "$cfg" -r "$CKPT" --use-amp --test-only > "$W/${P}${1}_eval_${lv}.log" 2>&1
    echo "rc=$? $lv" >> "$W/l4u3_na_run.log"
  done
}

echo "===== START L4U3_NA PAIR $(date) =====" >> "$W/l4u3_na_run.log"

# 缓存就绪守卫: 2000 个 .pt 齐 了才开工
want=2000
have=$(ls /home/constant/CodeBuddy/Dev/AIC/work_dir/deepir_cache_na/train 2>/dev/null | wc -l)
while [ "$have" -lt "$want" ]; do
  echo "cache_na incomplete: have=$have want=$want $(date)" >> "$W/l4u3_na_run.log"
  sleep 60
  have=$(ls /home/constant/CodeBuddy/Dev/AIC/work_dir/deepir_cache_na/train 2>/dev/null | wc -l)
done
echo "cache_na ready ($have) $(date)" >> "$W/l4u3_na_run.log"

wait_gpu
run_arm ""          # 基线臂
wait_gpu
run_arm "_sedcn"    # 对齐臂
echo "===== ALL DONE L4U3_NA PAIR $(date) =====" >> "$W/l4u3_na_run.log"
