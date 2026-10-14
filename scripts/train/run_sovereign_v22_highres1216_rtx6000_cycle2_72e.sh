#!/bin/bash
# =============================================================================
# Sovereign V22 极值高分辨率 (1216x672) Cycle 2 (第二周期 72-Epoch) 进阶精炼启动脚本
# 算力平台: NVIDIA RTX PRO 6000 Blackwell (96GB VRAM, 满血)
# =============================================================================
set -e

STAGE1_OUTPUT="/root/autodl-tmp/AIC/vendor/DEIMv2/outputs/sovereign_v22_highres1216_rtx6000_full72e"
CYCLE1_BEST="${STAGE1_OUTPUT}/best_stg2.pth"
if [ ! -f "$CYCLE1_BEST" ]; then
    CYCLE1_BEST="${STAGE1_OUTPUT}/best_stg1.pth"
fi

if [ ! -f "$CYCLE1_BEST" ]; then
    echo "[-] Error: Cycle 1 checkpoint not found at $CYCLE1_BEST"
    exit 1
fi

echo "[+] Found Cycle 1 best checkpoint: $CYCLE1_BEST"
mkdir -p /root/autodl-tmp/AIC/weight
CYCLE1_APEX="/root/autodl-tmp/AIC/weight/sovereign_v22_cycle1_apex.pth"

# 1. 提取并归档 Cycle 1 纯净 FP16 权威权重
echo "[+] Archiving Cycle 1 pure FP16 weights to $CYCLE1_APEX..."
/root/miniconda3/bin/python -c "
import torch
ckpt = torch.load('$CYCLE1_BEST', map_location='cpu')
state = ckpt.get('ema', {}).get('module', ckpt.get('model', ckpt))
clean = {k: v.half() if v.is_floating_point() else v for k, v in state.items()}
torch.save(clean, '$CYCLE1_APEX')
print(f'[+] Successfully saved Cycle 1 pure FP16 weights: {len(clean)} keys')
"

# 2. 启动 Cycle 2 (第二周期 72-Epoch) 进阶精炼
echo "[+] Starting Sovereign V22 Cycle 2 (72 Epochs) on NVIDIA RTX PRO 6000..."
cd /root/autodl-tmp/AIC/vendor/DEIMv2
mkdir -p /root/autodl-tmp/AIC/logs

export CUDA_VISIBLE_DEVICES=0
export NCCL_P2P_DISABLE=1
export PYTHONUNBUFFERED=1

nohup /root/miniconda3/bin/python train.py \
    -c configs/deimv2/sovereign_v22_highres1216_rtx6000_cycle2_72e.yml \
    -t "$CYCLE1_BEST" \
    --use-amp \
    --seed 42 \
    > /root/autodl-tmp/AIC/logs/run_v22_1216_cycle2_72e.log 2>&1 &

PID=$!
echo "[+] Sovereign V22 Cycle 2 training launched successfully! PID: $PID"
echo "[+] Logs streaming to: /root/autodl-tmp/AIC/logs/run_v22_1216_cycle2_72e.log"
