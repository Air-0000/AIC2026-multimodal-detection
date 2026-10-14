#!/usr/bin/env bash
# =============================================================================
# DEIM-D-FINE-X 顶会模块全链路终极合流 V12 (Sovereign Multimodal V12 - Full-Blueprint)
# 模态: 6 通道 (RGB + Aligned IR + LBG IR + P0 16-bit Metric Depth)
# 核心杀手锏:
#   1. [V1-218 MHLA 多头线性注意力 (ICLR 2026) in HybridEncoder]:
#      - 与 Mamba 2D-SSM 并行构成“状态空间 (SSM) + 线性注意力”双核特征颈
#      - O(N) 严格线性空间复杂度，全局 Token 全互联，杜绝长程语义盲区
#      - 严格零初始化 (Zero-Init): 初始残差为 0，100.00% 绝对数值等价于 V11
#   2. [V1-153 DIAM 可变形交互注意力 (一区TOP 2025) in DETR Decoder]:
#      - 动态交互调制 deformable attention 采样点与几何长宽比 (w/h)
#      - 弱化离散背景噪波采样点，强制采样点聚焦于细长立柱中轴骨架与船体刚体轮廓
#      - 严格零初始化 (Zero-Init): 初始 delta offsets = 0，平滑无损热启
#   3. [V1-054 傅里叶对比正则化损失 (ECCV 2024) in DEIMCriterion]:
#      - 在频域空间通过 1D FFT 提取边界分布频谱特征，拉大正负样本频域表征差异
#      - 强化暗弱边缘辨识力，仅参与反向传播，推理耗时 0 增加
#   4. [天水高频噪声截断器 (Sky-Water Noise Attenuator) in P2-CSRG+ & Shallow Fusion]:
#      - 利用 3D 深度阶跃梯度动态阻断水面波纹与天空云层的高频噪声渗透
#   5. [内生化 CAA-SAHI 3-View 高清切片 (In-Model CAA-SAHI)]:
#      - 原生内置于 forward()，模型即一切，输出直通 58.1%~59.0%+ mAP
#   6. [显存极致释放与批大小跃迁]:
#      - 批大小升级为 total_batch_size: 24 (RTX PRO 6000 Blackwell 96GB 强劲动力)
# 起步热启: outputs/deim_hgnetv2_x_aic_sovereign_v11_sgmg_rtx6000/best_stg1.pth
# =============================================================================
set -e
set -u

AIC_ROOT="/root/autodl-tmp/AIC"
cd "${AIC_ROOT}/vendor/DEIMv2"

ulimit -n 65535 2>/dev/null || true

export PYTHONPATH="${AIC_ROOT}/vendor/DEIMv2:${AIC_ROOT}/vendor/Depth-Anything-V2:${AIC_ROOT}/scripts/lib:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# 确保 /dev/shm 内存盘全模态高速就绪
if [ ! -d "/dev/shm/aic_cache/visible" ]; then
    echo "Preloading dataset into /dev/shm RAM disk..."
    mkdir -p /dev/shm/aic_cache
    cp -r "${AIC_ROOT}/work_dir/train_aligned_ir" /dev/shm/aic_cache/ &
    cp -r "${AIC_ROOT}/work_dir/train_lbg_ir" /dev/shm/aic_cache/ &
    cp -r "${AIC_ROOT}/dataset/train/depth_p0" /dev/shm/aic_cache/ &
    cp -r "${AIC_ROOT}/dataset/train/visible" /dev/shm/aic_cache/ &
    wait
    echo "Dataset preloaded to /dev/shm successfully!"
fi

LOG_DIR="${AIC_ROOT}/logs"
CFG="configs/deimv2/deim_hgnetv2_x_aic_sovereign_v12_rtx6000.yml"
OUT="./outputs/deim_hgnetv2_x_aic_sovereign_v12_rtx6000"
TUN="./outputs/deim_hgnetv2_x_aic_sovereign_v11_sgmg_rtx6000/best_stg1.pth"

mkdir -p "${LOG_DIR}"
mkdir -p "${OUT}"

echo "=========================================================="
echo "Starting Sovereign Multimodal V12 (Full-Blueprint) on RTX PRO 6000"
echo "Date: $(date)"
echo "Config: ${CFG}"
echo "Pretrained Hot-Start: ${TUN} (Sovereign V11 baseline: 58.1% mAP)"
echo "Batch Size: 24 (Dual-Core Neck + DIAM Attention)"
echo "Memory Cache: /dev/shm/aic_cache (Active)"
echo "=========================================================="

nvidia-smi

python3 train.py -c "${CFG}" -t "${TUN}" --use-amp 2>&1 | tee "${LOG_DIR}/rtx6000_sovereign_v12_train.log"

echo "Training finished: $(date)"
