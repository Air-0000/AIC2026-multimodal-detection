#!/usr/bin/env bash
# =============================================================================
# DEIM-D-FINE-X 物理因子终极内生化旗舰 V11 (Sovereign Multimodal V11 - SGMG + P2-CSRG+)
# 模态: 6 通道 (RGB + Aligned IR + LBG IR + P0 16-bit Metric Depth)
# 核心杀手锏:
#   1. [3D 场景几何连续深度门控 (SceneGeometryContextGate - SGMG)]:
#      - 从底半部深度连续统计量调制 12 类前向 logits，端到端自适应压制陆地/水面漂浮虚警
#      - 严格零初始化 (Zero-Init): 初始步完全等价于 V10
#   2. [P2 (160x160) 跨尺度小波保峰与立柱几何直通 (P2-CSRG+ in HybridEncoder)]:
#      - 11x1 垂直条带大卷积捕捉 3px 极细立柱
#      - 2D Haar 小波 (MWHL) 提取 HL 垂直高频，防止 160x160 -> 80x80 池化细节抹平
#      - 空间自适应高频门控 (CSR 机制)
#      - 零初始化专用输出投影 (proj_v10_1) 杜绝初始扰动
# 起步热启: outputs/deim_hgnetv2_x_aic_sovereign_v10_p2_res_rtx6000/best_stg1.pth
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
CFG="configs/deimv2/deim_hgnetv2_x_aic_sovereign_v11_sgmg_rtx6000.yml"
OUT="./outputs/deim_hgnetv2_x_aic_sovereign_v11_sgmg_rtx6000"
TUN="./outputs/deim_hgnetv2_x_aic_sovereign_v10_p2_res_rtx6000/best_stg1.pth"

mkdir -p "${LOG_DIR}"
mkdir -p "${OUT}"

echo "=========================================================="
echo "Starting Sovereign Multimodal V11 (SGMG + P2-CSRG+) on RTX PRO 6000"
echo "Date: $(date)"
echo "Config: ${CFG}"
echo "Pretrained Hot-Start: ${TUN} (Sovereign V10 baseline)"
echo "Memory Cache: /dev/shm/aic_cache (Active)"
echo "=========================================================="

nvidia-smi

python3 train.py -c "${CFG}" -t "${TUN}" --use-amp 2>&1 | tee "${LOG_DIR}/rtx6000_sovereign_v11_sgmg_train.log"

echo "Training finished: $(date)"
