#!/usr/bin/env bash
# ============================================================================
# AIC 2026 — Linux 服务器一键部署脚本
#
# 使用方法:
#   bash setup_linux.sh                      # 默认安装
#   bash setup_linux.sh --conda              # 使用 Conda 环境（推荐服务器）
#   bash setup_linux.sh --venv               # 使用 venv 环境
#   bash setup_linux.sh --data-dir /path     # 指定数据路径
#   bash setup_linux.sh --cuda 11.8          # 指定 CUDA 版本
#   bash setup_linux.sh --help               # 查看帮助
#
# 适用系统: Ubuntu 20.04+, CentOS 7+, 或其他 Linux 发行版
# ============================================================================

set -euo pipefail

# ============================== 颜色定义 ==============================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

info()  { echo -e "${BLUE}[INFO]${NC}  $1"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
err()   { echo -e "${RED}[ERROR]${NC} $1"; }
step()  { echo -e "\n${CYAN}━━━ $1 ━━━${NC}"; }

# ============================== 默认参数 ==============================
USE_CONDA=false
USE_VENV=false
DATA_DIR=""
CUDA_VERSION=""  # 自动检测

# ============================== 参数解析 ==============================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --conda)    USE_CONDA=true; shift ;;
        --venv)     USE_VENV=true; shift ;;
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        --cuda)     CUDA_VERSION="$2"; shift 2 ;;
        --help|-h)
            echo "用法: bash setup_linux.sh [选项]"
            echo ""
            echo "选项:"
            echo "  --conda            使用 Conda 环境（推荐服务器大项目）"
            echo "  --venv             使用 Python venv 环境（默认，轻量）"
            echo "  --data-dir DIR     指定竞赛数据路径"
            echo "  --cuda VERSION     指定 CUDA 版本 (如 11.8, 12.1)"
            echo "  --help, -h         显示此帮助"
            exit 0
            ;;
        *) err "未知参数: $1"; exit 1 ;;
    esac
done

# 默认使用 venv（如果都没指定）
if ! $USE_CONDA && ! $USE_VENV; then
    USE_VENV=true
fi

# ============================== 欢迎 ==============================
echo ""
echo -e "${CYAN}╔════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║   AIC 2026 — Linux 服务器一键部署脚本     ║${NC}"
echo -e "${CYAN}║   Track 1: M2I2HA Baseline                ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════╝${NC}"
echo ""

# ============================== 1. 系统检查 ==============================
step "1/7  系统环境检查"

# 检查操作系统
if [[ ! -f /etc/os-release ]]; then
    warn "无法确定操作系统类型，脚本主要适配 Ubuntu/CentOS"
else
    source /etc/os-release
    ok "操作系统: $NAME $VERSION_ID"
fi

# 检查 Python
if command -v python3 &>/dev/null; then
    PYTHON=$(command -v python3)
elif command -v python &>/dev/null; then
    PYTHON=$(command -v python)
else
    err "未找到 Python，请先安装 Python >= 3.10"
    info "Ubuntu: sudo apt install python3 python3-pip python3-venv -y"
    info "CentOS: sudo yum install python3 python3-pip -y"
    exit 1
fi

PY_VER=$($PYTHON --version 2>&1 | grep -oP '\d+\.\d+')
PY_MAJOR=$(echo $PY_VER | cut -d. -f1)
PY_MINOR=$(echo $PY_VER | cut -d. -f2)

if [[ "$PY_MAJOR" -lt 3 ]] || [[ "$PY_MAJOR" -eq 3 && "$PY_MINOR" -lt 10 ]]; then
    err "需要 Python >= 3.10，当前版本: $PY_VER"
    exit 1
fi
ok "Python: $($PYTHON --version 2>&1)"

# 检查 CUDA
if command -v nvidia-smi &>/dev/null; then
    CUDA_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -n1)
    GPU_NAME=$(echo "$CUDA_INFO" | cut -d, -f1 | xargs)
    GPU_MEM=$(echo "$CUDA_INFO" | cut -d, -f2 | xargs)
    ok "GPU: $GPU_NAME ($GPU_MEM)"

    # 检测 CUDA 版本
    if [[ -z "$CUDA_VERSION" ]]; then
        CUDA_VERSION=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[\d.]+' | head -n1)
    fi
    if [[ -n "$CUDA_VERSION" ]]; then
        ok "CUDA: $CUDA_VERSION"
    fi

    # 估算显存（MB/GB 统一转 GB）
    GPU_MEM_GB=$(echo "$GPU_MEM" | grep -oP '[\d.]+')
    if echo "$GPU_MEM" | grep -qi "mib\|mb"; then
        GPU_MEM_GB=$(echo "scale=1; $GPU_MEM_GB / 1024" | bc)
    fi

    if (( $(echo "$GPU_MEM_GB < 10" | bc -l) )); then
        warn "检测到显存约 ${GPU_MEM_GB}GB，建议使用 net_scale=n, batch=8"
    elif (( $(echo "$GPU_MEM_GB < 20" | bc -l) )); then
        info "显存约 ${GPU_MEM_GB}GB，可运行 net_scale=s, batch=16"
    else
        info "显存约 ${GPU_MEM_GB}GB，可以全速运行！"
    fi
else
    warn "未检测到 NVIDIA GPU，训练将无法运行"
    warn "请先安装 NVIDIA 驱动和 CUDA：https://developer.nvidia.com/cuda-downloads"
fi

# 检查 Git
if command -v git &>/dev/null; then
    ok "Git: $(git --version 2>&1)"
else
    warn "未安装 Git，尝试自动安装..."
    if command -v apt &>/dev/null; then
        sudo apt update && sudo apt install git -y
    elif command -v yum &>/dev/null; then
        sudo yum install git -y
    else
        err "请手动安装 Git"
        exit 1
    fi
fi

# 检查磁盘空间
PROJECT_DIR=$(cd "$(dirname "$0")" && pwd)
AVAIL_DISK=$(df -BG "$PROJECT_DIR" 2>/dev/null | tail -1 | awk '{print $4}' | sed 's/G//')
info "项目目录: $PROJECT_DIR"
info "可用磁盘: ${AVAIL_DISK}GB"
if (( AVAIL_DISK < 20 )); then
    warn "磁盘空间不足 20GB，训练时数据集和模型文件可能不够用"
fi

# ============================== 2. 环境创建 ==============================
step "2/7  创建 Python 环境"

ENV_NAME="aic2026"

if $USE_CONDA; then
    # ---------- Conda ----------
    if ! command -v conda &>/dev/null; then
        info "未检测到 Conda，尝试自动安装 Miniconda..."
        MINICONDA_PREFIX="$HOME/miniconda3"
        if [[ ! -d "$MINICONDA_PREFIX" ]]; then
            wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
            bash /tmp/miniconda.sh -b -p "$MINICONDA_PREFIX"
            rm /tmp/miniconda.sh
            export PATH="$MINICONDA_PREFIX/bin:$PATH"
            ok "Miniconda 已安装到 $MINICONDA_PREFIX"
        else
            export PATH="$MINICONDA_PREFIX/bin:$PATH"
            ok "Miniconda 已存在"
        fi
    fi

    # 创建 Conda 环境
    if conda env list | grep -q "^$ENV_NAME "; then
        info "Conda 环境 '$ENV_NAME' 已存在，直接激活"
    else
        info "创建 Conda 环境 '$ENV_NAME' (Python 3.10)..."
        conda create -y -n "$ENV_NAME" python=3.10
        ok "Conda 环境已创建"
    fi

    # 激活
    CONDA_BASE=$(conda info --base)
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
    PYTHON=$(which python)
    ok "激活环境: $ENV_NAME"

    # 如果指定了 CUDA 版本，安装对应 PyTorch
    if [[ -n "$CUDA_VERSION" ]]; then
        case "$CUDA_VERSION" in
            11.8) TORCH_INDEX="https://download.pytorch.org/whl/cu118" ;;
            12.1) TORCH_INDEX="https://download.pytorch.org/whl/cu121" ;;
            12.4) TORCH_INDEX="https://download.pytorch.org/whl/cu124" ;;
            *)    TORCH_INDEX="" ;;  # 让 pip 自动选
        esac
    else
        TORCH_INDEX=""
    fi

else
    # ---------- venv ----------
    VENV_DIR="$PROJECT_DIR/.venv"

    if [[ -d "$VENV_DIR" ]]; then
        info "venv 已存在: $VENV_DIR"
    else
        info "创建 venv: $VENV_DIR"
        $PYTHON -m venv "$VENV_DIR"
        ok "venv 已创建"
    fi

    source "$VENV_DIR/bin/activate"
    PYTHON=$(which python)
    ok "激活环境: $(python --version 2>&1) @ $VENV_DIR"

    # 升级 pip
    pip install --upgrade pip setuptools wheel -q

    # venv 下 PyTorch 自动选 CUDA 版本
    TORCH_INDEX=""
fi

# ============================== 3. 安装 PyTorch ==============================
step "3/7  安装 PyTorch"

if [[ -n "$TORCH_INDEX" ]]; then
    info "安装 PyTorch (CUDA $CUDA_VERSION)..."
    pip install torch torchvision --index-url "$TORCH_INDEX" -q
else
    info "安装 PyTorch (自动选择 CUDA 版本)..."
    pip install torch torchvision -q
fi

# 验证 PyTorch 和 CUDA
PYTORCH_CUDA=$($PYTHON -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
if [[ "$PYTORCH_CUDA" == "True" ]]; then
    CUDA_VER=$($PYTHON -c "import torch; print(torch.version.cuda)")
    GPU_COUNT=$($PYTHON -c "import torch; print(torch.cuda.device_count())")
    GPU_NAME=$($PYTHON -c "import torch; print(torch.cuda.get_device_name(0))")
    ok "PyTorch CUDA 可用: $GPU_NAME × $GPU_COUNT (CUDA $CUDA_VER)"
else
    warn "PyTorch 未检测到 CUDA，训练将退回到 CPU（极慢）"
    warn "建议: bash setup_linux.sh --cuda 12.1"
fi

# ============================== 4. 安装项目依赖 ==============================
step "4/7  安装项目依赖"

info "安装依赖包..."
pip install -r "$PROJECT_DIR/requirements.txt" -q

info "安装本包 (aic2026)..."
pip install -e "$PROJECT_DIR" -q
ok "aic2026 包安装完成"

# ============================== 5. 安装 M2I2HA ==============================
step "5/7  安装 M2I2HA (官方代码库)"

# 检查是否已安装
if $PYTHON -c "import machine_learning" 2>/dev/null; then
    ML_VER=$($PYTHON -c "import machine_learning; print(getattr(machine_learning, '__version__', '已安装'))")
    ok "M2I2HA 已安装: $ML_VER"
else
    info "克隆 M2I2HA 官方仓库..."
    VENDOR_DIR="$PROJECT_DIR/vendor"
    mkdir -p "$VENDOR_DIR"

    if [[ -d "$VENDOR_DIR/machine_learning" ]]; then
        info "vendor/machine_learning 已存在，更新中..."
        cd "$VENDOR_DIR/machine_learning" && git pull
    else
        git clone https://github.com/WSYANGSX/machine_learning.git "$VENDOR_DIR/machine_learning"
    fi

    info "安装 M2I2HA (本地开发模式)..."
    pip install -e "$VENDOR_DIR/machine_learning" -q

    if $PYTHON -c "import machine_learning" 2>/dev/null; then
        ok "M2I2HA 安装成功"
    else
        warn "M2I2HA 安装可能有问题，请检查 vendor/machine_learning 目录"
    fi
fi

# ============================== 6. 数据目录 & 配置 ==============================
step "6/7  数据目录配置"

# 获取数据路径
if [[ -z "$DATA_DIR" ]]; then
    # 尝试常见位置
    if [[ -d "$PROJECT_DIR/data/visible/train" ]]; then
        DATA_DIR="$PROJECT_DIR/data"
        info "检测到数据已存在: $DATA_DIR"
    else
        # 提示用户输入
        echo ""
        warn "未指定数据路径 (--data-dir)"
        echo -e "请输入竞赛数据所在的目录路径（例如 /data/aic2026），"
        echo -e "或留空稍后手动配置："
        read -r -p "> " INPUT_DIR
        if [[ -n "$INPUT_DIR" ]]; then
            DATA_DIR="$INPUT_DIR"
        fi
    fi
fi

# 创建目录结构（如果指定了路径且不存在）
if [[ -n "$DATA_DIR" ]]; then
    # 创建目录结构
    for split in train val test; do
        mkdir -p "$DATA_DIR/visible/$split"
        mkdir -p "$DATA_DIR/infrared/$split"
        mkdir -p "$DATA_DIR/depth/$split"   # Track 2 备用
        mkdir -p "$DATA_DIR/labels/$split"
    done
    ok "数据目录结构已创建: $DATA_DIR"

    # 写入数据集配置
    DATASET_CFG="$PROJECT_DIR/configs/aic_baseline.yaml"
    if [[ -f "$DATASET_CFG" ]]; then
        # 备份原文件
        cp "$DATASET_CFG" "${DATASET_CFG}.bak" 2>/dev/null || true
        # 替换 path 字段
        sed -i "s|^path:.*|path: $DATA_DIR|" "$DATASET_CFG"
        ok "数据集配置已更新: $DATASET_CFG -> path: $DATA_DIR"
    fi

    # 同时更新 triplet 配置（Track 2）
    TRIPLET_CFG="$PROJECT_DIR/configs/aic_triplet.yaml"
    if [[ -f "$TRIPLET_CFG" ]]; then
        sed -i "s|^path:.*|path: $DATA_DIR|" "$TRIPLET_CFG"
    fi
else
    warn "跳过了数据目录配置"
    info "部署后请手动编辑: configs/aic_baseline.yaml"
    info "将 path 改为实际数据路径"
fi

# 更新训练脚本
TRAIN_SCRIPT="$PROJECT_DIR/src/aic2026/scripts/train_baseline.sh"
if [[ -f "$TRAIN_SCRIPT" ]]; then
    if [[ -n "$DATA_DIR" ]]; then
        # 如果有数据路径，自动更新到脚本中
        sed -i "s|DATASET_CFG=\"/path/to/aic2026/|DATASET_CFG=\"$PROJECT_DIR/configs/|" "$TRAIN_SCRIPT" 2>/dev/null || true
    fi

    # 根据显存调整 batch_size
    if command -v nvidia-smi &>/dev/null; then
        GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -n1 | grep -oP '[\d.]+')
        if [[ -n "$GPU_MEM_MB" ]]; then
            if (( $(echo "$GPU_MEM_MB < 12000" | bc -l) )); then
                sed -i 's/BATCH_SIZE=16/BATCH_SIZE=8/' "$TRAIN_SCRIPT"
                sed -i 's/NET_SCALE="s"/NET_SCALE="n"/' "$TRAIN_SCRIPT"
                info "显存 <12GB，已自动调优: batch=8, net_scale=n"
            elif (( $(echo "$GPU_MEM_MB < 20000" | bc -l) )); then
                sed -i 's/BATCH_SIZE=16/BATCH_SIZE=16/' "$TRAIN_SCRIPT"
                info "显存 12-20GB，保持默认: batch=16, net_scale=s"
            else
                info "显存 >20GB，可以全速运行"
            fi
        fi
    fi
fi

# ============================== 7. 验证安装 ==============================
step "7/7  验证安装"

echo ""
info "=== 环境验证 ==="
echo -e "  Python:    $($PYTHON --version 2>&1)"
echo -e "  PyTorch:   $($PYTHON -c "import torch; print(torch.__version__)" 2>/dev/null || echo "未安装")"
echo -e "  CUDA可用:  $($PYTHON -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")"
echo -e "  aic2026:   $($PYTHON -c "import aic2026; print('OK')" 2>/dev/null || echo "未安装")"
echo -e "  machine_learning: $($PYTHON -c "import machine_learning; print('OK')" 2>/dev/null || echo "未安装")"

# 检查依赖完整性
echo ""
info "=== 关键依赖检查 ==="
MISSING=""
for pkg in torch torchvision ultralytics pyyaml numpy tqdm; do
    if $PYTHON -c "import $pkg" 2>/dev/null; then
        echo -e "  ${GREEN}✓${NC} $pkg"
    else
        echo -e "  ${RED}✗${NC} $pkg"
        MISSING="$MISSING $pkg"
    fi
done

if [[ -n "$MISSING" ]]; then
    warn "缺少依赖: $MISSING"
    warn "请手动运行: pip install$MISSING"
fi

# ============================== 完成 ==============================
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                🎉 部署完成！                         ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════╝${NC}"
echo ""

# 激活提示
if $USE_CONDA; then
    echo -e "  激活环境: ${CYAN}conda activate $ENV_NAME${NC}"
elif $USE_VENV; then
    echo -e "  激活环境: ${CYAN}source .venv/bin/activate${NC}"
fi

echo ""
echo -e "${YELLOW}📋 接下来：${NC}"
echo ""
echo -e "  1️⃣  将竞赛数据放入以下目录："
if [[ -n "$DATA_DIR" ]]; then
    echo -e "      ${CYAN}$DATA_DIR/visible/train/${NC}    ← RGB 图像"
    echo -e "      ${CYAN}$DATA_DIR/infrared/train/${NC}   ← 红外图像"
    echo -e "      ${CYAN}$DATA_DIR/labels/train/${NC}     ← YOLO 格式标签"
else
    echo -e "      ${CYAN}<数据目录>/visible/train/${NC}    ← RGB 图像"
    echo -e "      ${CYAN}<数据目录>/infrared/train/${NC}   ← 红外图像"
    echo -e "      ${CYAN}<数据目录>/labels/train/${NC}     ← YOLO 格式标签"
fi
echo ""
echo -e "  2️⃣  启动训练（Track 1 Baseline）："
echo -e "      ${CYAN}bash src/aic2026/scripts/train_baseline.sh${NC}"
echo ""
echo -e "  3️⃣  查看训练进度："
echo -e "      ${CYAN}tensorboard --logdir runs/${NC}"
echo ""
echo -e "  4️⃣  Track 2 三模态扩展（Baseline 完成后）："
echo -e "      ${CYAN}bash src/aic2026/scripts/train_triplet.sh${NC}"
echo ""

# 如果显存小，给提示
if [[ -n "$GPU_MEM_MB" ]] && (( $(echo "$GPU_MEM_MB < 12000" | bc -l) )); then
    echo -e "${YELLOW}💡 低显存建议：${NC}"
    echo -e "  已自动调优 batch=8 + net_scale=n。如果还 OOM，可以："
    echo -e "    - 开启 AMP: 在 train_baseline.sh 中去掉 # 号取消 --amp 的注释"
    echo -e "    - 再降 batch: 改 train_baseline.sh 中的 BATCH_SIZE=4"
    echo ""
fi
