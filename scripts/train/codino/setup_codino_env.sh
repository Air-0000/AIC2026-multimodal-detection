#!/usr/bin/env bash
# =============================================================================
# One-time setup of the Co-DINO environment for the pure-RGB ceiling runs on Blackwell GPUs (SM 12.0).
# A separate conda env; the deimv2 training env is not touched. Installs, in order:
#   python 3.10 · torch 2.7.1 / torchvision 0.22.1 cu128 (first release with sm_120 kernels)
#   · a CUDA 12.8 nvcc: CUDA_HOME if it is set (e.g. /usr/local/cuda-12.8), otherwise the toolkit is installed
#     inside the env (nvcc must match torch's CUDA major; seeta's system CUDA is 13.0)
#   · mmengine 0.10.7 · fairscale 0.4.13 (Co-DETR encoder activation checkpointing)
#   · mmcv 2.1.0 compiled for sm_120 (mmdet 3.3.0 needs mmcv < 2.2.0 and no prebuilt wheel covers Blackwell)
#   · mmdetection v3.3.0, editable, in vendor/mmdetection (Co-DETR lives in its projects/ folder)
# then freezes the package list into logs/ and runs check_codino_env.py. Checkpoints: fetch_codino_weights.sh.
# Re-running skips the env and clones that already exist; pip steps are idempotent.
# Usage: ROOT=/root/autodl-tmp/AIC_clean_v1 bash scripts/train/codino/setup_codino_env.sh
#   workstation: ROOT=... CONDA=~/miniconda3/bin/conda ENV_PREFIX=~/miniconda3/envs/codino \
#                CUDA_HOME=/usr/local/cuda-12.8 BUILD_DIR=... MAX_JOBS=6 bash .../setup_codino_env.sh
# Needs ~20 GB free under ENV_PREFIX / BUILD_DIR (on seeta keep both on the data disk; / is small) and access to
# GitHub, download.pytorch.org and PyPI (/etc/network_turbo is sourced when present).
# =============================================================================
set -euo pipefail
source "$(dirname "$0")/codino_env.sh"
CONDA=${CONDA:-/root/miniconda3/bin/conda}
BUILD_DIR=${BUILD_DIR:-/root/autodl-tmp/build}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}
MAX_JOBS=${MAX_JOBS:-16}

mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/codino_env_setup_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
step() { echo; echo "== $(date '+%F %T') $*"; }

if [ -f /etc/network_turbo ]; then set +u; source /etc/network_turbo; set -u; fi

command -v g++ >/dev/null || { echo "g++ not found (needed to compile mmcv)"; exit 1; }

step "conda env $ENV_PREFIX"
# conda-forge only: newer conda refuses the anaconda default channels until their terms are accepted.
# CONDA_FORGE can point at a mirror, e.g. https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
[ -x "$PY" ] || "$CONDA" create -y -p "$ENV_PREFIX" --override-channels -c "${CONDA_FORGE:-conda-forge}" python=3.10
PIP=("$PY" -m pip install)

step "torch 2.7.1 + torchvision 0.22.1 (cu128)"
# Behind the Great Firewall: TORCH_FIND_LINKS=https://mirrors.aliyun.com/pytorch-wheels/cu128/ with PIP_INDEX_URL
# pointing at a PyPI mirror (the CUDA runtime wheels come from the index).
if [ -n "${TORCH_FIND_LINKS:-}" ]; then TORCH_SRC=(-f "$TORCH_FIND_LINKS"); else TORCH_SRC=(--index-url "$TORCH_INDEX"); fi
"${PIP[@]}" torch==2.7.1+cu128 torchvision==0.22.1+cu128 "${TORCH_SRC[@]}"

if [ -n "${CUDA_HOME:-}" ]; then
  step "CUDA toolkit at $CUDA_HOME"
else
  step "CUDA 12.8 toolkit inside the env"
  "$CONDA" install -y -p "$ENV_PREFIX" --override-channels -c nvidia/label/cuda-12.8.1 -c conda-forge cuda-toolkit
  CUDA_HOME="$ENV_PREFIX"
  # NVIDIA's conda packages keep headers and libraries under targets/; torch's extension builder only looks in
  # $CUDA_HOME/include and lib64.
  export CPATH="$ENV_PREFIX/targets/x86_64-linux/include:${CPATH:-}"
  export LIBRARY_PATH="$ENV_PREFIX/targets/x86_64-linux/lib:$ENV_PREFIX/lib:${LIBRARY_PATH:-}"
fi
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
"$CUDA_HOME/bin/nvcc" --version | tail -n 2
"$CUDA_HOME/bin/nvcc" --version | grep -q 'release 12\.8' || { echo "need nvcc 12.8 to match torch cu128"; exit 1; }

step "build tools and pinned runtime packages"
# mmcv 2.1.0's setup.py imports pkg_resources (removed in setuptools 81; 69.x fails to import without
# backports.tarfile). opencv-python is pinned because the mmengine wheel requires it and newer opencv releases pull
# numpy 2.
"${PIP[@]}" setuptools==75.8.2 wheel ninja numpy==1.26.4 opencv-python==4.10.0.84 mmengine==0.10.7
"${PIP[@]}" --no-build-isolation fairscale==0.4.13

step "mmcv 2.1.0 compiled for sm_120 from the PyPI sdist (takes 10-30 min)"
# BUILD_DIR holds pip's temporary build tree (a few GB); on seeta / is too small for it.
mkdir -p "$BUILD_DIR"
TMPDIR="$BUILD_DIR" MMCV_WITH_OPS=1 FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=$MAX_JOBS \
  "${PIP[@]}" --no-build-isolation --no-binary mmcv -v mmcv==2.1.0

step "mmdetection v3.3.0 in $MMDET_DIR (imported through PYTHONPATH, not installed)"
# Without GitHub access, copy a v3.3.0 checkout to $MMDET_DIR beforehand; it is used as is.
[ -d "$MMDET_DIR" ] || git clone --depth 1 -b v3.3.0 https://github.com/open-mmlab/mmdetection.git "$MMDET_DIR"
# Not pip-installed: mmengine resolves mmdet::... config bases from an installed distribution's location, which for a
# PEP 660 editable install is site-packages/mmdet (absent). Without a distribution it falls back to the imported
# package, i.e. $MMDET_DIR/mmdet via codino_env.sh's PYTHONPATH. Only the runtime requirements are installed.
"$PY" -m pip uninstall -y mmdet 2>/dev/null || true
"${PIP[@]}" -r "$MMDET_DIR/requirements/runtime.txt"
# The config bases live under mmdet/.mim, which mmdet's setup.py would link; same links by hand.
mkdir -p "$MMDET_DIR/mmdet/.mim"
for f in tools configs demo model-index.yml dataset-index.yml; do
  ln -sfn "../../$f" "$MMDET_DIR/mmdet/.mim/$f"
done

step "opencv / numpy pins"
if ! "$PY" -c "import cv2" 2>/dev/null; then
  echo "opencv-python does not import (usually missing libGL); switching to the headless build"
  "$PY" -m pip uninstall -y opencv-python
  "${PIP[@]}" --no-deps opencv-python-headless==4.10.0.84
fi
"${PIP[@]}" --no-deps numpy==1.26.4
"$PY" -c "import cv2, numpy; print('cv2', cv2.__version__, 'numpy', numpy.__version__)"

step "package freeze"
"$PY" -m pip freeze > "$ROOT/logs/codino_env_freeze_$(date +%Y%m%d_%H%M%S).txt"

step "environment check"
cd "$ROOT"
"$PY" "$CODINO_DIR/check_codino_env.py"
echo "setup done; log $LOG"
