#!/usr/bin/env bash
# Teammate's card-A env (/root/venv_codino, 10-26) on seeta: same commands and pins as his setup_codino_wsl.sh,
# only paths changed (L=/root/tm) and the PyPI index switched to bfsu (aliyun PyPI ~1 MB/s here, bfsu ~15 MB/s).
set -euxo pipefail
HERE=/root/tm/setup
L=/root/tm
V=/root/venv_codino
W=$L/wheels
SRC=$L/src/onedl-mmdetection
TGZ=$HERE/onedl-mmdetection-v3.5.1-e1def01f.tgz
TGZ_SHA=8029c9927767f19e18e7b93fe02a4983af4b57ec9f3156ed10c6d09841dd5d08
IDX=https://mirrors.bfsu.edu.cn/pypi/web/simple
PINS=$HERE/wsl_codino_pins.txt
TORCH_WHL=torch-2.10.0+cu129-cp312-cp312-manylinux_2_28_x86_64.whl
TORCH_SHA=a3c703e74a88cccfeb4b1807c49d563a0a73793ba72d4fa035d4ac5885f3aefd
TV_WHL=torchvision-0.25.0+cu129-cp312-cp312-manylinux_2_28_x86_64.whl
TV_SHA=5d741dff079cd9b9dfe690f2f14ba6db22bd9415838726ab795c378a12adbc3b
PIP="$V/bin/pip --disable-pip-version-check"
PIPI="$PIP install --no-cache-dir -i $IDX --trusted-host mirrors.bfsu.edu.cn"
mkdir -p "$W" "$L/src"
# 1) venv (miniconda python 3.12.3, same as card A) + card A's pip / setuptools / wheel
[ -x "$V/bin/python" ] || /root/miniconda3/bin/python3.12 -m venv "$V"
$PIPI "pip==26.2.2" "setuptools==84.2.1" "wheel==0.48.0"
# 2) torch / torchvision wheels, sha256 checked
if [ ! -f "$W/$TORCH_WHL" ]; then
  /root/miniconda3/bin/python3 "$HERE/robust_range_download.py" --out "$W/$TORCH_WHL" --size 1278028350 --sha256 "$TORCH_SHA" \
    --src "noproxy:https://mirrors.aliyun.com/pytorch-wheels/cu129/$TORCH_WHL" \
    --src "noproxy:https://download.pytorch.org/whl/cu129/torch-2.10.0%2Bcu129-cp312-cp312-manylinux_2_28_x86_64.whl" \
    --workers 16
fi
if [ ! -f "$W/$TV_WHL" ]; then
  curl -sSL --retry 5 -A curl/7.81.0 -o "$W/$TV_WHL.part" "https://mirrors.aliyun.com/pytorch-wheels/cu129/$TV_WHL"
  mv "$W/$TV_WHL.part" "$W/$TV_WHL"
fi
echo "$TORCH_SHA  $W/$TORCH_WHL" | sha256sum -c -
echo "$TV_SHA  $W/$TV_WHL" | sha256sum -c -
# 3) dependencies pinned one by one
$PIPI -c "$PINS" "$W/$TORCH_WHL" "$W/$TV_WHL"
"$V/bin/python" -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list(), torch.cuda.is_available())"
$PIPI -c "$PINS" "onedl-mmengine==0.10.12" "fairscale==0.4.13"
$PIPI -c "$PINS" --only-binary=onedl-mmcv --extra-index-url https://mmwheels.onedl.ai/cu129-torch2100/simple/ "onedl-mmcv==2.3.7.post1"
# 4) onedl-mmdetection source + card A's three patched head files
echo "$TGZ_SHA  $TGZ" | sha256sum -c -
[ -f "$SRC/setup.py" ] || tar -xzf "$TGZ" -C "$L/src"
apply_card_file() {
  local dst="$SRC/$1"
  if [ "$(sha256sum "$dst" | cut -d' ' -f1)" = "$4" ]; then return 0; fi
  [ "$(sha256sum "$dst" | cut -d' ' -f1)" = "$3" ] || { echo "orig sha differs from card A: $dst"; exit 3; }
  cp -p "$dst" "$dst.orig"
  cp "$HERE/card_patches/$2" "$dst"
  [ "$(sha256sum "$dst" | cut -d' ' -f1)" = "$4" ] || { echo "patched sha differs: $dst"; exit 3; }
}
apply_card_file mmdet/models/dense_heads/detr_head.py mmdet__models__dense_heads__detr_head.py \
  5aa3bd7c4d3b5df1b6eecc7211d3d6dfab3933e240e71a41e425ee9d7d1d3338 7e0c7258c0fae84c84d4d769a3367d4d46eac246b3dceef9c12a06315db0b9cf
apply_card_file mmdet/models/dense_heads/dino_head.py mmdet__models__dense_heads__dino_head.py \
  cf8707aac5e56004af29c828f4890ce8136047bc3203d360eacbb06f86378bba 5b9b8c64b0b2f69b36a8273ec2aa98a72d1cc7995cb343c55c6e0aac27649a05
apply_card_file projects/CO-DETR/codetr/co_dino_head.py projects__CO-DETR__codetr__co_dino_head.py \
  428dfe0b7d29f2e24b7639e0cb549541a6d5c52ad46c18c5b39329ac49f7aec0 ff2430e3d668d51dcee409cf9b71c92644dacc5b33d36dcbdc34acf4d47d98b6
$PIPI -c "$PINS" -e "$SRC"
# 5) every pinned version once more, then self-check
$PIPI -r "$PINS"
$PIP check || true
$PIP freeze > "$L/pip_freeze.txt"
"$V/bin/python" -c "import torch, mmcv, mmengine, mmdet; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cudnn', torch.backends.cudnn.version()); print('mmcv', mmcv.__version__, 'mmengine', mmengine.__version__, 'mmdet', mmdet.__version__, mmdet.__file__)"
cd "$SRC" && PYTHONPATH="$SRC:$HERE" "$V/bin/python" "$HERE/smoke_mmcv_ops.py"
echo SETUP_DONE
