#!/usr/bin/env bash
# Deployment root is isolated; abort before training on any setup/contract failure.
set -euo pipefail
R=/root/744_gate
PY=$R/venv/bin/python
MM=$R/src/onedl-mmdetection-3.5.1
NG=$R/scripts/train/codino/swinl_rgb_full
export PYTHONPATH="$MM:$NG:$R/scripts/train/codino/aic_mm:$R/vendor/codino_fused/scripts/cloud/newbase/codino"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$MM"
if [ -n "${SETUP_PID:-}" ]; then
  while kill -0 "$SETUP_PID" 2>/dev/null; do sleep 10; done
fi
$PY -c 'import torch;assert torch.__version__.startswith("2.10.0");print(torch.__version__)'
$PY -m pip install --no-cache-dir -i https://mirrors.cloud.tencent.com/pypi/simple onedl-mmengine==0.10.12 fairscale numpy==2.2.6 yapf==0.43.0
$PY -m pip install --no-cache-dir -i https://mirrors.cloud.tencent.com/pypi/simple "$R/wheels/onedl_mmcv-2.3.7.post1-cp312-cp312-manylinux_2_34_x86_64.whl"
$PY -m pip install --no-cache-dir -i https://mirrors.cloud.tencent.com/pypi/simple -e "$MM"
$PY -c 'import torch;from mmcv.ops import MultiScaleDeformableAttention;print(torch.cuda.get_device_name())'
$PY -m pip freeze > "$R/logs/pip_freeze.txt"
test -f "$R/logs/gpu_selftest.json"
$PY - <<'PY'
import json
x=json.load(open('/root/744_gate/logs/gpu_selftest.json'))
print(x)
assert x['ok'], 'GPU self-test failed'
PY
echo 'df4f1cb85a0ca4b7f1cf0e5e6d1e97f5b4ba82f58f3bcfe99be71d3578b6c5d0  /root/744_gate/data/train.json' | sha256sum -c -
$PY "$NG/build_744_gate_cfg.py" --root "$R"
$PY "$NG/check_744_gate.py" > "$R/logs/contract.log" 2>&1
test -f "$R/CONTRACT_PASS"
echo "TRAIN_START $(date -Is)"
$PY tools/train.py "$R/744_gate.py" --work-dir "$R/train"
echo "TRAIN_COMPLETE $(date -Is)"
$PY "$NG/verify_744_checkpoint.py" "$R/train/epoch_4.pth"
