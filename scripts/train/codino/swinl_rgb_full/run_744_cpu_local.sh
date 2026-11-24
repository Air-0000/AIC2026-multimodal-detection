#!/usr/bin/env bash
set -euo pipefail
AIC_CPU_REPO=$(cd "$(dirname "$0")/../../../.." && pwd)
AIC_CPU_DATA="$AIC_CPU_REPO/work_dir/aic744_cpu_val300"
export PYTHONPATH="$AIC_CPU_REPO/archive/scratch/744_gate/onedl-mmdetection-3.5.1:$AIC_CPU_REPO/scripts/train/codino/swinl_rgb_full:$AIC_CPU_REPO/scripts/train/codino/aic_mm:$AIC_CPU_REPO/vendor/codino_fused/scripts/cloud/newbase/codino"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
exec /Users/constant/.cache/aic744-cpu-env/bin/python "$AIC_CPU_REPO/scripts/train/codino/swinl_rgb_full/eval_744_cpu.py" --threads 4 --root "$AIC_CPU_DATA" --checkpoint "$AIC_CPU_REPO/outputs/checkpoints/s0918_gate_20261101/epoch_4.pth" --images "$AIC_CPU_DATA/visible" --ir "$AIC_CPU_DATA/infrared" --depth "$AIC_CPU_DATA/depth" --ann "$AIC_CPU_DATA/val.json" "$@"
