#!/usr/bin/env bash
# =============================================================================
# Build an isolated tree for the RGB-only control next to the running clean baseline v1 on ailab40,
# without touching v1's frozen tree. Runs on ailab40.
#   - same data links as v1 (dataset, deimv2_dataset, train_lbg_ir), same COCO/teacher ckpts link
#   - v1's frozen engine/ and train.py copied verbatim; only engine/data/dataset/coco_dataset_mm.py
#     is replaced by the repo version that adds modality='rgb' (verified by SHA below)
#   - v1 config copied from the frozen tree, RGB control config from the repo (STAGE)
#   - CPU contract test copied in; run it with run_contract=1 (nice'd, few threads, CPU only)
# Usage: STAGE=/tmp/rgbctl_stage DST=/home/ailab40/hejianzhong/AIC_RGB_CONTROL_<date> [run_contract=1] bash deploy_rgb_control_ailab40.sh
# STAGE holds: coco_dataset_mm.py aic_clean_baseline_v1_rgb_control.yml aic_clean_baseline_v1.yml
#              test_rgb_control_contract.py run_clean_baseline_v1_rgb_control_ailab40.sh lib/*.py
# =============================================================================
set -euo pipefail

SRC=${SRC:-/home/ailab40/hejianzhong/AIC_CLEAN_BASELINE_20261021}
DST=${DST:?set DST to a new directory}
STAGE=${STAGE:?set STAGE to the staged repo files}
PY=${PY:-/home/ailab/ssd/bigdata/hejianzhong/_infra/envs/aic2026/bin/python}
EXPECT_DATASET_SHA=3f989b1bec8440359d81164b76dda52112903786110e42ae5ba6d20737313b84   # repo commit 0a8f66c
EXPECT_SRC_DATASET_SHA=4ae995c244e08c692a3ffe42747585357fbc3d37370f40690b64a461a3bf757f

[ -e "$DST" ] && { echo "refusing: $DST already exists"; exit 1; }
[ "$(sha256sum "$STAGE/coco_dataset_mm.py" | cut -d' ' -f1)" = "$EXPECT_DATASET_SHA" ] || { echo "staged coco_dataset_mm.py has an unexpected SHA"; exit 1; }
[ "$(sha256sum "$SRC/vendor/DEIMv2/engine/data/dataset/coco_dataset_mm.py" | cut -d' ' -f1)" = "$EXPECT_SRC_DATASET_SHA" ] || { echo "v1 frozen dataset file changed"; exit 1; }
cmp -s "$STAGE/aic_clean_baseline_v1.yml" "$SRC/vendor/DEIMv2/configs/deimv2/aic_clean_baseline_v1.yml" || { echo "repo v1 config differs from v1's frozen config"; exit 1; }

mkdir -p "$DST/vendor/DEIMv2/configs/deimv2" "$DST/work_dir" "$DST/scripts/diagnostics" "$DST/scripts/train" "$DST/logs"
ln -s "$(readlink -f "$SRC/dataset")" "$DST/dataset"
ln -s "$(readlink -f "$SRC/work_dir/deimv2_dataset")" "$DST/work_dir/deimv2_dataset"
ln -s "$(readlink -f "$SRC/work_dir/train_lbg_ir")" "$DST/work_dir/train_lbg_ir"   # used only by the contract test's v1 side
ln -s "$(readlink -f "$SRC/vendor/DEIMv2/ckpts")" "$DST/vendor/DEIMv2/ckpts"
cp -a "$SRC/vendor/DEIMv2/engine" "$SRC/vendor/DEIMv2/train.py" "$DST/vendor/DEIMv2/"
find "$DST/vendor/DEIMv2/engine" -name __pycache__ -type d -prune -exec rm -rf {} +
cp "$STAGE/coco_dataset_mm.py" "$DST/vendor/DEIMv2/engine/data/dataset/coco_dataset_mm.py"
cp "$SRC/vendor/DEIMv2/configs/deimv2/aic_clean_baseline_v1.yml" "$STAGE/aic_clean_baseline_v1_rgb_control.yml" "$DST/vendor/DEIMv2/configs/deimv2/"
cp -r "$STAGE/lib" "$DST/scripts/lib"
cp "$STAGE/test_rgb_control_contract.py" "$DST/scripts/diagnostics/"
cp "$STAGE/run_clean_baseline_v1_rgb_control_ailab40.sh" "$DST/scripts/train/"

# The control's code must equal v1's frozen code except the one dataset file.
sums() { (cd "$1/vendor/DEIMv2" && find engine train.py -name '*.py' -not -path '*/__pycache__/*' | sort | xargs sha256sum); }
changed=$(diff <(sums "$SRC") <(sums "$DST") | grep -E '^[<>]' | awk '{print $3}' | sort -u || true)
[ "$changed" = "engine/data/dataset/coco_dataset_mm.py" ] || { echo "unexpected code differences: ${changed:-none}"; exit 1; }
echo "code check: identical to v1 except engine/data/dataset/coco_dataset_mm.py"
(cd "$DST" && find vendor/DEIMv2/engine vendor/DEIMv2/train.py vendor/DEIMv2/configs scripts -type f | sort | xargs sha256sum) > "$DST/source_manifest.sha256"

if [ "${run_contract:-0}" = 1 ]; then
  cd "$DST"
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 nice -n 19 "$PY" scripts/diagnostics/test_rgb_control_contract.py \
    2>&1 | tee "$DST/logs/rgb_control_contract.log"
fi
echo "deployed: $DST"
