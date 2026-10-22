#!/usr/bin/env bash
# =============================================================================
# Pre-launch checks for clean-baseline training on a new machine, in a fixed order; stops at the
# first failure and writes everything under logs/preflight_<time>/.
#   1. environment snapshot (commit, Python, torch/CUDA/cuDNN, GPUs incl. ECC, pip freeze)
#   2. inputs: sha256sum -c work_dir/transfer/MANIFEST.sha256 (from scripts/prep/pack_training_inputs.py)
#   3. GPU self-test on every GPU in GPUS (scripts/diagnostics/gpu_selftest.py)
#   4. CPU contract tests: clean baseline (14), RGB control (7), night arm (3); a skipped test fails
#   5. optional BENCH=1: 30-step benchmark of v1 and the RGB control on the first GPU
# Usage: ROOT=/path/to/repo GPUS="0 1" [PY=python] [REQUIRE_ECC=1] [BENCH=1] [DATA_ROOT=$ROOT] \
#        bash scripts/train/preflight_cloud.sh
# =============================================================================
set -euo pipefail

ROOT=${ROOT:?set ROOT to the repo checkout}
GPUS=${GPUS:?set GPUS, e.g. "0 1"}
PY=${PY:-python}
DATA_ROOT=${DATA_ROOT:-$ROOT}
OUT="$ROOT/logs/preflight_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
step() { echo "== $(date '+%T') $*" | tee -a "$OUT/summary.txt"; }
fail() { step "FAIL $*"; exit 1; }
cd "$ROOT"

step "1/5 environment -> $OUT/env.txt"
{
  git rev-parse HEAD 2>/dev/null || echo "not a git checkout"
  "$PY" -c "import sys, torch; print(sys.version); print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cudnn', torch.backends.cudnn.version())"
  nvidia-smi --query-gpu=index,name,driver_version,ecc.mode.current,retired_pages.double_bit.count,memory.total --format=csv
} > "$OUT/env.txt" 2>&1
"$PY" -m pip freeze > "$OUT/pip_freeze.txt" 2>/dev/null || true
git status --short > "$OUT/git_status.txt" 2>/dev/null || true

step "2/5 inputs"
MANIFEST="$DATA_ROOT/work_dir/transfer/MANIFEST.sha256"
[ -f "$MANIFEST" ] || fail "missing $MANIFEST (stage inputs with scripts/prep/pack_training_inputs.py)"
(cd "$DATA_ROOT" && sha256sum -c --quiet work_dir/transfer/MANIFEST.sha256) > "$OUT/manifest_check.txt" 2>&1 \
  || fail "inputs differ from the manifest: $OUT/manifest_check.txt"
step "   $(wc -l < "$MANIFEST") files match the manifest"

step "3/5 GPU self-test"
for g in $GPUS; do
  CUDA_VISIBLE_DEVICES="$g" "$PY" scripts/diagnostics/gpu_selftest.py ${REQUIRE_ECC:+--require-ecc} \
    --json "$OUT/gpu$g.json" > "$OUT/gpu$g.log" 2>&1 || fail "GPU $g: $OUT/gpu$g.log"
  step "   GPU $g pass"
done

step "4/5 CPU contract tests"
for t in test_clean_baseline_contract test_rgb_control_contract test_night_aug_contract; do
  CUDA_VISIBLE_DEVICES='' "$PY" "scripts/diagnostics/$t.py" > "$OUT/$t.log" 2>&1 || fail "$t: $OUT/$t.log"
  grep -q 'skipped' "$OUT/$t.log" && fail "$t skipped a test: $OUT/$t.log"
  step "   $t: $(grep -E '^Ran ' "$OUT/$t.log")"
done

if [ "${BENCH:-0}" = 1 ]; then
  step "5/5 benchmark"
  g=${GPUS%% *}
  for cfg in aic_clean_baseline_v1 aic_clean_baseline_v1_rgb_control; do
    CUDA_VISIBLE_DEVICES="$g" "$PY" scripts/train/bench_clean_arm.py --config "configs/deimv2/$cfg.yml" \
      > "$OUT/bench_$cfg.log" 2>&1 || fail "benchmark $cfg: $OUT/bench_$cfg.log"
    step "   $(grep '^BENCH' "$OUT/bench_$cfg.log")"
  done
else
  step "5/5 benchmark skipped (BENCH=1 to run)"
fi
step "PREFLIGHT_PASS -> launch with scripts/train/run_clean_arm.sh"
