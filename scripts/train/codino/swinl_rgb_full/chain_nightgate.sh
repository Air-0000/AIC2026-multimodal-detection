#!/usr/bin/env bash
# After the pure-RGB Swin-L run (run_swinl_rgb_full.sh) finishes, on the same card:
#   1  Phase 3 raw TXT of the pure-RGB model (epoch_12; visible-only PNGs /root/rgb_p3, same decode as training)
#   2  night-gated fine-tune: build_nightgate_cfg.py (frozen detector, split A 1700, 4 epochs) + tools/train.py
#   3  Phase 3 raw TXT of the gated model (test IR / depth) and a byte comparison with 1; gates.json per image
#   4  split-A held-out probes (300 images): normal, night01, night005 for the pure-RGB model, the gated model and the
#      gated model with thermal / depth zeroed; mAP with the teammate's evaluator
# Launch: ( setsid nohup bash chain_nightgate.sh > /autodl-fs/data/lfb_runs/swinl_nightgate_s20260918.log 2>&1 < /dev/null & )
set -uo pipefail
MM=/root/tm/src/onedl-mmdetection
PY=/root/venv_codino/bin/python
NG=/root/tm/nightgate
export PYTHONPATH="$MM:/root/tm/setup:$NG:/root/tm/aic_mm_min"
B=/autodl-fs/data/lfb_runs/swinl_rgb_full_s20260918
BCFG=$B/codino_aic_full_rgb_s20260918.py
G=/autodl-fs/data/lfb_runs/swinl_nightgate_s20260918
GCFG=$G/codino_nightgate_s20260918.py
VAL=/root/rgb_all2000/splitA_val.json
EVAL=/root/tm/tools/eval_txt_coco_20261026.py
say() { echo "=== $(date '+%F %T') $*"; }

say "waiting for the pure-RGB run"
until grep -q "=== all done" "$B.log"; do
  grep -qE "=== train exit [1-9]" "$B.log" && { say "pure-RGB run failed"; exit 1; }
  sleep 60
done
CK=$B/epoch_12.pth
[ -f "$CK" ] || { say "no $CK"; exit 1; }
sha256sum "$CK"
cd "$MM"

say "1 Phase 3 raw TXT, pure RGB"
$PY $NG/predict_nightgate_txt.py --config $BCFG --ckpt $CK --ann /root/rgb_p3/test_index.json \
  --img-dir /root/rgb_p3/images --out $B/p3_raw_txt || exit 1

say "2 night-gated fine-tune"
mkdir -p $G
$PY $NG/build_nightgate_cfg.py --base $BCFG --load-from $CK --work-dir $G --out $GCFG || exit 1
$PY tools/train.py $GCFG --work-dir $G || { say "fine-tune failed"; exit 1; }
GCK=$G/epoch_4.pth
sha256sum "$GCK"

say "3 Phase 3 raw TXT, gated (test IR / depth)"
grep -q COPY_DONE /root/tm/p3_aux_copy.log || { say "Phase 3 IR / depth copy not finished"; exit 1; }
$PY $NG/predict_nightgate_txt.py --config $GCFG --ckpt $GCK --ann /root/rgb_p3/test_index.json \
  --img-dir /root/rgb_p3/images --ir-dir /root/p3_aux/infrared --depth-dir /root/p3_aux/depth --out $G/p3_raw_txt || exit 1
if diff -rq -x gates.json $B/p3_raw_txt $G/p3_raw_txt > $G/p3_txt_diff.txt; then say "P3_TXT_IDENTICAL"; else say "P3_TXT_DIFFER (see $G/p3_txt_diff.txt)"; fi

say "4 split-A held-out probes"
P=$G/probes
run() {  # name config ckpt [args]
  local name=$1 cfg=$2 ck=$3; shift 3
  $PY $NG/predict_nightgate_txt.py --config $cfg --ckpt $ck --ann $VAL --img-dir /root/rgb_all2000/images --out $P/$name "$@" > $P/$name.log 2>&1 \
    || say "probe $name failed"
}
mkdir -p $P
run rgb_normal $BCFG $CK
run gate_normal $GCFG $GCK
for lv in 0.01 0.005; do
  t=night$(echo $lv | sed 's/0\.//')
  run rgb_$t $BCFG $CK --night $lv
  run gate_$t $GCFG $GCK --night $lv
  run gate_${t}_auxoff $GCFG $GCK --night $lv --zero-aux
done
$PY $EVAL --gt $VAL --pred $P/rgb_normal $P/gate_normal $P/rgb_night01 $P/gate_night01 $P/gate_night01_auxoff \
  $P/rgb_night005 $P/gate_night005 $P/gate_night005_auxoff | tee $P/map.txt
say "all done"
