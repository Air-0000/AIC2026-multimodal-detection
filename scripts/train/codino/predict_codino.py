"""Co-DINO (mmdet 3.x) inference to official submission files, for final or intermediate checkpoints.

Single model, single forward per image at the config's test scale (no TTA). The checkpoint can be any checkpoint
mmengine saved during training (epoch_N.pth, best_*.pth): with the EMA hook these hold the EMA weights the
validation used. While training is still running, take the newest epoch file that is complete (not the one being
written); the script prints the checkpoint's epoch and iteration.

Output format and packaging are scripts/inference/predict.py's (same functions): one <stem>.txt per image,
"<class> <cx> <cy> <w> <h> <conf>" normalized to [0, 1], CRLF, descending confidence, at most --max-dets lines,
empty file when nothing passes --score-thr; check_submission.py on the folder; with --zip an archive with the txt
files at its root and registration in the visualization workstation (AGENTS.md rule 11) unless --no-workstation.
Class ids follow the checkpoint's class names mapped onto the 12 AIC classes by name.
--dump-preds / --load-preds keep the raw detections, so another threshold needs no second inference.

Run from the repo root with codino_env.sh sourced, e.g. on the Phase 3 test set:
    python scripts/train/codino/predict_codino.py --checkpoint work_dir/codino/vit_l/epoch_8.pth \
        --images dataset/test/AIC2026_PHASE_3_1000/visible --score-thr 0.3 \
        --out-dir work_dir/submissions/codino_vitl_e8/txt --zip work_dir/submissions/codino_vitl_e8/submission.zip \
        --dump-preds work_dir/submissions/codino_vitl_e8/preds.pkl
Choose --score-thr on the labelled val split (--images <val json> --rgb-dir dataset/train/visible, then
scripts/eval/eval_submission.py), never from test-set box counts.
"""
import argparse
import json
import pickle
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'scripts/inference'))
from predict import CLASS_NAMES, lines_for, stub_annotations, write_submission  # noqa: E402

CONFIG = ROOT/'scripts/train/codino/co_dino_vit_l_aic_rgb_1800.py'


def run(config, checkpoint, images, device, limit):
    import torch
    from mmengine.config import Config
    from mmdet.apis import inference_detector, init_detector
    from mmdet.utils import register_all_modules
    register_all_modules()
    meta = torch.load(checkpoint, map_location='cpu', weights_only=False).get('meta', {})
    print(f'checkpoint {checkpoint}: epoch {meta.get("epoch")}, iter {meta.get("iter")}')
    model = init_detector(Config.fromfile(str(config)), str(checkpoint), device=device)
    names = list(model.dataset_meta['classes'])
    if sorted(names) != sorted(CLASS_NAMES):
        raise SystemExit(f'checkpoint classes {names} are not the 12 AIC classes')
    to_aic = [CLASS_NAMES.index(n) for n in names]
    preds, t0 = {}, time.time()
    for i, im in enumerate(images[:limit or None]):
        res = inference_detector(model, str(im['path']))
        inst = res.pred_instances.cpu().numpy()
        h, w = res.metainfo['ori_shape'][:2]
        if (w, h) != (im['width'], im['height']):
            raise RuntimeError(f'{im["file_name"]}: model read {w}x{h}, listed {im["width"]}x{im["height"]}')
        preds[im['file_name']] = (w, h, [(to_aic[int(l)], float(s), *map(float, b))
                                         for b, s, l in zip(inst.bboxes, inst.scores, inst.labels)])
        if (i + 1) % 100 == 0:
            print(f'{i + 1} images, {(time.time() - t0) / (i + 1):.2f} s/image', flush=True)
    return preds


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', type=Path, default=CONFIG)
    p.add_argument('--checkpoint', type=Path, help='required unless --load-preds')
    p.add_argument('--images', type=Path, required=True, help='RGB image folder, or COCO json listing the images')
    p.add_argument('--rgb-dir', type=Path, help='folder holding the images (default: --images when it is a folder)')
    p.add_argument('--score-thr', type=float, required=True,
                   help='choose on labelled validation data, never from test-set box counts')
    p.add_argument('--max-dets', type=int, default=100)
    p.add_argument('--out-dir', type=Path, required=True, help='txt output folder (must not exist)')
    p.add_argument('--zip', type=Path)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--limit', type=int, default=0, help='first N images only (smoke test; skips the coverage check)')
    p.add_argument('--dump-preds', type=Path, help='save raw detections to a pickle')
    p.add_argument('--load-preds', type=Path, help='load raw detections from a pickle instead of running the model')
    p.add_argument('--no-workstation', action='store_true', help='skip workstation registration (e.g. cloud machines)')
    p.add_argument('--workstation-id')
    p.add_argument('--workstation-name')
    a = p.parse_args()
    rgb_dir = a.rgb_dir or a.images
    if not rgb_dir.is_dir():
        p.error('--rgb-dir is required when --images is a json')

    if a.load_preds:
        with open(a.load_preds, 'rb') as f:
            preds = pickle.load(f)
    else:
        if not a.checkpoint:
            p.error('--checkpoint is required unless --load-preds')
        listing = stub_annotations(a.images, rgb_dir)['images']
        images = [dict(im, path=rgb_dir/im['file_name']) for im in listing]
        preds = run(a.config, a.checkpoint.resolve(), images, a.device, a.limit)
        if a.dump_preds:
            a.dump_preds.parent.mkdir(parents=True, exist_ok=True)
            with open(a.dump_preds, 'wb') as f:
                pickle.dump(preds, f)
            print(f'dumped raw detections to {a.dump_preds}')

    out_dir = write_submission(preds, a.out_dir.resolve(), a.score_thr, a.max_dets)
    total = sum(len(lines_for(w, h, d, a.score_thr, a.max_dets)) for w, h, d in preds.values())
    print(f'{len(preds)} images, {total} boxes at score >= {a.score_thr} written to {out_dir}')
    (out_dir.parent/(out_dir.name + '_meta.json')).write_text(json.dumps(dict(
        checkpoint=str(a.checkpoint), config=str(a.config), score_thr=a.score_thr, max_dets=a.max_dets,
        images=str(a.images), n_images=len(preds), n_boxes=total), indent=1))
    if a.images.is_dir() and not a.limit:
        check = subprocess.run([sys.executable, str(ROOT/'scripts/eval/check_submission.py'), str(out_dir),
                                str(a.images.resolve())])
        if check.returncode:
            sys.exit('check_submission.py failed')
    else:
        print('skipped check_submission.py coverage check (image list from json, or --limit)')
    if a.zip:
        a.zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(a.zip, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(out_dir.glob('*.txt')):
                zf.write(f, arcname=f.name)
        print(f'zip: {a.zip}')
        if not a.no_workstation:
            sys.path.insert(0, str(ROOT))
            try:
                from scripts.viz.import_submission_to_workstation import auto_import_submission
                auto_import_submission(str(a.zip), model_id=a.workstation_id, name=a.workstation_name)
            except Exception as e:  # the submission itself is complete; registration is local tooling
                print(f'WARNING: workstation registration failed ({type(e).__name__}: {e}); rerun '
                      f'scripts/viz/import_submission_to_workstation.py --zip {a.zip}')


if __name__ == '__main__':
    main()
