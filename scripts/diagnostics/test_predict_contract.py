#!/usr/bin/env python3
"""Contract for scripts/inference/predict.py on the labelled validation set; inference only, no training.

  1. inputs   every validation image: predict's input tensor equals the training validation loader's
              tensor bitwise, with the same original size
  2. outputs  predict's detections give the expected standard-COCO mAP of a reference checkpoint
              (reference pycocotools, no per-image cap)
  3. files    the txt files pass scripts/eval/check_submission.py and parse back to the top --max-dets
              detections within print precision (classes, order, CRLF)

Both loaders use the same folders: the config's val-set folders, or the --*-dir overrides applied to both.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
FW = ROOT/'vendor/DEIMv2'
sys.path[:0] = [str(ROOT/'scripts/analysis'), str(ROOT/'scripts/inference')]
import numpy as np
import torch
# Bind the reference pycocotools before predict imports the engine (which swaps in faster_coco_eval).
from pycocotools.coco import COCO
import analyze_light_val_thresholds as lvt
from analyze_light_val_thresholds import evaluate
import predict as P


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else (FW/path).resolve()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--ann', type=Path, default=ROOT/'work_dir/deimv2_dataset/valid/annotations/instances_val.json')
    p.add_argument('--rgb-dir', type=Path)
    p.add_argument('--ir-dir', type=Path)
    p.add_argument('--lbg-dir', type=Path)
    p.add_argument('--depth-dir', type=Path)
    p.add_argument('--p3-quota', type=float)
    p.add_argument('--expect-map', type=float, help='expected standard-COCO mAP; asserted to 1e-6')
    p.add_argument('--max-dets', type=int, default=100)
    p.add_argument('--device', default='cuda')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    ann = a.ann.resolve()
    report = dict(config=str(a.config.resolve()), checkpoint=str(a.checkpoint.resolve()),
                  checkpoint_sha256=sha256(a.checkpoint), annotation=str(ann), predict_sha256=sha256(P.__file__),
                  script_sha256=sha256(__file__), checks={})

    overrides = {'DFINETransformer': {'p3_quota_ratio': a.p3_quota}} if a.p3_quota is not None else None
    base = P.load_config(a.config, overrides)
    ds = base['val_dataloader']['dataset']
    rgb_only = ds.get('modality') == 'rgb'
    keys = dict(img_folder=a.rgb_dir) if rgb_only else dict(img_folder=a.rgb_dir, ir_folder=a.ir_dir, raw_ir_folder=a.lbg_dir, depth_folder=a.depth_dir)
    # CLI overrides are relative to the working directory, config values to vendor/DEIMv2.
    folders = {k: str(v.resolve()) if v else str(resolve(ds[k])) for k, v in keys.items()}
    report['folders'] = folders

    train_c = copy.deepcopy(base)
    tds = train_c['val_dataloader']['dataset']
    tds.update(folders, ann_file=str(ann))
    train_c['val_dataloader'].update(total_batch_size=1, num_workers=2, shuffle=False, drop_last=False)
    train_c['val_dataloader'].pop('persistent_workers', None)
    stub = out/'images.json'
    stub.write_text(json.dumps(P.stub_annotations(ann, folders['img_folder'])))
    pred_c = P.configure_inputs(copy.deepcopy(base), stub, folders['img_folder'], folders.get('ir_folder'),
                                folders.get('raw_ir_folder'), folders.get('depth_folder'))
    cfg_train, cfg_pred = P.build(train_c, out, 'train_val'), P.build(pred_c, out, 'predict')
    assert lvt.COCOeval.__module__ == 'pycocotools.cocoeval' and COCO.__module__ == 'pycocotools.coco'

    # 1. inputs
    lt, lp = cfg_train.val_dataloader, cfg_pred.val_dataloader
    ct, cp = lt.dataset.coco, lp.dataset.coco
    n = 0
    for (xt, tt), (xp, tp) in zip(lt, lp):
        name_t = ct.imgs[int(tt[0]['image_id'].item())]['file_name']
        name_p = cp.imgs[int(tp[0]['image_id'].item())]['file_name']
        assert name_t == name_p, (name_t, name_p)
        assert xt.shape == xp.shape and torch.equal(xt, xp), f'input tensor differs for {name_t}'
        assert torch.equal(tt[0]['orig_size'], tp[0]['orig_size']), f'orig_size differs for {name_t}'
        n += 1
    assert n == len(ct.imgs) == len(cp.imgs), (n, len(ct.imgs), len(cp.imgs))
    report['checks']['inputs'] = dict(images=n, shape=list(xp.shape), bitwise_equal=True, train_loader_strict=bool(tds.get('strict_inputs', False)))
    print(f'PASS inputs: {n} images bitwise equal, shape {list(xp.shape)}', flush=True)

    # 2. outputs
    model, post = P.load_model(cfg_pred, a.checkpoint.resolve(), a.device)
    preds = P.run(cfg_pred, model, post, a.device)
    ids = {im['file_name']: im['id'] for im in json.loads(stub.read_text())['images']}
    rows = [dict(image_id=ids[name], category_id=lab, bbox=[x1, y1, x2-x1, y2-y1], score=s)
            for name, (w, h, dets) in preds.items() for lab, s, x1, y1, x2, y2 in dets]
    coco = COCO(str(ann))
    e = evaluate(coco, rows)
    got = float(e.stats[0])
    report['checks']['outputs'] = dict(detections=len(rows), mAP=got, AP50=float(e.stats[1]), AP75=float(e.stats[2]), expected=a.expect_map)
    if a.expect_map is not None:
        assert abs(got-a.expect_map) <= 1e-6, f'mAP {got} != expected {a.expect_map}'
    print(f'PASS outputs: mAP {got:.10f}' + (f' == expected {a.expect_map}' if a.expect_map is not None else ' (no expectation given)'), flush=True)

    # 3. files
    txt = P.write_submission(preds, out/'txt', 0., a.max_dets)
    stems = out/'stems'
    stems.mkdir()
    for name in preds:
        (stems/name).touch()
    check = subprocess.run([sys.executable, str(ROOT/'scripts/eval/check_submission.py'), str(txt), str(stems)], capture_output=True, text=True)
    (out/'check_submission.log').write_text(check.stdout+check.stderr)
    assert check.returncode == 0, check.stdout[-2000:]
    back, worst = [], 0.
    for name, (w, h, dets) in preds.items():
        raw = (txt/(Path(name).stem+'.txt')).read_bytes()
        lines = raw.decode().split('\r\n')
        assert lines[-1] == '', f'{name}: last line not CRLF-terminated'
        lines = [ln.split() for ln in lines[:-1]]
        valid = []
        for lab, s, x1, y1, x2, y2 in sorted(dets, key=lambda d: -d[1]):
            x1, x2 = min(max(x1, 0.), w), min(max(x2, 0.), w)
            y1, y2 = min(max(y1, 0.), h), min(max(y2, 0.), h)
            if x2 > x1 and y2 > y1:
                valid.append((lab, s, x1, y1, x2, y2))
        valid = valid[:a.max_dets]
        assert len(lines) == len(valid), (name, len(lines), len(valid))
        scores = [float(ln[5]) for ln in lines]
        assert scores == sorted(scores, reverse=True), f'{name}: not sorted by confidence'
        for ln, (lab, s, x1, y1, x2, y2) in zip(lines, valid):
            vals = np.array([float(v) for v in ln[1:]])
            ref = np.array([(x1+x2)/2/w, (y1+y2)/2/h, (x2-x1)/w, (y2-y1)/h, s])
            assert int(ln[0]) == lab, (name, ln, lab)
            worst = max(worst, float(np.abs(vals-ref).max()))
            cx, cy, bw, bh, sc = vals
            back.append(dict(image_id=ids[name], category_id=lab, bbox=[(cx-bw/2)*w, (cy-bh/2)*h, bw*w, bh*h], score=sc))
    assert worst <= 5.01e-7, worst
    e2 = evaluate(coco, back)
    report['checks']['files'] = dict(check_submission='PASS', max_abs_rounding=worst, boxes=len(back),
                                     roundtrip_mAP_with_cap=float(e2.stats[0]), roundtrip_minus_raw=float(e2.stats[0])-got)
    print(f'PASS files: check_submission ok, max rounding {worst:.2e}, round-trip mAP with cap {e2.stats[0]:.6f} '
          f'({e2.stats[0]-got:+.6f} vs raw)', flush=True)
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print('ALL_PASS')


if __name__ == '__main__':
    main()
