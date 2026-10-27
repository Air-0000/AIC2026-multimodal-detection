"""Sanity check of a converted Co-DINO COCO checkpoint: runs the 80-class model on the AIC val images and scores the
COCO classes that map one-to-one onto AIC classes (person, bicycle, car, boat, sports ball -> ball,
traffic light -> light) against the AIC val labels. A broken port or conversion gives AP near 0; a working one gives
clearly non-zero AP (label conventions differ, so do not read the numbers as the model's AIC quality).

Run from the repo root with codino_env.sh sourced (fits a 16 GB GPU at the test scale):
    python scripts/train/codino/check_codino_coco_on_aic_val.py \
        --checkpoint weights/codino/co_dino_vit_l_o365tococo_mmdet3.pth --out work_dir/codino/coco_check_vitl
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

CONFIG = 'scripts/train/codino/co_dino_vit_l_aic_rgb_1800.py'
VAL_ANN = 'work_dir/deimv2_dataset/valid/annotations/instances_val.json'
IMG_DIR = 'dataset/train/visible'
COCO_TO_AIC = {'person': 'person', 'bicycle': 'bicycle', 'car': 'car', 'boat': 'boat', 'sports ball': 'ball',
               'traffic light': 'light'}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config', default=CONFIG)
    parser.add_argument('--out', required=True)
    parser.add_argument('--limit', type=int, default=0, help='first N val images only')
    args = parser.parse_args()

    from mmengine.config import Config
    from mmdet.apis import inference_detector, init_detector
    from mmdet.utils import register_all_modules
    register_all_modules()
    cfg = Config.fromfile(args.config)
    cfg.model.query_head.num_classes = 80
    cfg.model.roi_head[0].bbox_head.num_classes = 80
    cfg.model.bbox_head[0].num_classes = 80
    model = init_detector(cfg, args.checkpoint, device='cuda:0')
    coco_names = list(model.dataset_meta['classes'])
    if len(coco_names) != 80:
        raise SystemExit(f'checkpoint dataset_meta has {len(coco_names)} classes, expected the 80 COCO ones')

    gt = COCO(VAL_ANN)
    aic_id = {c['name']: c['id'] for c in gt.loadCats(gt.getCatIds())}
    label_map = {coco_names.index(c): aic_id[a] for c, a in COCO_TO_AIC.items()}
    img_ids = sorted(gt.getImgIds())[:args.limit or None]
    os.makedirs(args.out, exist_ok=True)

    dets, t0 = [], time.time()
    for n, img_id in enumerate(img_ids):
        info = gt.loadImgs(img_id)[0]
        path = os.path.join(IMG_DIR, info['file_name'])
        pred = inference_detector(model, path).pred_instances.cpu().numpy()
        for box, score, label in zip(pred.bboxes, pred.scores, pred.labels):
            if int(label) in label_map:
                x1, y1, x2, y2 = (float(v) for v in box)
                dets.append(dict(image_id=img_id, category_id=label_map[int(label)], score=float(score),
                                 bbox=[x1, y1, x2 - x1, y2 - y1]))
        if n == 0:
            img = cv2.imread(path)
            for box, score, label in zip(pred.bboxes, pred.scores, pred.labels):
                if score >= 0.3:
                    x1, y1, x2, y2 = (int(v) for v in box)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, f'{coco_names[int(label)]} {score:.2f}', (x1, max(y1 - 4, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(args.out, f'vis_{info["file_name"]}'), img)
            print(f'first image: {int((pred.scores >= 0.3).sum())} boxes >= 0.3, '
                  f'top classes {[coco_names[int(l)] for l in pred.labels[:5]]}', flush=True)
    print(f'{len(img_ids)} images in {time.time() - t0:.0f} s, {len(dets)} mapped detections')
    with open(os.path.join(args.out, 'mapped_detections.json'), 'w') as f:
        json.dump(dets, f)

    if not dets:
        raise SystemExit('FAIL: no detections of the mapped classes')
    ev = COCOeval(gt, gt.loadRes(dets), 'bbox')
    ev.params.imgIds = img_ids
    ev.params.catIds = sorted(label_map.values())
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    names = {v: k for k, v in aic_id.items()}
    for i, cat in enumerate(ev.params.catIds):
        p = ev.eval['precision'][:, :, i, 0, -1]
        ap = float(np.mean(p[p > -1])) if (p > -1).any() else float('nan')
        print(f'AP {names[cat]:>8s}: {ap:.3f}  ({len(gt.getAnnIds(imgIds=img_ids, catIds=[cat]))} gt boxes)')


if __name__ == '__main__':
    main()
