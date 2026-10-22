#!/usr/bin/env python3
"""Offline scoring diagnostic: replacing sigmoid by class-softmax is not a calibrated fix."""
import json,sys,contextlib,io
from pathlib import Path
from collections import defaultdict
import numpy as np
from pycocotools.coco import COCO
from analyze_light_val_thresholds import nms,evaluate,metrics,scan,dump,sha
root=Path(__file__).resolve().parents[2];out=root/'outputs/reports/translation_class_margin_20261021';out.mkdir(exist_ok=False)
b=root/'work_dir/audits/single_image_stability_20261021/raw';z=np.load(b/'decoder_outputs.npz');m=json.loads((b/'manifest.json').read_text());assert sha(b/'decoder_outputs.npz')==m['runs'][0]['decoder_outputs_sha256']
with contextlib.redirect_stdout(io.StringIO()):c=COCO(str(root/'work_dir/deimv2_dataset/valid/annotations/instances_val.json'))
light=next(k for k,v in c.cats.items() if v['name']=='light');results={};matched={}
for arm in ['BASELINE','SHIFT_X1']:
 predictions=[]
 for i,iid in enumerate(z['image_ids']):
  iid=int(iid);log=z[arm+'_logits'][i].astype(float);ex=np.exp(log-log.max(1,keepdims=True));score=ex/ex.sum(1,keepdims=True);flat=score.ravel();indices=np.argsort(-flat,kind='stable')[:300];im=c.imgs[iid];boxes=z[arm+'_boxes'][i].astype(float).copy();boxes[:,:2]-=boxes[:,2:]/2;boxes*=np.array([im['width'],im['height'],im['width'],im['height']]);rr=[dict(image_id=iid,category_id=int(k%12),bbox=boxes[k//12].tolist(),score=float(flat[k])) for k in indices];predictions.extend(nms(rr,.6))
 e=evaluate(c,predictions);ss=scan(e,light);tp=set()
 for ev in e.evalImgs:
  if ev is None or ev['aRng']!=e.params.areaRng[0]:continue
  for j,s in enumerate(ev['dtScores']):
   if s>=.48 and not ev['dtIgnore'][0,j] and ev['dtMatches'][0,j]>0:tp.add(int(ev['dtMatches'][0,j]))
 matched[arm]=tp;results[arm]=dict(metrics=metrics(e,light),operating_points=[v for v in ss if v['threshold']==.48 and v['iou']==.5]);print(arm,results[arm],flush=True)
old=json.loads((root/'outputs/reports/single_image_stability_20261021/analysis/paired_gt.json').read_text());lost={r['ann_id'] for r in old if 'interior' in r['groups'] and r['arms']['BASELINE']['detected'] and not r['arms']['SHIFT_X1']['detected']};results['original_lost19_recovered_in_shift']=sorted(lost&matched['SHIFT_X1']);results['contract']='Offline diagnostic only. Softmax among12 classes has no background/objectness channel and forces every query into some class. Retop300 from all300x12 logits, same geometry and NMS; invalid to call improved detection from rescued cases alone. Fixed.48 operating point has different calibration; no threshold tuning or production change.';dump(out/'summary.json',results)
