#!/usr/bin/env python3
"""Audit IoU threshold confounding before labeling score collapse."""
from pathlib import Path
import json,numpy as np
from analyze_light_val_thresholds import ious,dump
root=Path(__file__).resolve().parents[2];out=root/'outputs/reports/translation_cause_validation_20261021';out.mkdir(exist_ok=True)
ann=json.loads((root/'work_dir/deimv2_dataset/valid/annotations/instances_val.json').read_text());gt={g['id']:g for g in ann['annotations']};ims={i['id']:i for i in ann['images']};pairs=json.loads((root/'outputs/reports/single_image_stability_20261021/analysis/paired_gt.json').read_text());lost=[r for r in pairs if 'interior' in r['groups'] and r['arms']['BASELINE']['detected'] and not r['arms']['SHIFT_X1']['detected']];assert len(lost)==19
z=np.load(root/'work_dir/audits/single_image_stability_20261021/raw/decoder_outputs.npz');pos={int(v):i for i,v in enumerate(z['image_ids'])};rows=[]
for r in lost:
 g=gt[r['ann_id']];im=ims[g['image_id']];i=pos[g['image_id']];d=dict(ann_id=g['id'],image_id=g['image_id'],category_id=g['category_id'],arms={})
 for arm in ['BASELINE','SHIFT_X1','SHIFT_X1_LOCKED']:
  b=z[arm+'_boxes'][i].astype(float).copy();b[:,:2]-=b[:,2:]/2;b*=np.array([im['width'],im['height'],im['width'],im['height']]);ov=ious(g['bbox'],b);ss=1/(1+np.exp(-np.clip(z[arm+'_logits'][i,:,g['category_id']].astype(float),-80,80)));high=ss>=.48
  d['arms'][arm]=dict(best_iou_of_high_score=float(ov[high].max()) if high.any() else 0.,max_score_at_iou={str(t):float(np.max(np.where(ov>=t,ss,0))) for t in [.1,.3,.4,.45,.48,.49,.5,.51,.55,.75]},high_score_queries_at_iou={str(t):int(np.count_nonzero(high&(ov>=t))) for t in [.3,.4,.45,.49,.5]})
 rows.append(d)
summary={}
for arm in ['SHIFT_X1','SHIFT_X1_LOCKED']:
 summary[arm]=dict(n=19,high_score_near_miss_iou40_to50=[r['ann_id'] for r in rows if .4<=r['arms'][arm]['best_iou_of_high_score']<.5],high_score_near_miss_iou45_to50=[r['ann_id'] for r in rows if .45<=r['arms'][arm]['best_iou_of_high_score']<.5],low_gt_score_even_at_iou30=[r['ann_id'] for r in rows if r['arms'][arm]['max_score_at_iou']['0.3']<.48],low_gt_score_even_at_iou10=[r['ann_id'] for r in rows if r['arms'][arm]['max_score_at_iou']['0.1']<.48])
dump(out/'iou_boundary_summary.json',summary);dump(out/'iou_boundary_gt.json',rows)
print(json.dumps(summary,ensure_ascii=False))
for r in rows:
 if r['ann_id'] in [1471,1296,9]:print(r['ann_id'],r['arms']['SHIFT_X1'])
