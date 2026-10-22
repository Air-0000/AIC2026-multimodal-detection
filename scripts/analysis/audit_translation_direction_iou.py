#!/usr/bin/env python3
"""Relaxed-IoU audit of each direction's own actual matched-TP losses."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from analyze_light_val_thresholds import ious
p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--analysis',type=Path,required=True);p.add_argument('--annotations',type=Path,required=True);a=p.parse_args();sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest();m=json.loads((a.bundle/'manifest.json').read_text());assert m['runs'][0]['status']=='complete';assert sha(a.annotations)==m['annotation_sha256'];assert sha(a.bundle/'intervention_outputs.npz')==m['runs'][0]['intervention_outputs_sha256'];pairs=json.loads((a.analysis/'paired_gt.json').read_text());ann=json.loads(a.annotations.read_text());gt={v['id']:v for v in ann['annotations']};ims={v['id']:v for v in ann['images']};z=np.load(a.bundle/'intervention_outputs.npz');pos={int(v):i for i,v in enumerate(z['image_ids'])};out={}
for arm in m['runs'][0]['arms']:
 if arm=='BASELINE':continue
 rows=[]
 for row in pairs:
  if not('interior'in row['groups'] and row['arms']['BASELINE']['detected'] and not row['arms'][arm]['detected']):continue
  g=gt[row['ann_id']];im=ims[g['image_id']];i=pos[g['image_id']];b=z[arm+'_boxes'][i].astype(float).copy();b[:,:2]-=b[:,2:]/2;b*=np.array([im['width'],im['height'],im['width'],im['height']]);ov=ious(g['bbox'],b);ss=1/(1+np.exp(-np.clip(z[arm+'_logits'][i,:,g['category_id']].astype(float),-80,80)));high=ss>=.48
  rows.append(dict(ann_id=g['id'],image_id=g['image_id'],category_id=g['category_id'],best_highscore_iou=float(np.max(np.where(high,ov,0))),score_at_iou={str(t):float(np.max(np.where(ov>=t,ss,0))) for t in [.1,.3,.45,.5]}))
 out[arm]=dict(actual_lost=len(rows),low_score_iou30=[v['ann_id'] for v in rows if v['score_at_iou']['0.3']<.48],low_score_iou10=[v['ann_id'] for v in rows if v['score_at_iou']['0.1']<.48],near_iou45_to50=[v['ann_id'] for v in rows if .45<=v['best_highscore_iou']<.5],details=rows)
(a.analysis/'direction_iou_audit.json').write_text(json.dumps(dict(manifest_sha256=sha(a.bundle/'manifest.json'),paired_gt_sha256=sha(a.analysis/'paired_gt.json'),script_sha256=sha(__file__),scope='Each direction own actual TP losses, allquery scores are oracle diagnostics, not recall.',results=out),indent=2)+'\n');print(json.dumps({k:dict(actual_lost=v['actual_lost'],strong=len(v['low_score_iou30']),near=len(v['near_iou45_to50'])) for k,v in out.items()}))
