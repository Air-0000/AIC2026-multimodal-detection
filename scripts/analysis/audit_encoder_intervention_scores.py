#!/usr/bin/env python3
"""Audit all-query scores at relaxed IoU, separately from matched detections."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from analyze_light_val_thresholds import ious

def main():
 p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--annotations',type=Path,required=True);p.add_argument('--iou-audit',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
 m=json.loads((a.bundle/'manifest.json').read_text());r=m['runs'][0];assert r['status']=='complete';assert sha(a.annotations)==m['annotation_sha256'];assert sha(a.bundle/'intervention_outputs.npz')==r['intervention_outputs_sha256']
 ann=json.loads(a.annotations.read_text());images={v['id']:v for v in ann['images']};gt={v['id']:v for v in ann['annotations']};audit=json.loads(a.iou_audit.read_text());strong=set(audit['SHIFT_X1']['low_gt_score_even_at_iou30']);near=set(audit['SHIFT_X1']['high_score_near_miss_iou45_to50']);z=np.load(a.bundle/'intervention_outputs.npz');pos={int(x):i for i,x in enumerate(z['image_ids'])};rows={}
 for arm in r['arms']:
  rows[arm]={}
  for gid in sorted(strong|near):
   g=gt[gid];im=images[g['image_id']];i=pos[g['image_id']];b=z[arm+'_boxes'][i].astype(float).copy();b[:,:2]-=b[:,2:]/2;b*=np.array([im['width'],im['height'],im['width'],im['height']]);ov=ious(g['bbox'],b);scores=1/(1+np.exp(-np.clip(z[arm+'_logits'][i,:,g['category_id']].astype(float),-80,80)))
   rows[arm][str(gid)]={str(t):float(np.max(np.where(ov>=t,scores,0))) for t in [.1,.3,.5]}
 summary={arm:dict(strong13_high_at_iou30=[g for g in sorted(strong) if vals[str(g)]['0.3']>=.48],strong13_high_at_iou50=[g for g in sorted(strong) if vals[str(g)]['0.5']>=.48],near4_high_at_iou30=[g for g in sorted(near) if vals[str(g)]['0.3']>=.48]) for arm,vals in rows.items()}
 a.out.write_text(json.dumps(dict(scope='Oracle all-query same-class score >=.48 at relaxed IoU; not matched TP/recall. Symptom-selected cohort.',manifest_sha256=sha(a.bundle/'manifest.json'),annotation_sha256=sha(a.annotations),iou_audit_sha256=sha(a.iou_audit),script_sha256=sha(__file__),summary=summary,per_gt=rows),indent=2)+'\n');print(json.dumps(summary))
if __name__=='__main__':main()
