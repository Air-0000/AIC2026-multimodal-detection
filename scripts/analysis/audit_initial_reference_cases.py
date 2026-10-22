#!/usr/bin/env python3
"""Bind recovered class scores to query identity and exported initial geometry."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from analyze_light_val_thresholds import ious
p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--annotations',type=Path,required=True);p.add_argument('--iou-audit',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest();m=json.loads((a.bundle/'manifest.json').read_text());r=m['runs'][0];assert r['status']=='complete';assert sha(a.annotations)==m['annotation_sha256'];assert sha(a.bundle/'intervention_outputs.npz')==r['intervention_outputs_sha256'];z=np.load(a.bundle/'intervention_outputs.npz');pos={int(v):i for i,v in enumerate(z['image_ids'])};ann=json.loads(a.annotations.read_text());gt={v['id']:v for v in ann['annotations']};ims={v['id']:v for v in ann['images']};strong=json.loads(a.iou_audit.read_text())['SHIFT_X1']['low_gt_score_even_at_iou30'];rows=[]
for gid in strong:
 g=gt[gid];im=ims[g['image_id']];i=pos[g['image_id']];baseq=None;out={}
 for arm in r['arms']:
  boxes=z[arm+'_boxes'][i].astype(float).copy();boxes[:,:2]-=boxes[:,2:]/2;boxes*=np.array([im['width'],im['height'],im['width'],im['height']]);ov=ious(g['bbox'],boxes);ss=1/(1+np.exp(-np.clip(z[arm+'_logits'][i,:,g['category_id']].astype(float),-80,80)));q=int(np.argmax(np.where(ov>=.5,ss,-1)));valid=ov[q]>=.5
  if arm=='BASELINE':baseq=q
  refs=z[arm+'_init_ref_unact'][i].astype(float);refs=1/(1+np.exp(-np.clip(refs,-80,80)));refs*=np.array([2560,1440,2560,1440]);refs[:,0]-=0 if arm=='BASELINE' or arm.startswith('BASELINE_') else 1
  btoken=int(z['BASELINE_ids'][i,baseq]);matches=np.where(z[arm+'_ids'][i]==btoken)[0];same=int(matches[0]) if len(matches) else None
  out[arm]=dict(best_valid_query_index=q if valid else None,best_valid_token_id=int(z[arm+'_ids'][i,q]) if valid else None,best_valid_score=float(ss[q]) if valid else 0.,best_valid_iou=float(ov[q]) if valid else 0.,best_valid_initial_box_originalframe_inputpixels=refs[q].tolist() if valid else None,same_baseline_token_query=same,same_baseline_token_score=float(ss[same]) if same is not None else None,same_baseline_token_iou=float(ov[same]) if same is not None else None,same_baseline_token_initial_box_originalframe_inputpixels=refs[same].tolist() if same is not None else None)
 rows.append(dict(ann_id=gid,image_id=g['image_id'],category_id=g['category_id'],baseline_query=baseq,arms=out))
a.out.write_text(json.dumps(dict(scope='All-query oracle scores/geometry and query provenance; matched TP kept in sibling summary. Initial encoder boxes are not final boxes.',manifest_sha256=sha(a.bundle/'manifest.json'),script_sha256=sha(__file__),rows=rows),indent=2)+'\n')
for row in rows:
 if row['ann_id'] in [1296,1471]:print(row['ann_id'],{k:(round(v['best_valid_score'],4),v['best_valid_query_index'],v['best_valid_token_id']) for k,v in row['arms'].items()})
