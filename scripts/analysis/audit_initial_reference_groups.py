#!/usr/bin/env python3
"""Compare initial geometry instability using the same baseline token per GT."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from analyze_light_val_thresholds import ious
p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--annotations',type=Path,required=True);p.add_argument('--paired-gt',type=Path,required=True);p.add_argument('--iou-audit',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest();m=json.loads((a.bundle/'manifest.json').read_text());assert sha(a.annotations)==m['annotation_sha256'];assert sha(a.bundle/'intervention_outputs.npz')==m['runs'][0]['intervention_outputs_sha256'];assert sha(a.paired_gt)==m['intervention_contract']['paired_gt_sha256'];z=np.load(a.bundle/'intervention_outputs.npz');pos={int(v):i for i,v in enumerate(z['image_ids'])};assert np.array_equal(z['BASELINE_ids'],z['SHIFT_X1_LOCKED_ids']);ann=json.loads(a.annotations.read_text());gt={v['id']:v for v in ann['annotations']};ims={v['id']:v for v in ann['images']};paired=json.loads(a.paired_gt.read_text());strong=set(json.loads(a.iou_audit.read_text())['SHIFT_X1']['low_gt_score_even_at_iou30']);rows=[]
for old in paired:
 if old['image_id'] not in pos or 'interior' not in old['groups'] or not old['arms']['BASELINE']['detected']:continue
 g=gt[old['ann_id']];im=ims[g['image_id']];i=pos[g['image_id']];b=z['BASELINE_boxes'][i].astype(float).copy();b[:,:2]-=b[:,2:]/2;b*=np.array([im['width'],im['height'],im['width'],im['height']]);ov=ious(g['bbox'],b);s=1/(1+np.exp(-np.clip(z['BASELINE_logits'][i,:,g['category_id']].astype(float),-80,80)));q=int(np.argmax(np.where(ov>=.5,s,-1)));assert ov[q]>=.5 and s[q]>=.48
 boxes=[]
 for arm in ['BASELINE','SHIFT_X1_LOCKED']:
  v=z[arm+'_init_ref_unact'][i,q].astype(float);v=1/(1+np.exp(-np.clip(v,-80,80)));v*=np.array([2560,1440,2560,1440]);v[0]-=0 if arm=='BASELINE' else 1;boxes.append(v)
 base,shift=boxes;d=shift-base;gwh=np.array([g['bbox'][2]*2560/im['width'],g['bbox'][3]*1440/im['height']]);logwh=np.log(np.maximum(shift[2:],1e-9)/np.maximum(base[2:],1e-9));center=np.sqrt(np.sum((d[:2]/gwh)**2));basecorner=base.copy();basecorner[:2]-=basecorner[2:]/2;shiftcorner=shift.copy();shiftcorner[:2]-=shiftcorner[2:]/2;initial_iou=float(ious(basecorner,shiftcorner[None,:])[0]);groups=['stable225' if old['arms']['SHIFT_X1']['detected'] else 'lost19'];
 if g['id'] in strong:groups.append('strong13')
 rows.append(dict(ann_id=g['id'],image_id=g['image_id'],category_id=g['category_id'],groups=groups,query=q,token_id=int(z['BASELINE_ids'][i,q]),baseline_initial_box_inputpixels=base.tolist(),shifted_initial_box_inputpixels=shift.tolist(),delta_inputpixels=d.tolist(),center_delta_normalized_by_gt_wh=float(center),size_log_change_norm=float(np.linalg.norm(logwh)),initial_box_pair_iou=initial_iou))
summary={}
for group,n in [('stable225',225),('lost19',19),('strong13',13)]:
 rr=[r for r in rows if group in r['groups']];assert len(rr)==n;summary[group]=dict(n=n,median_center_delta_normalized_by_gt_wh=float(np.median([r['center_delta_normalized_by_gt_wh'] for r in rr])),median_size_log_change_norm=float(np.median([r['size_log_change_norm'] for r in rr])),median_initial_box_pair_iou=float(np.median([r['initial_box_pair_iou'] for r in rr])))
a.out.write_text(json.dumps(dict(scope='Same baseline highest-valid-score query token, initial encoder boxes inverse-mapped to original frame. Descriptive symptom-selected cohort, not causal population statistics; initial boxes differ from final boxes.',manifest_sha256=sha(a.bundle/'manifest.json'),script_sha256=sha(__file__),summary=summary,rows=rows),indent=2)+'\n');print(json.dumps(summary))
