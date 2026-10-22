#!/usr/bin/env python3
"""GT score/IoU phase curves, with one-to-one TP outcomes retained separately."""
import argparse,json
from pathlib import Path
import numpy as np
from analyze_light_val_thresholds import ious,dump,sha
p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--analysis',type=Path,required=True);p.add_argument('--annotations',type=Path,required=True);p.add_argument('--iou-audit',type=Path,required=True);a=p.parse_args();m=json.loads((a.bundle/'manifest.json').read_text());assert m['runs'][0]['status']=='complete';assert sha(a.bundle/'intervention_outputs.npz')==m['runs'][0]['intervention_outputs_sha256'];assert sha(a.annotations)==m['annotation_sha256'];z=np.load(a.bundle/'intervention_outputs.npz');pos={int(v):i for i,v in enumerate(z['image_ids'])};ann=json.loads(a.annotations.read_text());gt={v['id']:v for v in ann['annotations']};ims={v['id']:v for v in ann['images']};audit=json.loads(a.iou_audit.read_text());strong=set(audit['SHIFT_X1']['low_gt_score_even_at_iou30']);near=set(audit['SHIFT_X1']['high_score_near_miss_iou45_to50']);actual=json.loads((a.analysis/'gt_details.json').read_text());losses=m['intervention_contract']['cohort_lost_gt_ids'];groups=json.loads((a.analysis/'summary.json').read_text());stable=set(groups['stable_controls']);records=[]
for aid in losses:
 g=gt[aid];im=ims[g['image_id']];i=pos[g['image_id']];row=dict(ann_id=aid,image_id=g['image_id'],category_id=g['category_id'],strong_score=aid in strong,near_boundary=aid in near,phases={})
 for arm in m['intervention_contract']['arms']:
  b=z[arm+'_boxes'][i].astype(float).copy();b[:,:2]-=b[:,2:]/2;b*=np.array([im['width'],im['height'],im['width'],im['height']]);ov=ious(g['bbox'],b);ss=1/(1+np.exp(-np.clip(z[arm+'_logits'][i,:,g['category_id']].astype(float),-80,80)));high=ss>=.48
  row['phases'][arm]=dict(gt_score_iou50=float(np.max(np.where(ov>=.5,ss,0))),gt_score_iou30=float(np.max(np.where(ov>=.3,ss,0))),best_highscore_iou=float(np.max(np.where(high,ov,0))),actual_detected=actual[arm][str(aid)]['detected'])
 records.append(row)
summary={}
for axis,arms in [('x',['SHIFT_X1']+[f'X{k}' for k in range(2,9)]),('y',[f'Y{k}' for k in range(1,9)])]:
 summary[axis]=dict(arms=arms,strong13_recovered_any=sorted(v['ann_id'] for v in records if v['strong_score'] and any(v['phases'][arm]['actual_detected'] for arm in arms)),strong13_oracle_score_range=[dict(ann_id=v['ann_id'],min_score=min(v['phases'][arm]['gt_score_iou50'] for arm in arms),max_score=max(v['phases'][arm]['gt_score_iou50'] for arm in arms)) for v in records if v['strong_score']],stable225_retained_all=sum(all(actual[arm][str(aid)]['detected'] for arm in arms) for aid in stable),lost19_recovered_any=sum(any(v['phases'][arm]['actual_detected'] for arm in arms) for v in records))
dump(a.analysis/'phase_curves.json',records);dump(a.analysis/'phase_curve_summary.json',summary)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(3,2,figsize=(10,8),sharex=True,sharey=True)
for i,aid in enumerate([1471,1296,9]):
 row=next(v for v in records if v['ann_id']==aid)
 for j,(axis,arms) in enumerate([('right',['BASELINE','SHIFT_X1']+[f'X{k}' for k in range(2,9)]),('down',['BASELINE']+[f'Y{k}' for k in range(1,9)])]):
  ax=axes[i,j];ax.plot(range(9),[row['phases'][arm]['gt_score_iou50'] for arm in arms],marker='o',label='GT score, IoU >= .50');ax.plot(range(9),[row['phases'][arm]['gt_score_iou30'] for arm in arms],marker='x',linestyle='--',label='GT score, IoU >= .30');ax.axhline(.48,color='gray',linestyle=':');ax.set_title(f'GT {aid}: {axis} shift');ax.set_ylim(-.03,1.03);ax.grid(alpha=.2)
axes[0,0].legend(fontsize=8);axes[2,0].set_xlabel('Input pixels');axes[2,1].set_xlabel('Input pixels');axes[1,0].set_ylabel('Oracle class score (not detection recall)');fig.suptitle('Frozen V36: independent-image translation phases, symptom-selected cases');fig.tight_layout();fig.savefig(a.analysis/'phase_curves.png',dpi=160);plt.close(fig)
print(json.dumps(summary,ensure_ascii=False))
