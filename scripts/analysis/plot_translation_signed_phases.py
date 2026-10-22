#!/usr/bin/env python3
"""Combine signed phase curves; this is a scientific diagnostic, not model output."""
from pathlib import Path
import json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[2];base=root/'outputs/reports';posfile=base/'translation_phases_20261021/phase_curves.json';negfile=base/'translation_negative_phases_20261021/phase_curves.json';pos={v['ann_id']:v for v in json.loads(posfile.read_text())};neg={v['ann_id']:v for v in json.loads(negfile.read_text())};out=base/'translation_cause_followup_20261021';fig,axes=plt.subplots(3,2,figsize=(11,8),sharex=True,sharey=True);records=[]
for i,(gid,name) in enumerate([(1471,'boat'),(1296,'garbage can'),(9,'light (IoU boundary)')]):
 assert pos[gid]['phases']['BASELINE']==neg[gid]['phases']['BASELINE'];assert pos[gid]['phases']['SHIFT_X1']==neg[gid]['phases']['SHIFT_X1']
 for j,(axis,prefix) in enumerate([('horizontal','X'),('vertical','Y')]):
  values=[]
  for shift in range(-8,9):
   if shift<0:v=neg[gid]['phases'][prefix+'M'+str(-shift)]
   elif shift==0:v=pos[gid]['phases']['BASELINE']
   else:v=pos[gid]['phases']['SHIFT_X1' if prefix=='X' and shift==1 else prefix+str(shift)]
   values.append(v);records.append(dict(ann_id=gid,axis=axis,shift=shift,**v))
  ax=axes[i,j];ax.plot(range(-8,9),[v['gt_score_iou50'] for v in values],'o-',label='GT score at IoU >= .50',markersize=3);ax.plot(range(-8,9),[v['gt_score_iou30'] for v in values],'x--',label='GT score at IoU >= .30',markersize=3);ax.axhline(.48,color='gray',linestyle=':');ax.axvline(0,color='gray',alpha=.3);ax.set_ylim(-.03,1.03);ax.set_title(f'{name}, GT {gid}: {axis}');ax.grid(alpha=.2)
axes[0,0].legend(fontsize=8);axes[2,0].set_xlabel('Input pixels (negative = left)');axes[2,1].set_xlabel('Input pixels (negative = up)');axes[1,0].set_ylabel('Oracle GT-class score (not matched recall)');fig.suptitle('Frozen V36: signed translation phases on symptom-selected cases');fig.tight_layout();fig.savefig(out/'signed_phase_curves.png',dpi=160);plt.close(fig);sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();(out/'signed_phase_curves.json').write_text(json.dumps(dict(positive_source_sha256=sha(posfile),negative_source_sha256=sha(negfile),script_sha256=sha(Path(__file__)),records=records),indent=2)+'\n')
