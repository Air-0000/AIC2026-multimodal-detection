#!/usr/bin/env python3
"""Plot a causal reference intervention while keeping query and GT outcomes distinct."""
from pathlib import Path
import json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
root=Path(__file__).resolve().parents[2];source=root/'outputs/reports/single_query_geometry_interventions_20261021';casefile=source/'initial_reference_cases.json';case=next(v for v in json.loads(casefile.read_text())['rows'] if v['ann_id']==1471);details=json.loads((source/'gt_details.json').read_text());arms=['BASELINE','BASELINE_H_TO_SHIFT','BASELINE_CY_TO_SHIFT','SHIFT_X1_LOCKED'];labels=['Original','Only q162 H changed','Only q162 Cy changed','Right 1px, IDs locked'];own=[case['arms'][a]['same_baseline_token_score'] for a in arms];best=[case['arms'][a]['best_valid_score'] for a in arms];iou=[case['arms'][a]['same_baseline_token_iou'] for a in arms];tp=[details[a]['1471']['detected'] for a in arms];fig,axes=plt.subplots(1,2,figsize=(12,4.6));x=np.arange(4);axes[0].bar(x-.18,own,.36,label='Original query 162: boat score');axes[0].bar(x+.18,best,.36,label='Best boat score at IoU >= .5');axes[0].axhline(.48,color='gray',linestyle=':');axes[0].set_ylim(0,1.1);axes[0].set_ylabel('GT-class score (oracle query diagnostic)');axes[0].legend(fontsize=8)
for i,v in enumerate(tp):axes[0].text(i,1.02,'Matched GT: '+('yes' if v else 'no'),ha='center',fontsize=8)
axes[1].bar(x,iou,color='seagreen');axes[1].set_ylim(0,1.1);axes[1].set_ylabel('Final IoU of original query 162')
for i,v in enumerate(iou):axes[1].text(i,v+.02,f'{v:.4f}',ha='center',fontsize=9)
for ax in axes:ax.set_xticks(x,labels,rotation=15,ha='right',fontsize=8);ax.grid(axis='y',alpha=.2)
fig.suptitle('Frozen V36, boat GT1471: initial geometry can collapse one query while another compensates');fig.tight_layout();out=root/'outputs/reports/translation_cause_followup_20261021';fig.savefig(out/'boat_reference_counterfactual.png',dpi=160);plt.close(fig);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();(out/'boat_reference_counterfactual.json').write_text(json.dumps(dict(source_sha256=sha(casefile),script_sha256=sha(Path(__file__)),arms=arms,same_query_scores=own,best_gt_scores=best,same_query_final_iou=iou,actual_gt_detected=tp,scope='Same input/content/memory for first3 arms, only one initial geometry scalar of query162 changed. Last arm is separate shifted input. Historical V36, selected case, not a fix or general-population conclusion.'),indent=2)+'\n')
