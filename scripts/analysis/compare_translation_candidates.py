#!/usr/bin/env python3
"""Compare candidates on common original-positive GT to prevent denominator artifacts."""
import argparse,json
from pathlib import Path
from analyze_light_val_thresholds import dump,sha
p=argparse.ArgumentParser();p.add_argument('--analysis',type=Path,required=True);p.add_argument('--candidates',nargs='+',required=True);p.add_argument('--iou-audit',type=Path,required=True);a=p.parse_args();rows=json.loads((a.analysis/'paired_gt.json').read_text());audit=json.loads(a.iou_audit.read_text());strong=set(audit['SHIFT_X1']['low_gt_score_even_at_iou30']);near=set(audit['SHIFT_X1']['high_score_near_miss_iou45_to50']);result={}
for candidate in a.candidates:
 shifted=candidate+'_SHIFT_X1';rr=[r for r in rows if 'interior' in r['groups']];base_orig={r['ann_id'] for r in rr if r['arms']['BASELINE']['detected']};base_shift={r['ann_id'] for r in rr if r['arms']['SHIFT_X1']['detected']};orig={r['ann_id'] for r in rr if r['arms'][candidate]['detected']};shift={r['ann_id'] for r in rr if r['arms'][shifted]['detected']};common=base_orig&orig;stable=base_orig&base_shift;lost=base_orig-base_shift
 result[candidate]=dict(original_common_positive_count=len(common),base_losses_on_common=len(common-base_shift),candidate_losses_on_common=len(common-shift),original_tp_lost_vs_base=sorted(base_orig-orig),original_tp_gained_vs_base=sorted(orig-base_orig),original_lost19_rescued=sorted(lost&shift),strong13_rescued=sorted(strong&shift),near4_rescued=sorted(near&shift),new_loss_on_originally_stable=sorted(stable-shift),common_orig_base_loss_rescued=sorted((common-base_shift)&shift),common_orig_new_loss=sorted((common&base_shift)-shift))
dump(a.analysis/'common_positive_comparison.json',dict(paired_gt_sha256=sha(a.analysis/'paired_gt.json'),iou_audit_sha256=sha(a.iou_audit),results=result));print(json.dumps(result,ensure_ascii=False))
