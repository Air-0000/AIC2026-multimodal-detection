#!/usr/bin/env python3
"""Verify single-query geometry interventions changed no other initial coordinate."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest();m=json.loads((a.bundle/'manifest.json').read_text());r=m['runs'][0];assert r['status']=='complete';assert sha(a.bundle/'intervention_outputs.npz')==r['intervention_outputs_sha256'];z=np.load(a.bundle/'intervention_outputs.npz');pos={int(v):i for i,v in enumerate(z['image_ids'])};rows=json.loads((a.bundle/'mask_records.json').read_text());checked=[]
for row in rows:
 if 'single_query_index' not in row:continue
 arm=row['arm'];i=pos[row['image_id']];q=row['single_query_index'];reverse=arm.startswith('BASELINE_');control='BASELINE' if reverse else 'SHIFT_X1_LOCKED';before=z[control+'_init_ref_unact'][i];after=z[arm+'_init_ref_unact'][i];coords=[3] if arm in ['Q_H_BASE','BASELINE_H_TO_SHIFT'] else [1] if arm in ['Q_CY_BASE','Q_CY_WINNER','BASELINE_CY_TO_SHIFT'] else [0,1,2,3];allowed=np.zeros_like(before,dtype=bool);allowed[q,coords]=True;changed=before!=after;assert not np.any(changed&~allowed),(row,np.argwhere(changed&~allowed));assert np.array_equal(z[arm+'_ids'][i],z['BASELINE_ids'][i]);source='SHIFT_X1_LOCKED' if reverse else 'BASELINE'
 for c in coords:
  if c!=0:assert after[q,c]==z[source+'_init_ref_unact'][i,q,c]
 checked.append(dict(image_id=row['image_id'],arm=arm,query=q,allowed_coordinates=coords,changed_coordinates=np.argwhere(changed).tolist(),prior_cy_fallback=row.get('prior_cy_no_valid_query_fallback',False)))
assert len(checked)==19*6
a.out.write_text(json.dumps(dict(manifest_sha256=sha(a.bundle/'manifest.json'),mask_records_sha256=sha(a.bundle/'mask_records.json'),script_sha256=sha(__file__),scope='Bitwise initial-reference/tokenID audit. Other decoder content/memory unchanged by source hook logic; final states can change through attention.',checked=checked),indent=2)+'\n');print('SINGLE_QUERY_SCOPE_VERIFIED',len(checked),'fallbacks',[(r['image_id'],r['arm']) for r in checked if r['prior_cy_fallback']])
