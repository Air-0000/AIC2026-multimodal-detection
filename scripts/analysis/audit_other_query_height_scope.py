#!/usr/bin/env python3
"""Verify the target query is unchanged while other query heights are restored."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest();m=json.loads((a.bundle/'manifest.json').read_text());assert m['runs'][0]['status']=='complete';assert sha(a.bundle/'intervention_outputs.npz')==m['runs'][0]['intervention_outputs_sha256'];z=np.load(a.bundle/'intervention_outputs.npz');pos={int(v):i for i,v in enumerate(z['image_ids'])};records=json.loads((a.bundle/'mask_records.json').read_text());rows=[]
for r in records:
 if r['arm']!='OTHERS_H':continue
 i=pos[r['image_id']];q=r['single_query_index'];before=z['SHIFT_X1_LOCKED_init_ref_unact'][i];after=z['OTHERS_H_init_ref_unact'][i];diff=before!=after;assert np.array_equal(after[q],before[q]);assert not diff[:,:3].any();assert np.array_equal(z['OTHERS_H_ids'][i],z['BASELINE_ids'][i]);assert np.all(after[:,3][diff[:,3]]==z['BASELINE_init_ref_unact'][i,:,3][diff[:,3]]);rows.append(dict(image_id=r['image_id'],excluded_target_query=q,changed_other_height_queries=np.where(diff[:,3])[0].tolist(),changed_count=int(diff.sum()),all4_target_reference_coordinates_bitwise_equal=True))
assert len(rows)==19;a.out.write_text(json.dumps(dict(manifest_sha256=sha(a.bundle/'manifest.json'),script_sha256=sha(__file__),rows=rows),indent=2)+'\n');print('OTHER_QUERY_HEIGHT_SCOPE_VERIFIED',len(rows))
