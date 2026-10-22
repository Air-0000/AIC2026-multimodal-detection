#!/usr/bin/env python3
"""Reconstruct declared masks and verify equal-area random controls."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--annotations',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest();m=json.loads((a.bundle/'manifest.json').read_text());assert m['runs'][0]['status']=='complete';assert sha(a.annotations)==m['annotation_sha256'];ann=json.loads(a.annotations.read_text());gt={v['id']:v for v in ann['annotations']};ims={v['id']:v for v in ann['images']};records=json.loads((a.bundle/'mask_records.json').read_text());by={(r['image_id'],r['level'],r['arm']):r for r in records};out=[]
for row in records:
 iid,j,arm=row['image_id'],row['level'],row['arm'];g=gt[row['ann_id']];im=ims[iid];h,w=[(180,320),(90,160),(45,80)][j];assert row['total_positions']==h*w;x,y,bw,bh=g['bbox'];x=x*2560/im['width'];y=y*1440/im['height'];bw=bw*2560/im['width'];bh=bh*1440/im['height'];x0=max(0,int(np.floor((x-64)*w/2560)));x1=min(w,int(np.ceil((x+bw+64)*w/2560)));y0=max(0,int(np.floor((y-64)*h/1440)));y1=min(h,int(np.ceil((y+bh+64)*h/1440)));area=(x1-x0)*(y1-y0);assert area==row['local_positions']
 if arm.startswith('ENC_RANDOM64'):
  offset=0 if arm=='ENC_RANDOM64' else int(arm.rsplit('_S',1)[1])*1000000;rng=np.random.default_rng(42000+offset+iid*10+j);rx=int(rng.integers(0,w-(x1-x0)+1));ry=int(rng.integers(0,h-(y1-y0)+1));assert row['replaced_positions']==area==by[(iid,j,'ENC_LOCAL64')]['replaced_positions'];overlap=max(0,min(x1,rx+x1-x0)-max(x0,rx))*max(0,min(y1,ry+y1-y0)-max(y0,ry));out.append(dict(image_id=iid,ann_id=g['id'],level=j,arm=arm,area=area,local_fraction=area/(h*w),random_overlap_with_local=overlap,random_fraction_overlapping_local=overlap/area))
 elif arm=='ENC_LOCAL64':assert row['replaced_positions']==area
 elif arm=='ENC_OUTSIDE64':assert row['replaced_positions']==h*w-area
 else:raise AssertionError(arm)
a.out.write_text(json.dumps(dict(manifest_sha256=sha(a.bundle/'manifest.json'),mask_records_sha256=sha(a.bundle/'mask_records.json'),script_sha256=sha(__file__),assumed_spatial_shapes=[[180,320],[90,160],[45,80]],verified_records=len(records),random_regions=out),indent=2)+'\n');print('MASK_RECORDS_VERIFIED',len(records))
