#!/usr/bin/env python3
"""Generate aligned IR and LBG IR for any image set exactly the way the training set was generated.

Wraps scripts/prep/batch_generate_server_ir_lbg.process_single_image, which reproduces the training
files byte for byte: PNG keeps the raw IR, JPG goes through align_image_pair_v3, and LBG is the cv2
median top-hat (k=31) of the saved aligned IR. Use it for test sets: work_dir/test_aligned_ir was made
with the older align_ir_pixel_pipeline.py (v1) and does not match training (MODEL_RECORDS §三十九).

Output: OUT/infrared_aligned, OUT/lbg_ir and OUT/manifest.json (per-file SHA256 and branch). OUT must not exist.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.prep.batch_generate_server_ir_lbg import process_single_image, compute_sha256

IMAGE_EXTS = ('.jpg', '.jpeg', '.png')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--vis-dir', type=Path, required=True)
    p.add_argument('--ir-dir', type=Path, required=True, help='raw infrared images, same stems as --vis-dir')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    vis_dir, ir_dir, out = a.vis_dir.resolve(), a.ir_dir.resolve(), a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    aligned, lbg = out/'infrared_aligned', out/'lbg_ir'
    aligned.mkdir()
    lbg.mkdir()
    vis = {Path(f).stem: f for f in os.listdir(vis_dir) if f.lower().endswith(IMAGE_EXTS)}
    irs = sorted(f for f in os.listdir(ir_dir) if f.lower().endswith(IMAGE_EXTS))
    missing = [f for f in irs if Path(f).stem not in vis]
    if not irs or missing:
        sys.exit(f'{len(irs)} IR images, {len(missing)} without a visible image: {missing[:5]}')
    tasks = [(f, str(vis_dir/vis[Path(f).stem]), str(ir_dir/f), str(aligned/f), str(lbg/f)) for f in irs]
    with ProcessPoolExecutor(a.workers) as ex:
        results = list(ex.map(process_single_image, tasks, chunksize=4))
    failed = [r for r in results if r['status'] != 'success']
    manifest = dict(vis_dir=str(vis_dir), ir_dir=str(ir_dir), count=len(results), failed=len(failed),
                    branches=dict(Counter(r['align_info'].get('branch') for r in results)),
                    generator=str(ROOT/'scripts/prep/batch_generate_server_ir_lbg.py'),
                    generator_sha256=compute_sha256(ROOT/'scripts/prep/batch_generate_server_ir_lbg.py'),
                    wrapper_sha256=compute_sha256(__file__), files=results)
    (out/'manifest.json').write_text(json.dumps(manifest, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
    print(f"{len(results)} images, {len(failed)} failed, branches {manifest['branches']} -> {out}")
    if failed:
        sys.exit(f'failed: {[r["file_name"] for r in failed[:5]]}')


if __name__ == '__main__':
    main()
