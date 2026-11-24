"""Pure-RGB counterpart of the teammate's fused_all2000 (scripts/cloud/dump_fused_images.py): same file names, same
annotation files, but each PNG holds only the visible image, decoded the way his TriModalCocoDetection decodes it
(PIL open + convert('RGB'), float32 /255, then rint(clip(x)*255) back to uint8) with no infrared or depth residual.

Checks: PNG read-back equals the array; size equals the annotation; on --fused-sample PNGs of his, the B channel
(which his formula never touches) must be identical, so the JPEG decode matches the one behind his training images.

Usage: python dump_rgb_images.py --visible DIR --meta DIR(with annotations.json, splitA_*.json) --out DIR
       [--fused-sample DIR] [--workers N]
"""
import argparse
import hashlib
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import PIL
from PIL import Image


def decode(src):
    rgb = Image.open(src).convert('RGB')
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    out = np.rint(np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    if not np.array_equal(out, np.asarray(rgb)):
        raise RuntimeError(f'float round trip changed pixels: {src}')
    return out


def work(job):
    src, dst, w, h = job
    arr = decode(src)
    Image.fromarray(arr).save(dst, compress_level=1)
    back = np.asarray(Image.open(dst).convert('RGB'))
    if back.shape != arr.shape or not np.array_equal(back, arr):
        raise RuntimeError(f'PNG read-back differs: {dst}')
    if (arr.shape[1], arr.shape[0]) != (w, h):
        raise RuntimeError(f'size differs from annotation: {dst} {arr.shape} vs {w}x{h}')
    return dst.name, hashlib.sha256(dst.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--visible', type=Path, required=True)
    ap.add_argument('--meta', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--fused-sample', type=Path)
    ap.add_argument('--workers', type=int, default=32)
    a = ap.parse_args()
    ann = json.loads((a.meta / 'annotations.json').read_text())
    by_stem = {}
    for p in a.visible.iterdir():
        if p.stem in by_stem:
            raise RuntimeError(f'two visible files share a stem: {p}')
        by_stem[p.stem] = p
    img_dir = a.out / 'images'
    img_dir.mkdir(parents=True, exist_ok=False)
    jobs = []
    for im in ann['images']:
        stem = Path(im['file_name']).stem
        if Path(im['file_name']).suffix != '.png' or stem not in by_stem:
            raise RuntimeError(f'no visible source for {im["file_name"]}')
        jobs.append((by_stem[stem], img_dir / im['file_name'], im['width'], im['height']))
    with ProcessPoolExecutor(a.workers) as ex:
        shas = dict(ex.map(work, jobs, chunksize=8))
    for f in ('annotations.json', 'splitA_train.json', 'splitA_val.json'):
        shutil.copy2(a.meta / f, a.out / f)
    sample = []
    if a.fused_sample:
        for fp in sorted(a.fused_sample.glob('*.png')):
            fused = np.asarray(Image.open(fp).convert('RGB'))
            ours = np.asarray(Image.open(img_dir / fp.name).convert('RGB'))
            if fused.shape != ours.shape or not np.array_equal(fused[..., 2], ours[..., 2]):
                raise RuntimeError(f'B channel differs from the fused image: {fp.name}')
            sample.append(dict(file=fp.name, r_changed=float((fused[..., 0] != ours[..., 0]).mean()),
                               g_changed=float((fused[..., 1] != ours[..., 1]).mean())))
    meta_sha = {f: hashlib.sha256((a.meta / f).read_bytes()).hexdigest()
                for f in ('annotations.json', 'splitA_train.json', 'splitA_val.json')}
    manifest = dict(source='visible only (no infrared / depth residual)', visible_dir=str(a.visible),
                    pillow=PIL.__version__, numpy=np.__version__, annotation_sha256=meta_sha,
                    images=len(shas), files=shas, fused_b_channel_check=sample)
    (a.out / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: v for k, v in manifest.items() if k != 'files'}, indent=1))
    print('RGB_DUMP_OK', len(shas))


if __name__ == '__main__':
    main()
