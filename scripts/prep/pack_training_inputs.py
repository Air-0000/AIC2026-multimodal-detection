#!/usr/bin/env python3
"""Stage the exact inputs of the clean-baseline configs for transfer to another machine.

The file list comes from the train/val COCO annotations and the training loader's own lookup rules
(IR and LBG: <stem>.png, else the RGB file name; depth: <stem>.png/.jpg/.jpeg). Each source is mapped
to the path the configs expect relative to the repo root, and OUT/ gets symlinks in that layout plus
OUT/work_dir/transfer/MANIFEST.sha256 (sha256sum format). Pinned weights are checked against the SHA
in the configs. With --reference (a clean-baseline acceptance inputs.json) every image must also
hash-match what that run trained on.

  tar -chf aic_inputs.tar -C OUT .                                   # source; -h stores files, not links
  tar -xf aic_inputs.tar -C REPO && cd REPO && sha256sum -c --quiet work_dir/transfer/MANIFEST.sha256
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = {  # target path -> SHA256 pinned in aic_clean_baseline_v1.yml
    'vendor/DEIMv2/ckpts/dfine_x_obj2coco.pth': '39361855c21f8c4757a9bb777a7ff134ae8aa110d0db67a374f081e44c2ce04b',
    'vendor/DEIMv2/ckpts/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth': '8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035',
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 23), b''):
            h.update(block)
    return h.hexdigest()


def lookup(folder, name, exts):
    stem = Path(name).stem
    for cand in [folder/(stem+e) for e in exts] + [folder/name]:
        if cand.is_file():
            return cand
    raise FileNotFoundError(f'{folder}: no file for {name}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', type=Path, required=True, help='staging folder (must not exist)')
    p.add_argument('--visible', type=Path, default=ROOT/'dataset/train/visible')
    p.add_argument('--ir', type=Path, default=ROOT/'dataset/train/infrared_aligned')
    p.add_argument('--lbg', type=Path, default=ROOT/'work_dir/train_lbg_ir')
    p.add_argument('--depth', type=Path, default=ROOT/'dataset/train/depth')
    p.add_argument('--train-ann', type=Path, default=ROOT/'work_dir/deimv2_dataset/train/annotations/instances_train.json')
    p.add_argument('--val-ann', type=Path, default=ROOT/'work_dir/deimv2_dataset/valid/annotations/instances_val.json')
    p.add_argument('--weights-dir', type=Path, default=ROOT/'vendor/DEIMv2/ckpts')
    p.add_argument('--reference', type=Path, help='acceptance inputs.json of the run to match')
    a = p.parse_args()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    entries = {}  # target relative path -> source path
    names = []
    for ann, target in ((a.train_ann, 'work_dir/deimv2_dataset/train/annotations/instances_train.json'),
                        (a.val_ann, 'work_dir/deimv2_dataset/valid/annotations/instances_val.json')):
        entries[target] = ann.resolve()
        names += [im['file_name'] for im in json.loads(ann.read_text())['images']]
    if len(names) != len(set(names)):
        raise SystemExit('duplicate image names across train/val')
    dst = dict(img_folder='dataset/train/visible', ir_folder='dataset/train/infrared_aligned',
               raw_ir_folder='work_dir/train_lbg_ir', depth_folder='dataset/train/depth')
    targets = {}  # image name -> {loader key: target path}
    for n in names:
        src = dict(img_folder=a.visible/n, ir_folder=lookup(a.ir, n, ('.png',)),
                   raw_ir_folder=lookup(a.lbg, n, ('.png',)), depth_folder=lookup(a.depth, n, ('.png', '.jpg', '.jpeg')))
        if not src['img_folder'].is_file():
            raise SystemExit(f'missing visible image {src["img_folder"]}')
        targets[n] = {k: f'{dst[k]}/{s.name}' for k, s in src.items()}
        for k, s in src.items():
            entries[targets[n][k]] = s.resolve()
    for target, pinned in WEIGHTS.items():
        entries[target] = (a.weights_dir/Path(target).relative_to('vendor/DEIMv2/ckpts')).resolve()

    digests = {t: sha256(s) for t, s in sorted(entries.items())}
    bad_weights = [t for t, pinned in WEIGHTS.items() if digests[t] != pinned]
    if bad_weights:
        raise SystemExit(f'weights differ from the SHA pinned in the configs: {bad_weights}')
    checked = 0
    if a.reference:
        ref = {item['file_name']: item['files'] for item in json.loads(a.reference.read_text())}
        if set(ref) != set(names):
            raise SystemExit(f'reference lists {len(ref)} images, annotations {len(names)}')
        diff = []
        for n, keys in targets.items():
            for k, target in keys.items():
                checked += 1
                if digests[target] != ref[n][k]['sha256']:
                    diff.append(f'{n}:{k}')
        if diff:
            raise SystemExit(f'{len(diff)} files differ from the reference run, e.g. {diff[:5]}')

    for t, s in entries.items():
        link = out/t
        link.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(s, link)
    manifest = out/'work_dir/transfer/MANIFEST.sha256'
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(''.join(f'{d}  {t}\n' for t, d in digests.items()))
    size = sum(os.path.getsize(s) for s in entries.values())
    summary = dict(images=len(names), files=len(entries), bytes=size, reference=str(a.reference) if a.reference else None,
                   reference_checked=checked, sources=dict(visible=str(a.visible), ir=str(a.ir), lbg=str(a.lbg), depth=str(a.depth)))
    (out/'work_dir/transfer/summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'{len(names)} images, {len(entries)} files, {size/1e9:.2f} GB staged in {out}'
          + (f'; {checked} image files match {a.reference.name}' if a.reference else ''))
    print(f'next: tar -chf aic_inputs.tar -C {out} .')


if __name__ == '__main__':
    main()
