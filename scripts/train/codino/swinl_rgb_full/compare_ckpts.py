"""Side-by-side of mmengine checkpoints: file, top-level keys, meta (epoch / iter / seed / time / versions / dataset
classes / the embedded config's model, preprocessor, loader and data lines), and the state_dict (tensor count,
parameter count, keys only in one, shape or dtype changes, how many shared tensors differ and by how much).
Usage: python compare_ckpts.py NAME=PATH NAME=PATH [...]   (pairwise against the first and between consecutive ones)
"""
import hashlib
import re
import sys

import torch

CFG_PAT = re.compile(r"type='(CoDETR|AICFrozenCoDETR|DetDataPreprocessor|AICNightGateDataPreprocessor|LoadImageFromFile|"
                     r"LoadAICGateInputs|AICRandomNightBGR)'|data_root=|ann_file='(annotations|splitA_train)|^load_from|"
                     r"^work_dir|trainable=|gate=dict")


def sha(path, n=1 << 24):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(n):
            h.update(chunk)
    return h.hexdigest()[:16]


def describe(name, path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    sd = ck['state_dict']
    meta = ck.get('meta', {})
    print(f'\n##### {name}: {path}')
    print('  file sha256', sha(path), ' top-level keys', sorted(ck))
    for k in ('epoch', 'iter', 'seed', 'time', 'mmengine_version', 'mmengine', 'experiment_name'):
        if k in meta:
            print(f'  meta.{k} = {meta[k]}')
    dm = meta.get('dataset_meta') or {}
    if dm:
        print('  meta.dataset_meta.classes =', dm.get('classes'))
    print('  meta keys', sorted(meta))
    cfg = meta.get('cfg', '')
    if cfg:
        lines = [ln.strip() for ln in cfg.splitlines() if CFG_PAT.search(ln.strip())]
        print('  embedded cfg lines:', *sorted(set(lines)), sep='\n    ')
    n = sum(v.numel() for v in sd.values() if torch.is_tensor(v))
    dt = sorted({str(v.dtype) for v in sd.values() if torch.is_tensor(v)})
    print(f'  state_dict: {len(sd)} tensors, {n:,} values, dtypes {dt}')
    return sd


def compare(a_name, a, b_name, b):
    only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
    shared = [k for k in a if k in b]
    shape = [k for k in shared if a[k].shape != b[k].shape or a[k].dtype != b[k].dtype]
    same = [k for k in shared if k not in shape and torch.equal(a[k], b[k])]
    diff = [k for k in shared if k not in shape and k not in same]
    print(f'\n=== {a_name} vs {b_name}')
    print(f'  only in {a_name}: {len(only_a)} {only_a[:12]}')
    print(f'  only in {b_name}: {len(only_b)} {only_b[:12]}')
    print(f'  shared {len(shared)}: bit-identical {len(same)}, different values {len(diff)}, shape/dtype changed {len(shape)}')
    if diff:
        rel = sorted(((float((a[k].float() - b[k].float()).abs().max()), k) for k in diff), reverse=True)
        print('  largest |diff|:', [(k, round(d, 5)) for d, k in rel[:3]])
        print('  median rel. change:', round(float(torch.tensor([
            float((a[k].float() - b[k].float()).norm() / (b[k].float().norm() + 1e-12)) for k in diff]).median()), 4))


def main():
    items = [arg.split('=', 1) for arg in sys.argv[1:]]
    sds = {name: describe(name, path) for name, path in items}
    names = [n for n, _ in items]
    pairs = [(names[0], n) for n in names[1:]] + [(names[i], names[i + 1]) for i in range(1, len(names) - 1)]
    for x, y in pairs:
        compare(x, sds[x], y, sds[y])


if __name__ == '__main__':
    main()
