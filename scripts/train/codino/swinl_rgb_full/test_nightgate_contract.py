"""GPU contract checks for the night-gated Co-DINO (aic_nightgate.py); no training beyond a 4-image smoke run.

  1 gate        image_gate: luma 0.06 → 0, 0.02 → 1, 0.035 → 0.5
  2 loader      LoadAICGateInputs: 6 channels; channels 0-2 bit-identical to LoadImageFromFile
  3 normal      on split-A held-out images the gated model's predictions equal the pure-RGB model's exactly (same
                checkpoint), every gate is 0; control: the pure-RGB model run twice gives equal predictions
  4 night       at night01 the gate opens; with the zero-initialised net the output still equals the pure-RGB model's on
                the same dark image; a non-zero net changes it; with thermal / depth zeroed it is the pure-RGB output again
  5 frozen      a 24-image training run changes only data_preprocessor.fuse.*; every other tensor of the saved
                checkpoint is bit-identical to the pure-RGB checkpoint
Usage: python test_nightgate_contract.py --base-cfg RGB_CFG --gate-cfg GATED_CFG --ckpt RGB_CKPT --work-dir TMP
"""
from __future__ import annotations

import argparse
import copy
import json
import os

import numpy as np
import torch
from mmengine.config import Config
from mmengine.runner import Runner

import aic_nightgate as ng
from mmcv.transforms import LoadImageFromFile
from mmdet.apis import inference_detector, init_detector


def preds(model, path):
    with torch.no_grad():
        p = inference_detector(model, path).pred_instances
    return p.bboxes.cpu(), p.scores.cpu(), p.labels.cpu()


def same(a, b):
    return all(torch.equal(x, y) for x, y in zip(a, b))


def with_probe(cfg, *extra):
    cfg = copy.deepcopy(cfg)
    pipe = list(cfg.test_dataloader.dataset.pipeline)
    cfg.test_dataloader.dataset.pipeline = [pipe[0], *extra, *pipe[1:]]
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-cfg', required=True)
    ap.add_argument('--gate-cfg', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--work-dir', required=True)
    ap.add_argument('--n', type=int, default=6)
    a = ap.parse_args()
    base_cfg, gate_cfg = Config.fromfile(a.base_cfg), Config.fromfile(a.gate_cfg)
    root = base_cfg.val_dataloader.dataset.data_root
    ims = sorted(json.load(open(root + 'splitA_val.json'))['images'], key=lambda x: x['file_name'])
    paths = [root + 'images/' + im['file_name'] for im in ims[:: max(1, len(ims) // a.n)][: a.n]]

    # 1
    for v, want in ((0.06, 0.0), (0.02, 1.0), (0.035, 0.5)):
        g = float(ng.image_gate(torch.full((8, 8), v), 0.02, 0.05))
        assert abs(g - want) < 1e-6, (v, g)
    print('PASS 1 gate')

    # 2
    loader = ng.LoadAICGateInputs(**{k: v for k, v in gate_cfg.test_dataloader.dataset.pipeline[0].items() if k != 'type'})
    for p in paths[:3]:
        six = loader(dict(img_path=p))['img']
        three = LoadImageFromFile()(dict(img_path=p))['img']
        assert six.shape[2] == 6 and np.array_equal(six[..., :3], three), p
    print('PASS 2 loader')

    # 3
    base = init_detector(base_cfg, a.ckpt, device='cuda:0')
    gated = init_detector(gate_cfg, a.ckpt, device='cuda:0')
    for p in paths:
        pb, pg, pb2 = preds(base, p), preds(gated, p), preds(base, p)
        assert same(pb, pb2), f'pure-RGB model not repeatable on {p}'
        assert gated.data_preprocessor.last_gates == [0.0], (p, gated.data_preprocessor.last_gates)
        assert same(pb, pg), f'gated output differs from pure RGB in normal light: {p}'
    print(f'PASS 3 normal ({len(paths)} images: gate 0, predictions identical)')

    # 4
    night = dict(type='AICNightProbeBGR', level=0.01)
    base_n = init_detector(with_probe(base_cfg, night), a.ckpt, device='cuda:0')
    gated_n = init_detector(with_probe(gate_cfg, night), a.ckpt, device='cuda:0')
    gated_nz = init_detector(with_probe(gate_cfg, night, dict(type='AICZeroAux')), a.ckpt, device='cuda:0')
    p = paths[0]
    pb, pg = preds(base_n, p), preds(gated_n, p)
    g = gated_n.data_preprocessor.last_gates[0]
    assert g > 0.5, g
    assert same(pb, pg), 'zero-initialised net changed the output'
    for m in (gated_n, gated_nz):
        torch.nn.init.constant_(m.data_preprocessor.fuse.net[-1].bias, 0.05)
    assert not same(pb, preds(gated_n, p)), 'non-zero net did not change the output'
    assert same(pb, preds(gated_nz, p)), 'thermal / depth zeroed but the output changed'
    print(f'PASS 4 night (gate {g:.3f})')
    del base, gated, base_n, gated_n, gated_nz
    torch.cuda.empty_cache()

    # 5
    cfg = copy.deepcopy(gate_cfg)
    cfg.work_dir = a.work_dir
    cfg.load_from = a.ckpt
    # fp16 dynamic loss scaling starts at 2^16 and skips the first overflowing steps (4 steps were all skipped)
    cfg.train_dataloader.dataset.indices = 24
    cfg.train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=1, val_interval=2)
    cfg.param_scheduler = [dict(type='ConstantLR', factor=1.0, by_epoch=False, begin=0, end=10)]
    for h in cfg.custom_hooks:
        if h.get('type') == 'AicIterRecorderHook':
            h['out_file'] = os.path.join(a.work_dir, 'iter_log_first.jsonl')
    Runner.from_cfg(cfg).train()
    new = torch.load(os.path.join(a.work_dir, 'epoch_1.pth'), map_location='cpu', weights_only=False)['state_dict']
    old = torch.load(a.ckpt, map_location='cpu', weights_only=False)['state_dict']
    fuse = [k for k in new if k.startswith('data_preprocessor.fuse.')]
    rest = [k for k in new if not k.startswith('data_preprocessor.fuse.')]
    assert fuse and set(rest) == set(old), (len(fuse), set(rest) ^ set(old))
    diff = [k for k in rest if not torch.equal(new[k], old[k])]
    assert not diff, f'frozen tensors changed: {diff[:10]}'
    assert new['data_preprocessor.fuse.net.8.weight'].abs().sum() > 0, 'fusion net did not train'
    print(f'PASS 5 frozen ({len(rest)} detector tensors bit-identical, {len(fuse)} fusion tensors trained)')
    print('NIGHTGATE_CONTRACT_OK')


if __name__ == '__main__':
    main()
