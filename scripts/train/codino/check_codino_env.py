"""Checks the Co-DINO environment on the target GPU before any training; exits non-zero on the first failure.

Covers: torch has kernels for this GPU's architecture; mmcv's compiled ops run on it and agree with their references
(multi-scale deformable attention forward and backward against mmcv's PyTorch implementation, NMS and RoIAlign against
CPU); mmdet / mmengine versions and the Co-DETR project import; whether fairscale checkpointing is available (Co-DETR
silently trains without it otherwise); torch.load accepting pickled numpy (mmengine resume). With --config it also
loads the training config and checks its class count, label files, category names and image files.

Run from the repo root with the codino env's python and codino_env.sh sourced:
    python scripts/train/codino/check_codino_env.py --config scripts/train/codino/co_dino_swin_l_aic_rgb_1800.py
"""
import argparse
import importlib
import json
import os
import sys
import tempfile

import numpy as np
import torch


def fail(msg):
    print(f'FAIL: {msg}')
    sys.exit(1)


def check_gpu():
    if not torch.cuda.is_available():
        fail('CUDA not available')
    major, minor = torch.cuda.get_device_capability()
    arch = f'sm_{major}{minor}'
    print(f'torch {torch.__version__} cuda {torch.version.cuda} cudnn {torch.backends.cudnn.version()}')
    print(f'gpu {torch.cuda.get_device_name()} {arch}; torch arch list {torch.cuda.get_arch_list()}')
    if arch not in torch.cuda.get_arch_list():
        fail(f'this torch build has no {arch} kernels')


def check_mmcv_ops():
    import mmcv
    from mmcv.ops import RoIAlign, nms
    from mmcv.ops.multi_scale_deform_attn import (MultiScaleDeformableAttnFunction,
                                                  multi_scale_deformable_attn_pytorch)
    print(f'mmcv {mmcv.__version__}')

    torch.manual_seed(0)
    bs, heads, head_dim, points, queries = 2, 8, 32, 4, 300
    shapes = torch.tensor([[64, 96], [32, 48], [16, 24], [8, 12]], device='cuda')
    levels = len(shapes)
    starts = torch.cat((shapes.new_zeros(1), shapes.prod(1).cumsum(0)[:-1]))
    keys = int(shapes.prod(1).sum())
    inputs = [torch.rand(bs, keys, heads, head_dim, device='cuda'),
              torch.rand(bs, queries, heads, levels, points, 2, device='cuda'),
              torch.rand(bs, queries, heads, levels, points, device='cuda')]
    inputs[2] = inputs[2] / inputs[2].sum((-1, -2), keepdim=True)
    cuda_in = [t.clone().requires_grad_() for t in inputs]
    ref_in = [t.clone().requires_grad_() for t in inputs]
    out = MultiScaleDeformableAttnFunction.apply(cuda_in[0], shapes, starts, cuda_in[1], cuda_in[2], 64)
    ref = multi_scale_deformable_attn_pytorch(ref_in[0], shapes, ref_in[1], ref_in[2])
    grad = torch.rand_like(ref)
    out.backward(grad)
    ref.backward(grad)
    diffs = [(out - ref).abs().max().item()] + [(c.grad - r.grad).abs().max().item() for c, r in zip(cuda_in, ref_in)]
    print('ms_deform_attn max |cuda - pytorch|: out {:.2e}, d_value {:.2e}, d_loc {:.2e}, d_attn {:.2e}'.format(*diffs))
    # Bilinear sampling on the CUDA kernel accumulates in a different order; 1e-3 is far above float32 rounding here.
    if max(diffs) > 1e-3:
        fail('multi-scale deformable attention CUDA kernel disagrees with the PyTorch reference')

    xy = torch.rand(500, 2) * 500
    boxes = torch.cat([xy, xy + 20 + torch.rand(500, 2) * 80], 1)
    scores = torch.rand(500)
    _, keep_cpu = nms(boxes, scores, 0.5)
    _, keep_gpu = nms(boxes.cuda(), scores.cuda(), 0.5)
    if not torch.equal(keep_cpu.sort().values, keep_gpu.cpu().sort().values):
        fail('NMS keeps different boxes on CPU and GPU')
    feat = torch.rand(2, 16, 64, 64)
    rois = torch.cat([torch.randint(0, 2, (50, 1)).float(), boxes[:50] / 10], 1)
    align = RoIAlign(7, spatial_scale=1.0, sampling_ratio=0, aligned=True)
    roi_diff = (align(feat, rois) - align(feat.cuda(), rois.cuda()).cpu()).abs().max().item()
    print(f'nms cpu/gpu identical; roi_align max |cpu - gpu| {roi_diff:.2e}')
    if roi_diff > 1e-4:
        fail('RoIAlign differs between CPU and GPU')


def check_frameworks():
    import mmdet
    import mmengine
    print(f'mmengine {mmengine.__version__} mmdet {mmdet.__version__}')
    importlib.import_module('projects.CO-DETR.codetr')
    importlib.import_module('aic_codino_runtime')
    print(f'Co-DETR project imported; TF32 matmul {torch.backends.cuda.matmul.allow_tf32} '
          f'cudnn {torch.backends.cudnn.allow_tf32}')
    try:
        from fairscale.nn.checkpoint import checkpoint_wrapper  # noqa: F401
        print('fairscale checkpoint_wrapper available (encoder activation checkpointing on)')
    except Exception as exc:  # Co-DETR catches the same failure and trains without checkpointing
        print(f'WARNING: fairscale import failed ({exc!r}); the encoder will train without checkpointing')
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'probe.pth')
        torch.save({'array': np.zeros(2)}, path)
        try:
            torch.load(path)
        except Exception as exc:
            fail(f'torch.load rejects pickled numpy ({exc!r}); export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1')


def check_config(path):
    from mmengine.config import Config
    cfg = Config.fromfile(path)
    classes = tuple(cfg.train_dataloader.dataset.metainfo['classes'])
    heads = {'query_head': cfg.model.query_head.num_classes,
             'roi_head[0]': cfg.model.roi_head[0].bbox_head.num_classes,
             'bbox_head[0]': cfg.model.bbox_head[0].num_classes}
    print(f'config {path}: {len(classes)} classes, head num_classes {heads}')
    if any(n != len(classes) for n in heads.values()):
        fail('a head has a different class count than metainfo')
    if not os.path.isfile(cfg.load_from):
        fail(f'load_from not found: {cfg.load_from} (run from the repo root)')
    for name, loader in (('train', cfg.train_dataloader), ('val', cfg.val_dataloader)):
        ds = loader.dataset
        ann_path = os.path.join(ds.data_root, ds.ann_file)
        img_dir = os.path.join(ds.data_root, ds.data_prefix['img'])
        with open(ann_path) as f:
            ann = json.load(f)
        names = tuple(c['name'] for c in sorted(ann['categories'], key=lambda c: c['id']))
        if names != classes:
            fail(f'{name}: label file categories {names} != config classes {classes}')
        missing = [im['file_name'] for im in ann['images'] if not os.path.isfile(os.path.join(img_dir, im['file_name']))]
        print(f'{name}: {ann_path} {len(ann["images"])} images, {len(ann["annotations"])} boxes, '
              f'{len(missing)} image files missing')
        if missing:
            fail(f'{name}: missing images, e.g. {missing[:3]}')
    if cfg.val_evaluator.ann_file != os.path.join(cfg.val_dataloader.dataset.data_root,
                                                  cfg.val_dataloader.dataset.ann_file):
        fail('val_evaluator.ann_file is not the val label file')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--config', help='training config to check as well')
    args = parser.parse_args()
    check_gpu()
    check_mmcv_ops()
    check_frameworks()
    if args.config:
        check_config(args.config)
    print('CODINO_ENV_OK')


if __name__ == '__main__':
    main()
