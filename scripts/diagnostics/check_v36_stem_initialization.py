#!/usr/bin/env python3
"""CPU-only audit of IR stem initialization before/after the actual -t loader.

Compares the full transferred 64x2x3x3 tensor against the documented channel
mean/Laplacian transform. No training, re-transfer, model-file edit or GPU use.
"""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts/lib'), str(ROOT / 'vendor/DEIMv2')]


def transformed(rgb):
    import torch
    mean = rgb.mean(dim=1, keepdim=True)
    lap = rgb.new_tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3) * .5
    return torch.cat([mean.expand(-1, 2, -1, -1), (mean * .5 + lap * .5).expand(-1, 2, -1, -1)], dim=0).clone()


def difference(a, b):
    import torch
    delta = (a-b).abs()
    return {'exact_equal': torch.equal(a, b), 'max_abs': delta.max().item(), 'mean_abs': delta.mean().item()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=ROOT / 'vendor/DEIMv2/ckpts/dfine_x_obj2coco.pth')
    p.add_argument('--out', type=Path, default=ROOT / 'outputs/reports/v36_stem_initialization_20261019')
    args = p.parse_args()
    args.checkpoint, args.out = args.checkpoint.resolve(), args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    import torch
    from engine.backbone.hgnetv2_mm import HGNetv2MultiModal
    from engine.solver._solver import BaseSolver
    from audit_v27_v36_eval_alignment import sha
    assert not torch.cuda.is_initialized(), 'Run this probe in a fresh CPU-only process'
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = torch.nn.Module()
    model.backbone = HGNetv2MultiModal(
        name='B5', ir_in_chans=2, inject_stages=[2, 3], return_idx=[0, 1, 2, 3],
        pretrained=False, pretrained_ir_stem=True, dcn_stages=[],
        use_dcda=True, use_dog=True, use_3d_lift=True, ir_drop_prob=.05).cpu()
    rgb_key = 'backbone.rgb_net.stem.stem1.conv.weight'
    ir_key = 'backbone.ir_stem.stem.0.weight'
    before = {k: model.state_dict()[k].clone() for k in (rgb_key, ir_key)}
    expected_init = transformed(before[rgb_key])
    torch.testing.assert_close(before[ir_key], expected_init, rtol=0, atol=0)

    load = torch.load
    def cpu_load(*pos, **kw):
        kw.update(map_location='cpu', weights_only=False, mmap=True)
        return load(*pos, **kw)
    checkpoint = cpu_load(args.checkpoint)
    selection = 'ema.module' if 'ema' in checkpoint else ('model' if 'model' in checkpoint else 'bare_state_dict')
    state = checkpoint['ema']['module'] if selection == 'ema.module' else checkpoint.get('model', checkpoint)
    source_key = rgb_key if rgb_key in state else rgb_key.replace('backbone.rgb_net.', 'backbone.')
    expected_loaded = state[source_key].clone()
    source_ir_keys = [k for k in state if k.startswith('backbone.ir_stem.')]
    assert not source_ir_keys, 'Source includes IR weights; this is not the COCO-only initialization case'
    # Exercise the real loader, including its selection, key mapping and fallback;
    # skip _setup(), which would instantiate optimizer/DDP and choose a GPU.
    solver = BaseSolver(None)
    solver.model = model
    log = io.StringIO()
    with contextlib.redirect_stdout(log), patch('torch.load', cpu_load):
        solver.load_tuning_state(str(args.checkpoint))
    after = {k: model.state_dict()[k].clone() for k in (rgb_key, ir_key)}
    torch.testing.assert_close(after[rgb_key], expected_loaded, rtol=0, atol=0)
    torch.testing.assert_close(after[ir_key], before[ir_key], rtol=0, atol=0)
    assert not torch.equal(before[rgb_key], after[rgb_key])
    expected_pretrained = transformed(after[rgb_key])
    assert not torch.equal(after[ir_key], expected_pretrained)
    assert not torch.cuda.is_initialized()
    tensors = {'rgb_before': before[rgb_key], 'rgb_after': after[rgb_key],
               'ir_before': before[ir_key], 'ir_after': after[ir_key],
               'expected_from_random_rgb': expected_init, 'expected_from_loaded_rgb': expected_pretrained}
    torch.save(tensors, args.out / 'stem_weights.pt')
    (args.out / 'loader.log').write_text(log.getvalue())
    matrix = {stage: {reference: difference(actual, expected) for reference, expected in
                     [('transformed_random_rgb', expected_init), ('transformed_loaded_rgb', expected_pretrained)]}
              for stage, actual in [('ir_after_constructor', before[ir_key]), ('ir_after_tuning', after[ir_key])]}
    report = {'status': 'PASS', 'finding': 'IR stem copied transformed random RGB before COCO loading; -t updates RGB and leaves IR unchanged',
              'seed': 42, 'device': 'cpu', 'cuda_initialized': torch.cuda.is_initialized(),
              'checkpoint': str(args.checkpoint), 'checkpoint_sha256': sha(args.checkpoint),
              'selection': selection, 'source_rgb_key': source_key, 'target_rgb_key': rgb_key,
              'target_ir_key': ir_key, 'rgb_shape': list(before[rgb_key].shape), 'ir_shape': list(before[ir_key].shape),
              'rgb_before_after': difference(before[rgb_key], after[rgb_key]),
              'ir_before_after': difference(before[ir_key], after[ir_key]), 'difference_matrix': matrix,
              'loader': 'BaseSolver.load_tuning_state (actual implementation; backbone-only wrapper)',
              'weights_sha256': sha(args.out / 'stem_weights.pt'), 'script_sha256': sha(__file__),
              'source_sha256': {f: sha(ROOT / 'vendor/DEIMv2' / f) for f in
                               ['engine/solver/_solver.py', 'engine/backbone/hgnetv2_mm.py',
                                'engine/backbone/hgnetv2.py', 'engine/backbone/dynamic_multimodal_modules.py']},
              'scope': 'Fresh COCO warm-start ordering, not trained V36 weights, AP loss or proof that random IR initialization is harmful'}
    (args.out / 'initialization.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'difference_matrix': matrix, 'cuda_initialized': False}, indent=2))


if __name__ == '__main__':
    main()
