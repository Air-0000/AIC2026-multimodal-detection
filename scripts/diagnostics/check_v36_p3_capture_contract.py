#!/usr/bin/env python3
"""Compare a frozen pre-fix forward with the installed P3 capture repair.

CPU synthetic-input backbone contract, not dataset AP or full detector parity.
Optional --checkpoint tests learned inference backbone weights, strictly loaded.
"""
import argparse
import ast
import inspect
import json
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts/lib'), str(ROOT / 'vendor/DEIMv2')]


def legacy_forward(cls, source):
    tree = ast.parse(Path(source).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls.__name__)
    fn = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
    loop = next(n for n in fn.body if isinstance(n, ast.For))
    capture = next(n for n in loop.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'idx == 1')
    if len(capture.body) != 4 or not isinstance(capture.body[0], ast.Assign):
        raise ValueError('Expected the pre-fix P3 capture layout in the legacy source')
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    namespace = dict(cls.forward.__globals__)
    exec(compile(module, str(source) + ':legacy_forward', 'exec'), namespace)
    return namespace['forward']


def native_baseline_forward(args, legacy, repaired):
    """One real full-resolution input through the whole frozen detector."""
    import os
    import torch
    from unittest.mock import patch
    from engine.core import YAMLConfig
    from audit_v27_v36_eval_alignment import load_effective, select_state, sha
    baseline = json.loads(args.baseline_report.read_text())
    row = next(r for r in baseline['runs'] if r['name'] == 'V36' and r['status'] == 'evaluated')
    checkpoint = Path(row['checkpoint'])
    config = Path(row['aligned_config'])
    assert sha(checkpoint) == row['checkpoint_sha256']
    assert sha(config) == row['aligned_config_sha256']
    for relative, expected in baseline['source_sha256'].items():
        if relative != 'engine/backbone/hgnetv2_mm.py':
            assert sha(ROOT / 'vendor/DEIMv2' / relative) == expected, f'Other baseline source changed: {relative}'
    os.chdir(ROOT / 'vendor/DEIMv2')
    with patch('engine.core.yaml_config.load_config', load_effective):
        cfg = YAMLConfig(str(config))
    model = cfg.model.float().eval()
    state = select_state(torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True),
                         baseline['selection'], row.get('provenance'))
    state = {k: v for k, v in state.items() if not k.startswith(('dinov3_distiller.', 'co_aux_head.'))}
    model.load_state_dict(state, strict=True)
    del state
    loader = cfg.val_dataloader
    x, targets = next(iter(loader))
    image_id = int(targets[0]['image_id'].item())
    inputs = json.loads((ROOT / 'work_dir/audits/v36_mechanism_20261018/val_inputs.json').read_text())
    item = next(r for r in inputs if r['id'] == image_id)
    for f in item['files'].values():
        assert sha(f['path']) == f['sha256']
    assert list(x.shape[-2:]) == baseline['size']
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model.to(device)
    x = x.to(device)

    def tensors(value, prefix='output'):
        if isinstance(value, torch.Tensor):
            return {prefix: value.detach().cpu().clone()}
        result = {}
        if isinstance(value, dict):
            for k, v in value.items():
                result.update(tensors(v, prefix + '.' + str(k)))
        elif isinstance(value, (list, tuple)):
            for k, v in enumerate(value):
                result.update(tensors(v, prefix + '.' + str(k)))
        return result

    def run(fn):
        captured = []
        def hook(module, inputs, outputs):
            captured.extend(t.detach().cpu().clone() for t in outputs)
        handle = model.backbone.register_forward_hook(hook)
        model.backbone.forward = types.MethodType(fn, model.backbone)
        try:
            with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
                out = tensors(model(x))
        finally:
            handle.remove()
        out.update({f'backbone.P{i+2}': t for i, t in enumerate(captured)})
        return out
    before, after = run(legacy), run(repaired)
    assert before.keys() == after.keys() and 'output.pred_logits' in before and 'output.pred_boxes' in before
    checks = []
    for name in before:
        a, b = before[name], after[name]
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        checks.append({'tensor': name, 'shape': list(a.shape), 'max_abs_diff': (a-b).abs().max().item(), 'exact_equal': True})
    return {'status': 'PASS', 'image_id': image_id, 'input_shape': list(x.shape),
            'config': str(config), 'config_sha256': sha(config), 'checkpoint_sha256': sha(checkpoint),
            'device': str(device), 'precision': 'FP32, TF32 disabled', 'input_files_verified': item['files'],
            'checks': checks, 'scope': 'One real validation image, full detector; not a 200-image metric re-evaluation'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--selection', choices=['model', 'ema.module'], default='model')
    p.add_argument('--out', type=Path, default=ROOT / 'outputs/reports/v36_p3_capture_contract.json')
    p.add_argument('--legacy-source', type=Path, default=ROOT / 'work_dir/audits/v36_p3_capture_fix_20261019/source/hgnetv2_mm_before.py')
    p.add_argument('--baseline-report', type=Path, default=ROOT / 'outputs/reports/v27_v36_split_gate_eval_20261019_final/alignment.json')
    p.add_argument('--native-forward', action='store_true')
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    args.out, args.legacy_source, args.baseline_report = args.out.resolve(), args.legacy_source.resolve(), args.baseline_report.resolve()
    if args.checkpoint:
        args.checkpoint = args.checkpoint.resolve()
    import torch
    from engine.backbone.hgnetv2_mm import HGNetv2MultiModal
    from audit_v27_v36_eval_alignment import sha
    torch.manual_seed(42)
    torch.set_num_threads(2)
    baseline = json.loads(args.baseline_report.read_text())
    assert sha(args.legacy_source) == baseline['source_sha256']['engine/backbone/hgnetv2_mm.py']
    legacy = legacy_forward(HGNetv2MultiModal, args.legacy_source)
    repaired = HGNetv2MultiModal.forward
    model = HGNetv2MultiModal(name='B5', ir_in_chans=2, inject_stages=[2, 3],
                             return_idx=[0, 1, 2, 3], pretrained=False, dcn_stages=[],
                             use_dcda=True, use_dog=True, use_3d_lift=True,
                             pretrained_ir_stem=True, ir_drop_prob=0.05).float().eval()
    if args.checkpoint:
        from audit_v27_v36_eval_alignment import select_state, sha
        provenance_path = args.checkpoint.with_name('provenance.json')
        provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else None
        state = select_state(torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True),
                             args.selection, provenance)
        model.load_state_dict({k[len('backbone.'):]: v for k, v in state.items()
                               if k.startswith('backbone.')}, strict=True)
        del state
    x = torch.randn(1, 6, 64, 96)
    rows = []
    with torch.no_grad():
        for label in ('checkpoint_or_initial', 'nonzero_fusion'):
            if label == 'nonzero_fusion':
                for layer in model.ir_res:
                    torch.nn.init.normal_(layer.weight, std=0.001)
                    torch.nn.init.constant_(layer.bias, 0.1)
            model.forward = types.MethodType(legacy, model)
            before = model(x)
            model.forward = types.MethodType(repaired, model)
            after = model(x)
            for level, (a, b) in enumerate(zip(before, after), start=2):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
                rows.append({'case': label, 'level': f'P{level}', 'max_abs_diff': (a-b).abs().max().item(),
                             'shape': list(a.shape), 'exact_equal': True})
        # Activate stage 1 on a separate correctly dimensioned model. A nonzero
        # residual prevents zero initialization from hiding the capture defect.
        del before, after, model
        model = HGNetv2MultiModal(name='B5', ir_in_chans=2, inject_stages=[1, 2],
                                 return_idx=[0, 1, 2, 3], pretrained=False, dcn_stages=[],
                                 use_dcda=True, use_dog=True, use_3d_lift=True,
                                 pretrained_ir_stem=True).eval()
        captured = []
        def spatial_residual(module, inputs, output):
            h, w = output.shape[-2:]
            yy = torch.arange(h, dtype=output.dtype, device=output.device).view(1, 1, h, 1)
            xx = torch.arange(w, dtype=output.dtype, device=output.device).view(1, 1, 1, w)
            pattern = (yy * 0.03125 + xx * 0.0078125).expand_as(output).clone()
            captured.append(pattern)
            return pattern
        handle = model.ir_res[0].register_forward_hook(spatial_residual)
        model.forward = types.MethodType(legacy, model)
        before = model(x)
        model.forward = types.MethodType(repaired, model)
        after = model(x)
        handle.remove()
        expected = captured[-1]
        delta = after[1] - before[1]
        torch.testing.assert_close(delta, expected, rtol=1e-4, atol=1e-5)
        assert not torch.allclose(delta, expected.flip(-1), rtol=1e-4, atol=1e-5)
        assert all(torch.isfinite(t).all() for t in before + after)
        rows.append({'case': 'stage1_spatial_residual_full_v36_modules', 'level': 'P3',
                     'shape': list(expected.shape), 'max_abs_error': (delta-expected).abs().max().item(),
                     'pattern': 'row*0.03125 + column*0.0078125', 'horizontal_flip_negative_control': True,
                     'scope': 'residual output to P3 coordinates; not DCDA sampling geometry validation'})
    result = {'status': 'PASS', 'scope': 'CPU FP32 backbone, eval mode, synthetic 64x96 input',
              'model_repair_installed': True, 'probe_edits_model_files': False,
              'comparison': 'snapshotted pre-fix forward vs installed repaired forward',
              'source_before_sha256': sha(args.legacy_source),
              'source_after_sha256': sha(inspect.getsourcefile(HGNetv2MultiModal)),
              'checkpoint': str(args.checkpoint) if args.checkpoint else None,
              'limitations': 'Not full detector or real-image parity; MixStyle training behavior not covered',
              'checks': rows}
    if args.checkpoint:
        result['checkpoint_sha256'] = sha(args.checkpoint)
        result['selection'] = args.selection
    if args.native_forward:
        result['native_baseline_forward'] = native_baseline_forward(args, legacy, repaired)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(f'PASS: 8 exact layer comparisons; full-module stage-1 spatial residual contract. {args.out}')


if __name__ == '__main__':
    main()
