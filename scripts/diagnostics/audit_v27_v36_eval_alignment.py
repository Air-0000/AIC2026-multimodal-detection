#!/usr/bin/env python3
"""Freeze a common validation protocol; optionally evaluate explicitly supplied checkpoints.

Example (paths supplied relative to the caller's directory):
  python scripts/diagnostics/audit_v27_v36_eval_alignment.py \
    --run V27 CONFIG CHECKPOINT --run V36 CONFIG CHECKPOINT --selection model
Add --evaluate --device cuda:0 to score. Preparation alone never produces AP.
"""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
FRAMEWORK = ROOT / 'vendor/DEIMv2'
sys.path[:0] = [str(ROOT / 'scripts/lib'), str(FRAMEWORK)]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def load_effective(path):
    # Framework load_config uses a mutable default cfg={}. An explicit fresh
    # dictionary prevents one model/training config from contaminating another.
    from engine.core.yaml_utils import load_config
    return copy.deepcopy(load_config(str(path), cfg={}))


def depth_image_stats(path):
    """Statistics use the dataset reader's channel mean, scale and clipping."""
    import numpy as np
    from PIL import Image
    with Image.open(path) as image:
        mode, fmt = image.mode, image.format
        raw = np.asarray(image)
    if raw.ndim == 3:
        # Integer channel sum preserves the unique-value count after averaging.
        counts = np.bincount(raw.astype(np.int32).sum(axis=2).ravel())
        values = np.flatnonzero(counts).astype(np.float64) / raw.shape[2]
    else:
        counts = np.bincount(raw.astype(np.int32).ravel())
        values = np.flatnonzero(counts).astype(np.float64)
    freq = counts[counts > 0]
    scale = 19999.0 if mode in ('I;16', 'I') else 255.0
    normalized = np.clip(values / scale, 0, 1)
    zero = float(freq[normalized == 0].sum() / freq.sum())
    return {'path': str(path), 'format': fmt, 'mode': mode, 'dtype': str(raw.dtype),
            'uniq': int(len(values)), 'zero_fraction': zero,
            'ge_098_fraction': float(freq[normalized >= .98].sum() / freq.sum()),
            'reader_scale': scale,
            'warning_8bit_majority_zero': mode not in ('I;16', 'I') and zero > .5}


def depth_contract(dataset, path_maps=()):
    folder = str(dataset['depth_folder'])
    for source, target in path_maps:
        if folder == source or folder.startswith(source.rstrip('/') + '/'):
            folder = str(Path(target) / folder[len(source):].lstrip('/'))
            break
    return {'depth_folder': str(Path(folder).resolve()),
            'depth_read': dataset.get('depth_read', 'raw16'),
            'depth_mode': dataset.get('depth_mode', 'raw')}


def select_state(checkpoint, selection, provenance=None):
    # No implicit EMA/model fallback: every row uses the requested policy.
    if selection == 'ema.module' and 'ema' not in checkpoint:
        if not provenance or provenance.get('selection') != 'ema.module':
            raise ValueError('No EMA state and no explicit frozen-EMA provenance')
        state = checkpoint['model']  # An explicitly documented EMA export.
    else:
        state = checkpoint['ema']['module'] if selection == 'ema.module' else checkpoint['model']
    if selection == 'model' and provenance and provenance.get('selection') == 'ema.module' and 'ema' not in checkpoint:
        raise ValueError('Frozen export contains EMA, not raw model; request --selection ema.module')
    if not isinstance(state, dict) or not state:
        raise ValueError('Selected state is empty or not a dictionary')
    return {k.removeprefix('module.'): v for k, v in state.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', nargs=3, action='append', metavar=('NAME', 'CONFIG', 'CHECKPOINT'), required=True)
    p.add_argument('--selection', choices=['model', 'ema.module'], required=True)
    p.add_argument('--reference', type=Path, default=ROOT / 'work_dir/audits/v36_mechanism_20261018')
    p.add_argument('--out', type=Path, default=ROOT / 'outputs/reports/v27_v36_eval_alignment')
    p.add_argument('--size', type=int, nargs=2, default=[1440, 2560])
    p.add_argument('--evaluate', action='store_true')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--training-config', nargs=2, action='append', default=[], metavar=('NAME', 'CONFIG'),
                   help='Original training config; otherwise the run config is inspected')
    p.add_argument('--path-map', nargs=2, action='append', default=[], metavar=('SOURCE_ROOT', 'LOCAL_ROOT'),
                   help='Explicit relocation for comparing depth source paths; recorded in report')
    args = p.parse_args()
    args.run = [(n, Path(c).resolve(), Path(w).resolve()) for n, c, w in args.run]
    args.reference, args.out = args.reference.resolve(), args.out.resolve()
    training_configs = {n: Path(c).resolve() for n, c in args.training_config}
    if set(training_configs) - {n for n, _, _ in args.run}:
        p.error('--training-config names must match --run names')
    if len({n for n, _, _ in args.run}) != len(args.run):
        p.error('Run names must be unique')
    if any(x <= 0 or x % 32 for x in args.size):
        p.error('Image dimensions must be positive multiples of 32')
    args.out.mkdir(parents=True, exist_ok=False)
    os.chdir(FRAMEWORK)
    import yaml

    manifest = json.loads((args.reference / 'manifest.json').read_text())
    inputs_path = args.reference / 'val_inputs.json'
    if sha(inputs_path) != manifest['val_inputs_sha256']:
        raise ValueError('Reference val_inputs.json hash changed')
    inputs = json.loads(inputs_path.read_text())
    if len(inputs) != 200 or len({x['id'] for x in inputs}) != 200:
        raise ValueError('Expected 200 unique validation images')
    baseline = manifest['configs']['baseline']
    if sha(baseline['path']) != baseline['sha256']:
        raise ValueError('Frozen reference baseline configuration changed')
    reference = load_effective(baseline['path'])
    val = copy.deepcopy(reference['val_dataloader'])
    ann = Path(val['dataset']['ann_file']).resolve()
    if sha(ann) != manifest['val_annotations_sha256']:
        raise ValueError('Reference validation annotation changed')
    annotations = json.loads(ann.read_text())
    if {(x['id'], x['file_name']) for x in annotations['images']} != {(x['id'], x['name']) for x in inputs}:
        raise ValueError('Annotation IDs/names do not match frozen inputs')
    depth_stats = []
    for item in inputs:
        for key in ('img_folder', 'ir_folder', 'raw_ir_folder', 'depth_folder'):
            folder = Path(val['dataset'][key]).resolve()
            stem = Path(item['name']).stem
            candidates = ([folder / item['name']] if key == 'img_folder' else
                          [folder / (stem + '.png'), folder / item['name']])
            if key == 'depth_folder':
                candidates = [folder / (stem + e) for e in ('.png', '.jpg', '.jpeg')]
            actual = next((f for f in candidates if f.is_file()), None)
            expected = item['files'][key]
            if actual is None or sha(actual) != expected['sha256']:
                raise ValueError(f'Input mismatch: {item["name"]} {key}')
            if key == 'depth_folder':
                depth_stats.append(dict(id=item['id'], name=item['name'], **depth_image_stats(actual)))
    dump(args.out / 'depth_health.json', depth_stats)
    with (args.out / 'depth_health.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(depth_stats[0]))
        writer.writeheader()
        writer.writerows(depth_stats)
    ops = val['dataset']['transforms']['ops']
    if [op['type'] for op in ops] != ['Resize', 'Normalize', 'ConvertBoxes']:
        raise ValueError('Unexpected validation transforms; inspect before aligning')
    ops[0]['size'] = args.size
    val.update(total_batch_size=1, num_workers=0, drop_last=False)
    val.pop('persistent_workers', None)
    if isinstance(val.get('collate_fn'), dict):
        val['collate_fn']['base_size'] = args.size[1]
        val['collate_fn']['base_size_repeat'] = None
    report = {'kind': 'aligned_checkpoint_evaluation', 'selection': args.selection,
              'size': args.size, 'precision': 'FP32; TF32 disabled', 'tta': False,
              'annotation_sha256': sha(ann), 'val_inputs_sha256': sha(inputs_path),
              'input_files_verified': 800, 'common_val_dataloader': val,
              'common_postprocessor': reference.get('PostProcessor'),
              'evaluator': 'framework CocoEvaluator',
              'source_sha256': {str(f.relative_to(FRAMEWORK)): sha(f)
                                for f in sorted((FRAMEWORK / 'engine').rglob('*.py'))},
              'script_sha256': sha(__file__), 'runs': []}
    depth_dir = Path(val['dataset']['depth_folder']).resolve()
    report.update(depth_health=depth_stats, depth_health_sha256=sha(args.out / 'depth_health.json'),
                  depth_directory={'path': str(depth_dir),
                                   'file_count': sum(f.is_file() for f in depth_dir.iterdir())},
                  path_maps=args.path_map,
                  depth_health_policy='Warning only for non-I/I;16 depth with zero_fraction > 0.5; does not block matched training/evaluation inputs',
                  hard_depth_gate='Exact relocated depth_folder/depth_read/depth_mode match',
                  argv=sys.argv)
    for index, (name, config, checkpoint) in enumerate(args.run):
        row = {'name': name, 'config': str(config), 'checkpoint': str(checkpoint),
               'status': 'prepared', 'metrics': None}
        report['runs'].append(row)
        try:
            cfg = load_effective(config)
            cfg.pop('__include__', None)
            training_path = training_configs.get(name, config)
            trained = load_effective(training_path)['train_dataloader']['dataset']
            train_depth = depth_contract(trained, args.path_map)
            eval_depth = depth_contract(val['dataset'])
            row.update(training_config=str(training_path), training_config_sha256=sha(training_path),
                       training_depth=train_depth, evaluation_depth=eval_depth,
                       training_provenance_level='supplied config; not immutable historical runtime proof')
            reasons = []
            row['config_sha256'] = sha(config)
            if not checkpoint.is_file():
                reasons.append(f'Checkpoint unavailable: {checkpoint}')
            if train_depth != eval_depth:
                reasons.append('Training/evaluation depth provenance mismatch: ' +
                               ', '.join(k for k in train_depth if train_depth[k] != eval_depth[k]))
            bad = [s['name'] for s in depth_stats if s['warning_8bit_majority_zero']]
            row['depth_contract_passed'] = train_depth == eval_depth
            row['depth_health_warning_images'] = bad
            row['depth_health_warning_count'] = len(bad)
            row['depth_health_note'] = (
                'Mixed-format inputs accepted because the supplied training and evaluation depth contracts match; performance impact unknown'
                if bad and train_depth == eval_depth else
                'Health statistics are descriptive; they do not establish metric calibration or historical training input identity')
            row['blocking_reasons'] = reasons
            if reasons:
                raise ValueError('; '.join(reasons))
            if cfg.get('num_classes') != 12:
                raise ValueError('Expected 12 classes')
            cfg.update(val_dataloader=copy.deepcopy(val), eval_spatial_size=args.size,
                       use_amp=False, remap_mscoco_category=False,
                       evaluator=copy.deepcopy(reference['evaluator']))
            cfg['DEIM'].update(use_dinov3_distill=False, use_co_aux=False,
                               use_internal_tta=False, use_internal_sahi=False)
            # Keep learned gates and architecture; disable only pretrained downloads.
            backbone = cfg['DEIM']['backbone']
            cfg[backbone]['pretrained'] = False
            cfg['PostProcessor'] = copy.deepcopy(reference['PostProcessor'])
            frozen = args.out / f'run_{index}.yml'
            frozen.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
            row.update(config_sha256=sha(config), aligned_config=str(frozen),
                       aligned_config_sha256=sha(frozen))
            if not checkpoint.is_file():
                raise FileNotFoundError(f'Checkpoint unavailable: {checkpoint}')
            row['checkpoint_sha256'] = sha(checkpoint)
            provenance_path = checkpoint.with_name('provenance.json')
            if provenance_path.is_file():
                row['provenance'] = json.loads(provenance_path.read_text())
                row['provenance_sha256'] = sha(provenance_path)
                if args.selection == 'model' and row['provenance'].get('selection') == 'ema.module':
                    raise ValueError('Frozen EMA export: use --selection ema.module for both runs')
            if args.evaluate:
                score(row, frozen, checkpoint, args)
        except Exception as exc:
            row.update(status='blocked', metrics=None, error=f'{type(exc).__name__}: {exc}')
        dump(args.out / 'alignment.json', report)
    render_report(report, args.out)
    if any(r['status'] == 'blocked' for r in report['runs']):
        raise SystemExit(2)


def render_report(report, out):
    """Render existing COCO statistics; never starts checkpoint inference."""
    table = ['| Model | Status | mAP | AP50 | AP75 | AP_small | AR100 | AR_s | AR_m | AR_l |',
             '|---|---|---|---|---|---|---|---|---|---|']
    for row in report['runs']:
        m = row['metrics']
        values = ['—'] * 8 if m is None else [f'{m[i]:.6f}' for i in (0, 1, 2, 3, 8, 9, 10, 11)]
        table.append('| ' + ' | '.join([row['name'], row['status'], *values]) + ' |')
    details = ['\nPreparation is not checkpoint evaluation. Depth contract mismatches block scoring; health statistics only warn.',
               '\nDepth mode/unique/zero/saturation statistics: depth_health.json and depth_health.csv. Neither format nor the warning threshold proves metric calibration.']
    details += [f'\n- {r["name"]}: {r.get("error", "Checkpoint evaluation completed on 200 images" if r["status"] == "evaluated" else "No checkpoint evaluation performed")}' for r in report['runs']]
    details += [f'\n- {r["name"]}: {r.get("depth_health_note", "")}; warnings={r.get("depth_health_warning_count", 0)}/200.' for r in report['runs']]
    details += ['\nAR100 and AR_s/m/l use IoU 0.50:0.95 and maxDets=100. AP minus AR does not isolate localization or score calibration errors.']
    if report.get('comparison_scope'):
        details += ['\n' + report['comparison_scope']]
    (Path(out) / 'alignment.md').write_text('\n'.join(table + details) + '\n')
    print('\n'.join(table))


def score(row, frozen, checkpoint, args):
    import random
    import numpy as np
    import torch
    from engine.core import YAMLConfig
    from engine.solver.det_engine import evaluate
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)
    from unittest.mock import patch
    with patch('engine.core.yaml_config.load_config', load_effective):
        cfg = YAMLConfig(str(frozen))
    model = cfg.model.float().eval()
    state = select_state(torch.load(checkpoint, map_location='cpu', weights_only=False),
                         args.selection, row.get('provenance'))
    omitted = [k for k in state if k.startswith(('dinov3_distiller.', 'co_aux_head.'))]
    state = {k: v for k, v in state.items() if k not in omitted}
    model.load_state_dict(state, strict=True)
    del state
    device = torch.device(args.device)
    model.to(device)
    evaluator = cfg.evaluator
    raw_predictions = []
    original_update = evaluator.update
    def capture_predictions(predictions):
        raw_predictions.extend(evaluator.prepare(predictions, 'bbox'))
        original_update(predictions)
    evaluator.update = capture_predictions
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
        stats, evaluator = evaluate(model, cfg.criterion.to(device), cfg.postprocessor.to(device),
                                    cfg.val_dataloader, evaluator, device)
    if len(set(evaluator.img_ids)) != 200:
        raise ValueError('Evaluator did not cover exactly 200 images')
    metrics = [float(x) for x in stats['coco_eval_bbox']]
    if not np.isfinite(metrics).all():
        raise ValueError('Nonfinite COCO metrics')
    predictions_file = args.out / (frozen.stem + '_predictions.json')
    dump(predictions_file, raw_predictions)
    torch.save(evaluator.coco_eval['bbox'].eval, args.out / (frozen.stem + '_coco_eval.pth'))
    row.update(status='evaluated', metrics=metrics,
               predictions_file=str(predictions_file), predictions_sha256=sha(predictions_file),
               evaluated_image_ids=sorted(int(i) for i in set(evaluator.img_ids)),
               omitted_training_keys=omitted, torch_version=torch.__version__,
               runtime={'device': str(device), 'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                        'cuda': torch.version.cuda, 'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
                        'cudnn_tf32': torch.backends.cudnn.allow_tf32,
                        'peak_cuda_memory_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None})


if __name__ == '__main__':
    main()
