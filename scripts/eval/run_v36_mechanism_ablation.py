#!/usr/bin/env python3
"""Reproducible V36 ablation using the framework's dataset and COCO evaluator.

Run from the repository root. prepare freezes five native train.py configs;
eval compares frozen weights (no training); train runs matched warm starts.
No test-set images or submission artifacts are produced by this tool.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
FRAMEWORK = ROOT / 'vendor/DEIMv2'
sys.path[:0] = [str(ROOT / 'scripts/lib'), str(FRAMEWORK)]
os.chdir(FRAMEWORK)

import numpy as np
import torch
import yaml
from engine.core import YAMLConfig
from engine.core.yaml_utils import load_config

VARIANTS = ('baseline', 'routing', 'fallback', 'aux_only', 'combined')
REMOTE_ROOT = '/home/ailab/ssd/bigdata/hejianzhong/AIC_V29'


def digest(path):
    with Path(path).open('rb') as stream:
        hasher = hashlib.sha256()
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            hasher.update(block)
        return hasher.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def resolve(path):
    return (FRAMEWORK / path).resolve()


def remap(value):
    if isinstance(value, dict):
        return {k: remap(v) for k, v in value.items()}
    if isinstance(value, list):
        return [remap(v) for v in value]
    if isinstance(value, str):
        res = value.replace(REMOTE_ROOT, str(ROOT))
        res = res.replace('/home/constant/CodeBuddy/Dev/AIC', str(ROOT))
        return res
    return value


def modality_files(dataset, name):
    files = {}
    for key in ('img_folder', 'ir_folder', 'raw_ir_folder', 'depth_folder'):
        folder = resolve(dataset[key])
        stem = Path(name).stem
        candidates = ([folder / name] if key == 'img_folder' else
                      [folder / (stem + '.png'), folder / name])
        if key == 'depth_folder':
            candidates = [folder / (stem + ext) for ext in ('.png', '.jpg', '.jpeg')]
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            raise FileNotFoundError(f'{key}: {name} under {folder}')
        files[key] = {'path': str(path), 'sha256': digest(path)}
    return files


def prepare(args):
    if args.epochs < 1 or args.batch_size < 1 or args.limit < 0:
        raise ValueError('Epochs/batch must be positive and limit nonnegative')
    if any(size <= 0 or size % 32 for size in args.train_size):
        raise ValueError('Training height/width must be positive multiples of 32 for the neck')
    if (args.out / 'manifest.json').exists():
        raise FileExistsError('Manifest already frozen; use a new --out for a new run')
    cfg = remap(load_config(str(args.config)))
    cfg.pop('__include__', None)
    train = cfg['train_dataloader']['dataset']
    val = cfg['val_dataloader']['dataset']
    # Explicit relocation only: never substitute raw IR for trained LBG inputs.
    for ds in (train, val):
        if getattr(args, 'aligned_ir_dir', None):
            ds['ir_folder'] = str(args.aligned_ir_dir)
        ds['raw_ir_folder'] = str(args.lbg_dir)
        if ds.get('n_channels') != 6 or ds.get('depth_read') != 'raw16':
            raise ValueError('Expected six-channel raw16 V36 dataset')
    train_ann = resolve(train['ann_file'])
    val_ann = resolve(val['ann_file'])
    train_images = json.loads(train_ann.read_text())['images']
    val_images = json.loads(val_ann.read_text())['images']
    overlap = {Path(x['file_name']).stem for x in train_images} & {Path(x['file_name']).stem for x in val_images}
    if overlap or len(train_images) != 1800 or len(val_images) != 200:
        raise ValueError(f'Invalid 1800/200 split: overlap={len(overlap)}')
    # Validate every training input exists; hash the full evaluation input set.
    for item in train_images:
        for key in ('img_folder', 'ir_folder', 'raw_ir_folder', 'depth_folder'):
            folder = resolve(train[key])
            if not any((folder / (Path(item['file_name']).stem + ext)).is_file()
                       for ext in ('.png', '.jpg', '.jpeg')):
                raise FileNotFoundError(f'Train {key}: {item["file_name"]}')
    inputs = [{'id': x['id'], 'name': x['file_name'],
               'files': modality_files(val, x['file_name'])} for x in val_images]
    write_json(args.out / 'val_inputs.json', inputs)
    no_aug_epochs = min(2, args.epochs // 3)
    cfg.update(epoches=args.epochs, output_dir='', warmup_iter=20,
               flat_epoch=max(0, args.epochs - no_aug_epochs - 1), no_aug_epoch=no_aug_epochs,
               checkpoint_freq=1, use_amp=args.amp)
    # det_engine splits each loader batch into micro-batches; it does NOT
    # accumulate across loader iterations. Effective batch is total_batch_size.
    effective_batch = args.batch_size
    # Scale explicit parameter-group LRs as well as the optimizer default.
    ratio = args.lr / float(cfg['optimizer']['lr'])
    for group in cfg['optimizer'].get('params', []):
        if 'lr' in group:
            group['lr'] = float(group['lr']) * ratio
    cfg['optimizer']['lr'] = args.lr
    for op in train['transforms']['ops']:
        if op.get('type') == 'Resize':
            op['size'] = list(args.train_size)
    if isinstance(cfg['train_dataloader'].get('collate_fn'), dict):
        cfg['train_dataloader']['collate_fn']['base_size'] = args.train_size[1]
    for name in ('train_dataloader', 'val_dataloader'):
        cfg[name]['total_batch_size'] = args.batch_size
        cfg[name]['num_workers'] = args.workers
        if args.workers == 0:
            cfg[name].pop('persistent_workers', None)
        if isinstance(cfg[name].get('collate_fn'), dict):
            cfg[name]['collate_fn']['stop_epoch'] = args.epochs
    configs = {}
    config_dir = FRAMEWORK / 'configs/deimv2' / args.out.name
    config_dir.mkdir(parents=True, exist_ok=True)
    for variant in VARIANTS:
        item = copy.deepcopy(cfg)
        routing = variant in ('routing', 'combined')
        fallback = variant in ('fallback', 'combined')
        aux = variant in ('aux_only', 'combined')
        item['HGNetv2MultiModal']['disable_fallback'] = not fallback
        item['DFINETransformer'].update(
            use_pure_topk=False, use_anisotropic_routing=routing,
            p3_quota_ratio=0.30, p4_quota_ratio=0.30,
            use_offset_clamp=False, use_spatial_attn_bias=False,
            use_layer_ensemble=False)
        item['DEIMCriterion'].update(
            use_discounted_aux_loss=aux,
            aux_loss_discount_steps=[0.4, 0.5, 0.6, 0.8, 0.9],
            aux_loss_discount_losses=['tadl'])
        item['output_dir'] = str(args.out / 'training' / variant)
        path = config_dir / (variant + '.yml')
        path.write_text(yaml.safe_dump(item, allow_unicode=True, sort_keys=False))
        configs[variant] = {'path': str(path), 'sha256': digest(path)}
    sources = ['engine/backbone/hgnetv2_mm.py', 'engine/deim/dfine_decoder.py',
               'engine/deim/deim_criterion.py', 'engine/data/dataset/coco_dataset_mm.py',
               'engine/data/transforms/_transforms.py', 'engine/deim/postprocessor.py',
               'engine/deim/deim.py', 'engine/deim/hybrid_encoder.py',
               'engine/solver/det_engine.py', 'engine/data/dataset/coco_eval.py',
               'engine/backbone/dinov3_teacher.py']
    manifest = {
        'checkpoint': str(args.checkpoint), 'checkpoint_sha256': digest(args.checkpoint),
        'source_config': str(args.config), 'source_config_sha256': digest(args.config),
        'configs': configs, 'seed': args.seed, 'epochs': args.epochs,
        'train_annotations_sha256': digest(train_ann),
        'val_annotations_sha256': digest(val_ann), 'val_inputs_sha256': digest(args.out / 'val_inputs.json'),
        'source_sha256': {f: digest(FRAMEWORK / f) for f in sources},
        'python': sys.version, 'torch': torch.__version__, 'cuda': torch.version.cuda,
        'platform': platform.platform(),
        'precision': 'native hardware-selected AMP' if args.amp else 'student fp32; frozen teacher retains its native autocast',
        'effective_training_batch': effective_batch,
        'training_size': list(args.train_size), 'validation_size': cfg['eval_spatial_size'],
        'experiment_scope': ('native-resolution fine tuning' if list(args.train_size) == cfg['eval_spatial_size']
                             else 'reduced-resolution pilot; not proof of full-resolution adaptation'),
        'accumulation_semantics': 'within-loader-batch micro splitting, no cross-iteration accumulation',
        'baseline_routing': 'historical P3 35% + global 65% (not pure Top-K)',
        'warm_start': 'same selected EMA/model weights; fresh optimizer and EMA per arm',
        'aux_scope': 'tadl classification only; geometry unchanged; pre uses first weight',
        'provenance': json.loads(args.checkpoint.with_name('provenance.json').read_text())
                      if args.checkpoint.with_name('provenance.json').exists() else None,
    }
    write_json(args.out / 'manifest.json', manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


def verify(args):
    manifest = json.loads((args.out / 'manifest.json').read_text())
    if digest(manifest['checkpoint']) != manifest['checkpoint_sha256']:
        raise ValueError('Frozen checkpoint changed')
    for info in manifest['configs'].values():
        if digest(info['path']) != info['sha256']:
            raise ValueError(f'Frozen config changed: {info["path"]}')
    for name, expected in manifest['source_sha256'].items():
        if digest(FRAMEWORK / name) != expected:
            raise ValueError(f'Frozen source changed: {name}; prepare a new run')
    baseline = yaml.safe_load(Path(manifest['configs']['baseline']['path']).read_text())
    for kind in ('train', 'val'):
        ann = resolve(baseline[f'{kind}_dataloader']['dataset']['ann_file'])
        if digest(ann) != manifest[f'{kind}_annotations_sha256']:
            raise ValueError(f'Frozen {kind} annotations changed')
    if digest(args.out / 'val_inputs.json') != manifest['val_inputs_sha256']:
        raise ValueError('Frozen input manifest changed')
    for row in json.loads((args.out / 'val_inputs.json').read_text()):
        for info in row['files'].values():
            if digest(info['path']) != info['sha256']:
                raise ValueError(f'Frozen input changed: {info["path"]}')
    return manifest


def evaluate(args, manifest):
    from engine.solver.det_engine import evaluate as framework_evaluate
    random.seed(manifest['seed']); np.random.seed(manifest['seed']); torch.manual_seed(manifest['seed'])
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    base = YAMLConfig(manifest['configs']['baseline']['path'])
    base.yaml_cfg['DEIM'].update(use_dinov3_distill=False, use_co_aux=False)
    base.yaml_cfg['HGNetv2MultiModal']['pretrained'] = False
    base.yaml_cfg['val_dataloader']['total_batch_size'] = 1
    base.yaml_cfg['val_dataloader']['num_workers'] = args.workers
    base.yaml_cfg['val_dataloader'].pop('persistent_workers', None)
    model = base.model.eval()
    ck = torch.load(manifest['checkpoint'], map_location='cpu', weights_only=False)
    state = ck['model']
    omitted = [k for k in state if k.startswith(('dinov3_distiller.', 'co_aux_head.'))]
    state = {k: v for k, v in state.items() if k not in omitted}
    # Shape or non-training-head mismatches abort; never score random parameters.
    model.load_state_dict(state, strict=True)
    del ck, state
    model.to(device)
    criterion, postprocessor = base.criterion.to(device), base.postprocessor.to(device)
    loader, evaluator = base.val_dataloader, base.evaluator
    if args.limit:
        # Diagnostic subset only. Restrict evaluator as well to avoid missing-image FNs.
        from torch.utils.data import Subset
        loader = torch.utils.data.DataLoader(Subset(loader.dataset, range(min(args.limit, len(loader.dataset)))),
                                            batch_size=1, collate_fn=loader.collate_fn, num_workers=args.workers)
    scorecard = {'kind': 'checkpoint_evaluation_without_training',
                 'diagnostic_subset': bool(args.limit), 'checkpoint_sha256': manifest['checkpoint_sha256'],
                 'omitted_training_head_keys': omitted, 'results': {},
                 'evaluator_script_sha256': digest(__file__),
                 'runtime': {'torch': torch.__version__, 'cuda': torch.version.cuda,
                             'gpu': torch.cuda.get_device_name(device),
                             'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
                             'cudnn_tf32': torch.backends.cudnn.allow_tf32,
                             'cudnn_enabled': torch.backends.cudnn.enabled}}
    original_update = evaluator.update
    raw_predictions = []
    def capture_predictions(predictions):
        raw_predictions.extend(evaluator.prepare(predictions, 'bbox'))
        original_update(predictions)
    evaluator.update = capture_predictions
    for variant in args.variants:
        raw_predictions.clear()
        variant_cfg = yaml.safe_load(Path(manifest['configs'][variant]['path']).read_text())
        dog_modules = [m for m in model.modules() if m.__class__.__name__ == 'DepthOcclusionGating']
        if len(dog_modules) != 2:
            raise ValueError(f'Expected two DOG modules, found {len(dog_modules)}')
        for dog in dog_modules:
            dog.disable_fallback = variant_cfg['HGNetv2MultiModal']['disable_fallback']
        model.decoder.use_pure_topk = False
        model.decoder.use_anisotropic_routing = variant_cfg['DFINETransformer']['use_anisotropic_routing']
        model.decoder.p3_quota_ratio = model.decoder.p4_quota_ratio = 0.30
        criterion.use_discounted_aux_loss = variant_cfg['DEIMCriterion']['use_discounted_aux_loss']
        print(f'\nVARIANT {variant}: fallback={not dog_modules[0].disable_fallback}, '
              f'routing={model.decoder.use_anisotropic_routing}', flush=True)
        started = time.monotonic()
        stats, evaluator = framework_evaluate(model, criterion, postprocessor, loader, evaluator, device)
        write_json(args.out / f'{variant}_predictions.json', raw_predictions)
        result = {'metrics': stats['coco_eval_bbox'], 'seconds': time.monotonic() - started,
                  'images': len(evaluator.img_ids), 'fallback': not dog_modules[0].disable_fallback,
                  'routing': model.decoder.use_anisotropic_routing}
        result['predictions_sha256'] = digest(args.out / f'{variant}_predictions.json')
        # Raw predictions and eval object support later paired re-evaluation.
        torch.save(evaluator.coco_eval['bbox'].eval, args.out / f'{variant}_coco_eval.pth')
        scorecard['results'][variant] = result
        write_json(args.out / ('smoke_scores.json' if args.limit else 'zeroshot_scores.json'), scorecard)
    print(json.dumps(scorecard, indent=2))


def train(args, manifest):
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT / 'scripts/lib'), str(FRAMEWORK), env.get('PYTHONPATH', '')])
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    for variant in args.variants:
        destination = args.out / 'training' / variant
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError(f'Refusing to overwrite training outputs: {destination}')
        command = [sys.executable, '-u', str(Path(__file__).resolve()), 'fit',
                   '--out', str(args.out), '--variants', variant]
        if getattr(args, 'device', None):
            command.extend(['--device', str(args.device)])
        log = ROOT / 'logs' / f'{args.out.name}_{variant}_train.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        with log.open('w') as stream:
            result = subprocess.run(command, cwd=FRAMEWORK, env=env, stdout=stream, stderr=subprocess.STDOUT)
        write_json(args.out / f'{variant}_train_exit.json',
                   {'command': command, 'exit_code': result.returncode, 'seconds': time.monotonic() - started})
        if result.returncode:
            raise RuntimeError(f'{variant} failed, see {log}')


def fit(args, manifest):
    """Execute native train.py while mapping known local ZIP checkpoints in RAM."""
    import runpy
    if len(args.variants) != 1:
        raise ValueError('fit requires one variant; use train for a matrix')
    variant = args.variants[0]
    load = torch.load
    def mapped_load(path, *positional, **kwargs):
        if isinstance(path, (str, Path)) and Path(path).is_file():
            kwargs.setdefault('mmap', True)
        return load(path, *positional, **kwargs)
    torch.load = mapped_load
    sys.argv = [str(FRAMEWORK / 'train.py'), '-c', manifest['configs'][variant]['path'],
                '-t', manifest['checkpoint'], '--seed', str(manifest['seed'])]
    if getattr(args, 'device', None):
        sys.argv.extend(['-d', str(args.device)])
    if yaml.safe_load(Path(manifest['configs'][variant]['path']).read_text()).get('use_amp'):
        sys.argv.append('--use-amp')
    runpy.run_path(str(FRAMEWORK / 'train.py'), run_name='__main__')


def step(args, manifest):
    """Real full-resolution forward/backward/AdamW checks; no checkpoint scores."""
    import gc
    results = {}
    for variant in args.variants:
        random.seed(manifest['seed']); np.random.seed(manifest['seed']); torch.manual_seed(manifest['seed'])
        cfg = YAMLConfig(manifest['configs'][variant]['path'])
        cfg.yaml_cfg['HGNetv2MultiModal']['pretrained'] = False
        # Map large teacher/checkpoint tensors rather than duplicating them in
        # host RAM during this diagnostic. Model values and dtypes are unchanged.
        load = torch.load
        def mapped_load(path, *positional, **kwargs):
            kwargs.setdefault('mmap', True)
            return load(path, *positional, **kwargs)
        torch.load = mapped_load
        try:
            model = cfg.model
        finally:
            torch.load = load
        checkpoint = torch.load(manifest['checkpoint'], map_location='cpu', weights_only=False, mmap=True)
        model.load_state_dict(checkpoint['model'], strict=True)
        print('Frozen checkpoint restored with strict=True, including teacher and auxiliary heads', flush=True)
        del checkpoint
        device = torch.device(args.device)
        model.to(device).train()
        criterion = cfg.criterion.to(device)
        optimizer = cfg.optimizer
        scaler = cfg.scaler
        ema = cfg.ema
        if ema is not None:
            ema.module.to(device)
        loader = cfg.train_dataloader
        loader.set_epoch(0)
        torch.cuda.reset_peak_memory_stats(device)
        losses = []
        started = time.monotonic()
        for index, (samples, targets) in enumerate(loader):
            optimizer.zero_grad(set_to_none=True)
            count = len(targets)
            total_loss = 0.0
            # Match native within-batch splitting at the original batch size.
            for micro in range(count):
                image = samples[micro:micro+1].to(device)
                target = [{k: v.to(device) for k, v in targets[micro].items()}]
                dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=cfg.yaml_cfg.get('use_amp', False)):
                    outputs = model(image, targets=target)
                loss_dict = criterion(outputs, target, epoch=0)
                loss = sum(loss_dict.values()) / count
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'{variant}: nonfinite training loss')
                if scaler is None:
                    loss.backward()
                else:
                    scaler.scale(loss).backward()
                total_loss += loss.item()
                del outputs, loss_dict, loss, image, target
            if scaler is not None:
                scaler.unscale_(optimizer)
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError(f'{variant}: nonfinite training gradients')
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_max_norm)
            if scaler is None:
                optimizer.step()
            else:
                scaler.step(optimizer)
                scaler.update()
            if ema is not None:
                ema.update(model)
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise FloatingPointError(f'{variant}: nonfinite parameters after AdamW')
            losses.append({'loss': total_loss, 'gradient_norm': norm.item()})
            print(f'{variant} real step {index}: {losses[-1]}', flush=True)
            if index >= 1:
                break
        results[variant] = {'kind': 'two_step_training_smoke_only', 'steps': losses,
                            'peak_cuda_bytes': torch.cuda.max_memory_allocated(device),
                            'seconds': time.monotonic() - started}
        write_json(args.out / 'training_smoke.json', results)
        del model, optimizer, criterion, loader, cfg, samples, targets, scaler, ema
        gc.collect(); torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare', 'eval', 'step', 'train', 'fit'])
    parser.add_argument('--out', type=Path, default=ROOT / 'work_dir/audits/v36_mechanism_20261018')
    parser.add_argument('--config', type=Path, default=FRAMEWORK / 'configs/deimv2/sovereign_v36_p2_direct_injection_50ep_v100.yml')
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'outputs/checkpoints/v36_mechanism_20261018/v36_frozen.pth')
    parser.add_argument('--aligned-ir-dir', type=Path, default=ROOT / 'work_dir/server_generated_ir/infrared_aligned',
                        help='directory containing aligned IR images')
    parser.add_argument('--lbg-dir', type=Path, default=ROOT / 'work_dir/server_generated_ir/train_lbg_ir',
                        help='directory containing LBG images')
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--amp', action='store_true', help='native AMP: BF16 on Ampere/newer, FP16 on older GPUs')
    parser.add_argument('--train-size', nargs=2, type=int, default=[1440, 2560], metavar=('HEIGHT', 'WIDTH'))
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--limit', type=int, default=0, help='diagnostic image count; 0 evaluates all 200')
    parser.add_argument('--variants', nargs='+', choices=VARIANTS,
                        default=list(VARIANTS))
    args = parser.parse_args()
    if args.phase in ('train', 'fit', 'step') and (
            str(ROOT).startswith('/home/constant/') or
            socket.gethostname() == 'constant-B850M-EAGLE-WIFI6E-ICE'):
        parser.error('Local training is disabled by user instruction; run this phase on ailab4/ailab40')
    args.out = args.out.resolve(); args.config = args.config.resolve()
    args.checkpoint = args.checkpoint.resolve(); args.lbg_dir = args.lbg_dir.resolve()
    if getattr(args, 'aligned_ir_dir', None):
        args.aligned_ir_dir = args.aligned_ir_dir.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.phase == 'prepare':
        prepare(args)
    elif args.phase == 'eval':
        evaluate(args, verify(args))
    elif args.phase == 'step':
        step(args, verify(args))
    elif args.phase == 'fit':
        fit(args, verify(args))
    else:
        if '--variants' not in sys.argv:
            args.variants = list(VARIANTS)
        train(args, verify(args))


if __name__ == '__main__':
    main()
