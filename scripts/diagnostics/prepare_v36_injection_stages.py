#!/usr/bin/env python3
"""Prepare/verify V36 stage-migration arms; run is an explicit, separate action.

  python scripts/diagnostics/prepare_v36_injection_stages.py prepare
  python scripts/diagnostics/prepare_v36_injection_stages.py verify
  python scripts/diagnostics/prepare_v36_injection_stages.py run --run-id seed42_control --device cuda:0

Preparation and dry loads use CPU only, with all training heads retained.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
FRAMEWORK = ROOT / 'vendor/DEIMv2'
sys.path[:0] = [str(ROOT / 'scripts/lib'), str(ROOT / 'scripts/diagnostics'), str(FRAMEWORK)]
from audit_v27_v36_eval_alignment import load_effective, sha, select_state
from injection_stage_migration import load_migrated_state, stage_key

ARMS = {'control': [2, 3], 'p3p4': [1, 2], 'p3p4p5': [1, 2, 3]}
DEFAULT_OUT = ROOT / 'work_dir/audits/v36_injection_warmstart_20261019'


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def reset_rng(seed, cuda=False):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


def input_files(dataset, image):
    name, stem = image['file_name'], Path(image['file_name']).stem
    result = {}
    for key in ('img_folder', 'ir_folder', 'raw_ir_folder', 'depth_folder'):
        folder = Path(dataset[key]).resolve()
        candidates = [folder / name] if key == 'img_folder' else [folder / (stem + '.png'), folder / name]
        if key == 'depth_folder':
            candidates = [folder / (stem + e) for e in ('.png', '.jpg', '.jpeg')]
        actual = next((p for p in candidates if p.is_file()), None)
        if actual is None:
            raise FileNotFoundError(f'{key}: {name} in {folder}')
        result[key] = {'path': str(actual), 'sha256': sha(actual)}
    return {'id': image['id'], 'name': name, 'files': result}


def normalized_config(cfg):
    cfg = copy.deepcopy(cfg)
    for key in ('seed', 'output_dir', 'tuning'):
        cfg.pop(key, None)
    cfg['HGNetv2MultiModal'].pop('inject_stages')
    return cfg


def prepare(args):
    import torch
    import yaml
    from engine.core import YAMLConfig
    torch.set_num_threads(2)
    if torch.cuda.is_initialized():
        raise RuntimeError('Preparation must start in a fresh CPU-only process')
    args.out.mkdir(parents=True, exist_ok=False)
    configs_dir = FRAMEWORK / 'configs/deimv2' / args.out.name
    configs_dir.mkdir(parents=True, exist_ok=False)
    (args.out / 'warmstarts').mkdir()
    reference_path = ROOT / 'work_dir/audits/v36_mechanism_20261018/manifest.json'
    reference = json.loads(reference_path.read_text())
    base_path = Path(reference['configs']['baseline']['path'])
    if sha(base_path) != reference['configs']['baseline']['sha256']:
        raise ValueError('Frozen source baseline config changed')
    base = load_effective(base_path)
    base.pop('__include__', None)
    source_stages = base['HGNetv2MultiModal']['inject_stages']
    if source_stages != [2, 3]:
        raise ValueError('Expected V36 [2,3] frozen source')
    checkpoint = Path(reference['checkpoint'])
    if sha(checkpoint) != reference['checkpoint_sha256']:
        raise ValueError('Frozen source weights changed')
    repair = json.loads((ROOT / 'work_dir/audits/v36_p3_capture_fix_20261019/manifest.json').read_text())
    if sha(FRAMEWORK / 'engine/backbone/hgnetv2_mm.py') != repair['source_after_sha256']:
        raise ValueError('Expected the verified installed P3 repair')
    # Freeze a fixed screening budget, not a claim of full convergence.
    base.update(epoches=8, flat_epoch=4, no_aug_epoch=2, warmup_iter=20,
                lr_gamma=.5, use_amp=False, accum_steps=8, checkpoint_freq=1,
                sync_bn=False, print_freq=20, resume=None)
    ratio = 1e-5 / float(base['optimizer']['lr'])
    base['optimizer']['lr'] = 1e-5
    for group in base['optimizer'].get('params', []):
        if 'lr' in group:
            group['lr'] = float(group['lr']) * ratio
    base['HGNetv2MultiModal']['pretrained'] = False
    train = base['train_dataloader']
    val = base['val_dataloader']
    for name, loader in (('train', train), ('val', val)):
        loader.update(total_batch_size=8 if name == 'train' else 1, num_workers=0)
        loader.pop('persistent_workers', None)
        # The trainer reloads best_stg1 when collate.stop_epoch is reached.
        # Keep that boundary outside the 0..7 screening loop; flip stops via
        # the dataset policy below, independently, at epoch 6.
        loader['collate_fn'].update(base_size=2560, base_size_repeat=None, stop_epoch=8)
    # Explicitly stop the only stochastic image transform in the final 2 epochs.
    train['dataset']['transforms']['policy'] = {'name': 'stop_epoch', 'epoch': 6, 'ops': ['RandomHorizontalFlip']}
    annotations = {}
    inputs = {}
    for split, loader in (('train', train), ('val', val)):
        dataset = loader['dataset']
        if dataset['n_channels'] != 6 or dataset['depth_read'] != 'raw16':
            raise ValueError('Six-channel raw16 dataset required')
        ops = dataset['transforms']['ops']
        if next(op['size'] for op in ops if op['type'] == 'Resize') != [1440, 2560]:
            raise ValueError('Native resolution required for both splits')
        ann_path = Path(dataset['ann_file'])
        if sha(ann_path) != reference[f'{split}_annotations_sha256']:
            raise ValueError(f'{split} annotation hash changed')
        items = json.loads(ann_path.read_text())['images']
        annotations[split] = {'path': str(ann_path), 'sha256': sha(ann_path)}
        with ThreadPoolExecutor(max_workers=4) as pool:
            inputs[split] = list(pool.map(lambda image: input_files(dataset, image), items))
        print(f'Frozen {split} inputs: {len(inputs[split])} images / {4*len(inputs[split])} file hashes', flush=True)
    if len(inputs['train']) != 1800 or len(inputs['val']) != 200:
        raise ValueError('Expected 1800/200 split')
    if {r['name'] for r in inputs['train']} & {r['name'] for r in inputs['val']}:
        raise ValueError('Train/val image overlap')
    previous = json.loads((reference_path.parent / 'val_inputs.json').read_text())
    assert {(r['name'], k, v['sha256']) for r in inputs['val'] for k, v in r['files'].items()} == {
        (r['name'], k, v['sha256']) for r in previous for k, v in r['files'].items()}
    write_json(args.out / 'inputs.json', inputs)
    provenance = json.loads(checkpoint.with_name('provenance.json').read_text())
    load = torch.load
    def mapped_load(*pos, **kw):
        kw.update(map_location='cpu', weights_only=False, mmap=True)
        return load(*pos, **kw)
    source = select_state(mapped_load(checkpoint), 'ema.module', provenance)
    entries, templates = [], {}
    baseline_normalized = None
    for seed in (42, 43):
        for arm, stages in ARMS.items():
            run_id = f'seed{seed}_{arm}'
            cfg_dict = copy.deepcopy(base)
            cfg_dict['HGNetv2MultiModal']['inject_stages'] = stages
            cfg_dict.update(seed=seed, output_dir=str(args.out / 'training' / run_id),
                            tuning=str(args.out / 'warmstarts' / (run_id + '.pth')))
            normalized = normalized_config(cfg_dict)
            if baseline_normalized is None:
                baseline_normalized = normalized
            assert normalized == baseline_normalized, 'Arms differ beyond topology/seed/output/warmstart'
            cfg_path = configs_dir / (run_id + '.yml')
            cfg_path.write_text(yaml.safe_dump(cfg_dict, sort_keys=False, allow_unicode=True))
            reset_rng(seed)
            with patch('engine.core.yaml_config.load_config', load_effective), patch('torch.load', mapped_load):
                cfg = YAMLConfig(str(cfg_path))
                model = cfg.model.cpu()
            if 1 in stages:
                initial = {k: v.clone() for k, v in model.state_dict().items()
                           if (parsed := stage_key(k, stages)) and parsed[1] == 1}
                if seed not in templates:
                    templates[seed] = initial
                    torch.save(initial, args.out / f'stage1_template_seed{seed}.pth')
                else:
                    assert set(initial) == set(templates[seed])
                    target_state = model.state_dict()
                    with torch.no_grad():
                        for key, value in templates[seed].items():
                            assert target_state[key].shape == value.shape and target_state[key].dtype == value.dtype
                            target_state[key].copy_(value)
                    del target_state
                assert all(torch.equal(model.state_dict()[k], v) for k, v in templates[seed].items())
            migration = load_migrated_state(model, source, source_stages, stages)
            if 1 in stages:
                for k, v in model.state_dict().items():
                    if k.startswith('backbone.ir_res.0.'):
                        assert torch.count_nonzero(v) == 0, 'New stage-1 residual must remain zero initialized'
            warmstart = Path(cfg_dict['tuning'])
            torch.save({'model': model.state_dict(), 'provenance': {
                'source_checkpoint_sha256': reference['checkpoint_sha256'], 'source_selection': 'ema.module',
                'source_stages': source_stages, 'target_stages': stages, 'seed': seed,
                'scope': 'V36 warm-start topology adaptation, not pure architecture ablation'}}, warmstart)
            dry_path = args.out / (run_id + '_migration.json')
            write_json(dry_path, migration)
            entries.append({'run_id': run_id, 'seed': seed, 'arm': arm, 'stages': stages,
                            'config': str(cfg_path), 'config_sha256': sha(cfg_path),
                            'warmstart': str(warmstart), 'warmstart_sha256': sha(warmstart),
                            'migration_report': str(dry_path), 'migration_sha256': sha(dry_path),
                            'output_dir': cfg_dict['output_dir']})
            print(f'{run_id}: migrated={migration["migrated_stages"]}, fresh={migration["new_stages"]}, '
                  f'removed={migration["removed_stages"]}, unresolved={migration["unresolved_keys"]}', flush=True)
            del model, cfg, migration
            if 1 in stages:
                del initial
            gc.collect()
    assert any(not torch.equal(templates[42][k], templates[43][k]) for k in templates[42])
    assert not torch.cuda.is_initialized()
    manifest = {'status': 'prepared_not_started', 'training_started': False, 'epochs': 8,
                'seeds': [42, 43], 'arms': ARMS, 'runs': entries, 'checkpoint': str(checkpoint),
                'checkpoint_sha256': reference['checkpoint_sha256'], 'source_selection': 'ema.module',
                'source_config': str(base_path), 'source_config_sha256': sha(base_path),
                'schedule': {k: base[k] for k in ('epoches','flat_epoch','no_aug_epoch','warmup_iter','lr_gamma','lrsheduler')},
                'optimizer': base['optimizer'], 'annotations': annotations,
                'inputs_file': str(args.out / 'inputs.json'), 'inputs_sha256': sha(args.out / 'inputs.json'),
                'source_sha256': {str(p.relative_to(FRAMEWORK)): sha(p) for p in sorted((FRAMEWORK/'engine').rglob('*.py'))},
                'tool_sha256': {str(p): sha(p) for p in [Path(__file__), ROOT/'scripts/lib/injection_stage_migration.py',
                                                        ROOT/'scripts/diagnostics/audit_v27_v36_eval_alignment.py']},
                'stage1_templates': {str(s): {'path': str(args.out/f'stage1_template_seed{s}.pth'),
                                            'sha256': sha(args.out/f'stage1_template_seed{s}.pth')} for s in (42,43)},
                'effective_batch': 8, 'micro_batch': 1, 'accumulation': '8 splits within one loader batch, single process only',
                'size': [1440,2560], 'student_precision': 'FP32; teacher retains existing native FP16 autocast',
                'rng_policy': 'reset Python/NumPy/torch CPU/CUDA after model and warmstart load, before training data iteration',
                'scope': 'warm-start topology adaptation with changed initialization history; screening only',
                'criteria': {'small_tight_gain_pp': 3, 'no_candidate_must_not_increase': True,
                             'interpretation': 'Not meeting target does not prove mechanism inactive; report paired seed outcomes and spread'},
                'limitations': 'CPU construction/loading verified; GPU memory, finite gradients and native-resolution throughput not yet tested',
                'cuda_initialized_during_preparation': False, 'torch_version': torch.__version__}
    write_json(args.out/'manifest.json',manifest)
    print(f'Prepared {len(entries)} runs. No training started. {args.out / "manifest.json"}',flush=True)


def verify(out):
    manifest = json.loads((out/'manifest.json').read_text())
    for path, expected in manifest['tool_sha256'].items():
        if sha(path) != expected:
            raise ValueError(f'Tool changed: {path}')
    for relative, expected in manifest['source_sha256'].items():
        if sha(FRAMEWORK/relative) != expected:
            raise ValueError(f'Engine changed: {relative}')
    for path_key, hash_key in [('checkpoint','checkpoint_sha256'),('source_config','source_config_sha256'),('inputs_file','inputs_sha256')]:
        if sha(manifest[path_key]) != manifest[hash_key]:
            raise ValueError(f'Frozen artifact changed: {path_key}')
    for record in list(manifest['annotations'].values()) + list(manifest['stage1_templates'].values()):
        if sha(record['path']) != record['sha256']:
            raise ValueError(f'Frozen artifact changed: {record["path"]}')
    for run in manifest['runs']:
        for path_key, hash_key in [('config','config_sha256'),('warmstart','warmstart_sha256'),('migration_report','migration_sha256')]:
            if sha(run[path_key]) != run[hash_key]:
                raise ValueError(f'Frozen run artifact changed: {run["run_id"]} {path_key}')
    inputs = json.loads(Path(manifest['inputs_file']).read_text())
    files = [f for rows in inputs.values() for row in rows for f in row['files'].values()]
    with ThreadPoolExecutor(max_workers=4) as pool:
        if not all(pool.map(lambda r: sha(r['path'])==r['sha256'], files)):
            raise ValueError('Frozen input file changed')
    print(f'PASS: manifest, source, 6 configs/warmstarts, {len(files)} input hashes',flush=True)
    return manifest


def run(args):
    import runpy
    import torch
    from engine.solver._solver import BaseSolver
    manifest = verify(args.out)
    row = next((r for r in manifest['runs'] if r['run_id']==args.run_id),None)
    if row is None:
        raise ValueError('Unknown --run-id')
    if int(os.environ.get('WORLD_SIZE','1')) != 1:
        raise ValueError('This frozen budget requires single-process training')
    destination = Path(row['output_dir'])
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f'Refusing to overwrite {destination}')
    original = BaseSolver.load_tuning_state
    def exact_load_and_reset(self, path):
        if Path(path).resolve() != Path(row['warmstart']).resolve():
            raise ValueError('Unexpected warmstart override')
        state = torch.load(path,map_location='cpu',weights_only=False,mmap=True)['model']
        self.model.load_state_dict(state,strict=True)
        del state
        reset_rng(row['seed'],cuda=True)
        print('Full migrated state strict=True; training RNG reset after construction/loading',flush=True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.set_num_threads(4)
    os.chdir(FRAMEWORK)
    sys.argv=[str(FRAMEWORK/'train.py'),'-c',row['config'],'-t',row['warmstart'],
              '--seed',str(row['seed']),'-d',args.device]
    with patch.object(BaseSolver,'load_tuning_state',exact_load_and_reset), patch('engine.core.yaml_config.load_config',load_effective):
        runpy.run_path(str(FRAMEWORK/'train.py'),run_name='__main__')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','verify','run'])
    p.add_argument('--out',type=Path,default=DEFAULT_OUT)
    p.add_argument('--run-id')
    p.add_argument('--device',default='cuda:0')
    args=p.parse_args();args.out=args.out.resolve()
    os.chdir(FRAMEWORK)
    if args.action=='prepare':prepare(args)
    elif args.action=='verify':verify(args.out)
    else:run(args)


if __name__=='__main__':
    main()
