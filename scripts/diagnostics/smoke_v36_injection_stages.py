#!/usr/bin/env python3
"""Fail-closed CPU preflight / one real micro-batch GPU backward, without updates.

Run once per topology, in separate processes. Remote copies require --path-map
JSON {old_absolute_prefix: new_absolute_prefix}; original hashes remain binding.
"""
import argparse
import contextlib
import copy
import gc
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
FRAMEWORK = ROOT / 'vendor/DEIMv2'
sys.path[:0] = [str(ROOT/'scripts/lib'), str(ROOT/'scripts/diagnostics'), str(FRAMEWORK)]
from prepare_v36_injection_stages import reset_rng, sha, write_json

TEACHER_SHA = '8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035'
MANIFEST_SHA = '6d17d9d7f31699033fe40ae2ba27fcf882e44d0f5321ca8c4403fe0b50e89e98'


def relocate(value, mapping):
    if isinstance(value, dict):
        return {k: relocate(v, mapping) for k, v in value.items()}
    if isinstance(value, list):
        return [relocate(v, mapping) for v in value]
    if isinstance(value, str):
        for old, new in sorted(mapping.items(), key=lambda x: -len(x[0])):
            if value == old or value.startswith(old.rstrip('/') + '/'):
                return new.rstrip('/') + value[len(old.rstrip('/')):]
    return value


def checked(path, expected):
    if sha(path) != expected:
        raise ValueError(f'Content hash mismatch: {path}')


def nvidia(*args):
    return subprocess.check_output(['nvidia-smi', *args], text=True).strip()


def execute(args, report):
    import torch
    import yaml
    from engine.core import YAMLConfig
    from engine.backbone.dinov3_teacher import DINOv3TeacherDistiller
    torch.set_num_threads(4)
    os.chdir(FRAMEWORK)
    mapping = json.loads(args.path_map.read_text()) if args.path_map else {}
    checked(args.manifest, MANIFEST_SHA)
    report.update(path_map=mapping, script_sha256=sha(__file__),
                  manifest_path=str(args.manifest), manifest_sha256=sha(args.manifest),
                  training_started=False, optimizer_steps=0)
    original = json.loads(args.manifest.read_text())
    manifest = relocate(original, mapping)
    row = next(r for r in manifest['runs'] if r['run_id'] == args.run_id)
    report['run_id'] = args.run_id
    for relative, expected in manifest['source_sha256'].items():
        checked(FRAMEWORK/relative, expected)
    for path, expected in original['tool_sha256'].items():
        checked(relocate(path, mapping), expected)
    for key, hashkey in [('config', 'config_sha256'), ('warmstart', 'warmstart_sha256'),
                         ('migration_report', 'migration_sha256')]:
        checked(row[key], row[hashkey])
    checked(manifest['inputs_file'], manifest['inputs_sha256'])
    for record in manifest['annotations'].values():
        checked(record['path'], record['sha256'])
    inputs = relocate(json.loads(Path(manifest['inputs_file']).read_text()), mapping)
    count = 0
    for records in inputs.values():
        for record in records:
            for file in record['files'].values():
                checked(file['path'], file['sha256'])
                count += 1
    report['verified_input_files'] = count
    cfg_dict = relocate(yaml.safe_load(Path(row['config']).read_text()), mapping)
    report['effective_config'] = cfg_dict
    assert cfg_dict['HGNetv2MultiModal']['inject_stages'] == row['stages']
    assert cfg_dict['DEIM']['use_dinov3_distill'] and cfg_dict['DEIM']['use_co_aux']
    teacher = (FRAMEWORK/cfg_dict['DEIM']['dinov3_teacher_path']).resolve()
    checked(teacher, TEACHER_SHA)
    report['teacher'] = {'path': str(teacher), 'sha256': TEACHER_SHA}
    # Enforce the configured path, so a fallback cannot make this gate pass.
    def required_teacher(self):
        if Path(self.model_path).resolve() != teacher:
            raise ValueError('Unexpected teacher path')
        return str(teacher)
    reset_rng(row['seed'])
    captured = io.StringIO()
    with patch.object(DINOv3TeacherDistiller, '_find_checkpoint', required_teacher), \
         patch('engine.core.yaml_config.load_config', lambda *a, **kw: copy.deepcopy(cfg_dict)), \
         contextlib.redirect_stdout(captured):
        cfg = YAMLConfig(row['config'])
        model = cfg.model.cpu()
    log = captured.getvalue()
    print(log, end='', flush=True)
    report['construction_log'] = log
    if 'Weights loaded successfully (strict=True)' not in log or 'WARNING' in log:
        raise RuntimeError('Teacher strict-load gate failed')
    official = torch.load(teacher, map_location='cpu', weights_only=False, mmap=True)
    state = torch.load(row['warmstart'], map_location='cpu', weights_only=False, mmap=True)['model']
    model.load_state_dict(state, strict=True)
    teacher_state = model.dinov3_distiller.teacher.state_dict()
    deltas = []
    for key, tensor in official.items():
        delta = (tensor.float() - teacher_state[key].float()).abs()
        deltas.append({'key': key, 'equal': torch.equal(tensor, teacher_state[key]),
                       'max_abs_diff': float(delta.max()), 'mean_abs_diff': float(delta.mean())})
    report['teacher']['official_vs_warmstart'] = deltas
    report['teacher']['policy'] = 'Official constructor strict-load checked; retain frozen V36 EMA teacher after full-state strict-load'
    del state, official, teacher_state
    gc.collect()
    report['G2'] = 'passed_official_constructor_and_full_warmstart_strict_load'
    if args.action == 'preflight':
        report.update(status='cpu_preflight_passed', G1='not_run', G3='not_bound')
        return
    if not args.device.startswith('cuda:') or not torch.cuda.is_available():
        raise RuntimeError('GPU backward requires CUDA')
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    uuid = str(torch.cuda.get_device_properties(device).uuid)
    # Refuse to add work to an occupied GPU. Recheck immediately before .to().
    processes = nvidia('--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits')
    report['gpu_process_snapshot'] = processes
    for line in processes.splitlines():
        parts = [v.strip() for v in line.split(',')]
        if len(parts) == 2 and parts[0] == uuid and int(parts[1]) != os.getpid():
            raise RuntimeError(f'Target GPU occupied: {line}')
    report['environment'] = dict(host=platform.node(), python=sys.version,
        torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
        gpu=torch.cuda.get_device_name(device), gpu_uuid=uuid,
        driver=nvidia('--query-gpu=uuid,driver_version', '--format=csv,noheader'),
        nvidia_smi_L=nvidia('-L'))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    reset_rng(row['seed'], cuda=True)
    # Use the unchanged train dataset / collator and the first sampled item of
    # its seeded batch. One micro-batch has B=1 in the frozen accumulation plan.
    loader = cfg.train_dataloader
    loader.set_epoch(0)
    index = next(iter(loader.batch_sampler))[0]
    item = loader.dataset[index]
    samples, targets = loader.collate_fn([item])
    image_id = int(targets[0]['image_id'].reshape(-1)[0])
    report['input'] = dict(dataset_index=int(index), image_id=image_id,
                           shape=list(samples.shape), epoch=0)
    assert list(samples.shape) == [1, 6, 1440, 2560]
    model = model.to(device).train()
    criterion = cfg.criterion.to(device).train()
    if cfg_dict.get('freeze_bn', False):
        for module in model.modules():
            if isinstance(module, (torch.nn.BatchNorm2d, torch.nn.SyncBatchNorm)):
                module.eval()
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
    targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
    torch.cuda.reset_peak_memory_stats(device)
    model.zero_grad(set_to_none=True)
    outputs = model(samples.to(device), targets=targets)
    for key in ('pred_boxes', 'pred_logits'):
        if not torch.isfinite(outputs[key]).all():
            raise RuntimeError(f'Nonfinite {key}')
    if 'loss_distill' not in outputs:
        raise RuntimeError('Distillation branch did not run')
    losses = criterion(outputs, targets, epoch=0, step=0, global_step=0, epoch_step=len(loader))
    if not {'loss_distill', 'loss_co_cls', 'loss_co_reg'} <= losses.keys():
        raise RuntimeError('Expected teacher/Co-ATSS losses are missing')
    report['losses'] = {key: float(value.detach()) for key, value in losses.items()}
    loss = sum(losses.values())
    if not torch.isfinite(loss):
        raise RuntimeError('Nonfinite loss')
    (loss / 8).backward()
    gradients = [name for name, p in model.named_parameters() if p.grad is not None]
    bad = [name for name, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    missing = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
    report['gradients'] = dict(present=gradients, missing=missing, nonfinite=bad)
    if bad or not gradients or not any(n.startswith('co_aux_head.') for n in gradients):
        raise RuntimeError('Finite-gradient / auxiliary-head gate failed')
    torch.cuda.synchronize(device)
    report.update(status='gpu_smoke_passed', G1='passed', G3='observed_environment_only',
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(device))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['preflight', 'smoke'])
    parser.add_argument('--manifest', type=Path, default=ROOT/'work_dir/audits/v36_injection_warmstart_20261019/manifest.json')
    parser.add_argument('--run-id', choices=['seed42_p3p4', 'seed42_p3p4p5'], required=True)
    parser.add_argument('--path-map', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    args.manifest = args.manifest.resolve()
    args.report = args.report.resolve()
    if args.report.exists():
        raise FileExistsError(args.report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = {'status': 'failed', 'G1': 'not_run', 'G2': 'not_run', 'G3': 'not_bound'}
    try:
        execute(args, report)
    except Exception:
        report['error'] = traceback.format_exc()
        raise
    finally:
        write_json(args.report, report)


if __name__ == '__main__':
    main()
