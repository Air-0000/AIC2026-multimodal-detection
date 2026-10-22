#!/usr/bin/env python3
"""Exact replay of a crashed clean-lifecycle epoch to locate where non-finite values first appear.

Rebuilds the solver the way train.py does, resumes strictly from a clean-lifecycle checkpoint (model,
EMA, optimizer and Python/NumPy/torch/CUDA RNG), and runs the real train_one_epoch for the next epoch
with the original print frequency, so its loss lines can be compared with the original log. The
optimizer steps happen in memory only (they are needed to reach the crash-time weights); nothing is
saved and the run stops after --stop-after iterations or at the first non-finite forward.

Hooks: every module under encoder, plus the top of backbone, decoder, co_aux_head and the distiller
projection head, records a non-finite output together with whether its inputs and own parameters were
finite. Inputs of the first few originating modules (finite inputs, non-finite output) are saved.
Writes only under --out (must not exist): replay.log, events.json, origin_*.pt, encoder_absmax.json.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time


def tensors(x):
    import torch
    if isinstance(x, torch.Tensor):
        yield x
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from tensors(v)
    elif isinstance(x, dict):
        for v in x.values():
            yield from tensors(v)


class StopAfter:
    """Loader view that keeps len() (LR schedule, metas) but stops early; tracks the iteration index."""

    def __init__(self, loader, n, state):
        self.loader, self.n, self.state = loader, n, state

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for i, batch in enumerate(self.loader):
            if i >= self.n:
                return
            self.state['iter'], self.state['forward'] = i, 0
            yield batch


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--framework', type=Path, required=True, help='frozen vendor/DEIMv2 of the crashed run (read only)')
    p.add_argument('--lib', type=Path, required=True, help='scripts/lib of the crashed run')
    p.add_argument('--config', required=True, help='config path relative to --framework')
    p.add_argument('--resume', type=Path, required=True, help='last.pth written before the crashed epoch')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--stop-after', type=int, default=175)
    p.add_argument('--max-saves', type=int, default=6)
    a = p.parse_args()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    framework, resume = a.framework.resolve(), a.resume.resolve()
    sys.path[:0] = [str(framework), str(a.lib.resolve())]
    os.chdir(framework)
    import torch
    from engine.core import YAMLConfig
    from engine.misc import dist_utils
    from engine.optim.lr_scheduler import FlatCosineLRScheduler
    from engine.solver import TASKS
    from engine.solver.det_engine import train_one_epoch

    # Same construction as train.py with `-r resume -u tuning=null`, outputs redirected to --out.
    cfg = YAMLConfig(a.config, resume=str(resume), tuning=None, output_dir=str(out/'solver_outputs'),
                     summary_dir=str(out/'tensorboard'))
    if cfg.seed is None:
        cfg.seed = 0
    cfg.yaml_cfg['seed'] = cfg.seed
    dist_utils.setup_distributed(0, 'builtin', seed=cfg.seed, strict=cfg.clean_baseline)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # train.py's import-time settings: no cuDNN autotuning, and no cuDNN at all on Volta (SM 7.0).
    torch.backends.cudnn.benchmark = False
    if torch.cuda.get_device_capability() == (7, 0):
        torch.backends.cudnn.enabled = False
    for bb in ['HGNetv2', 'HGNetv2MultiModal', 'HGNetv2ViTDualStream']:
        if bb in cfg.yaml_cfg:
            cfg.yaml_cfg[bb]['pretrained'] = False
    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    solver.train()                                  # builds everything, then strict resume incl. RNG
    epoch = solver.last_epoch + 1
    solver.self_lr_scheduler = True
    solver.lr_scheduler = FlatCosineLRScheduler(solver.optimizer, cfg.lr_gamma, len(solver.train_dataloader),
                                                cfg.epoches, cfg.warmup_iter, cfg.flat_epoch, cfg.no_aug_epoch)
    solver.train_dataloader.set_epoch(epoch)

    model = dist_utils.de_parallel(solver.model)
    state = dict(iter=-1, forward=0, saves=0)
    events, absmax = [], []

    def finite(t):
        return bool(torch.isfinite(t).all())

    def hook(name):
        def fn(mod, inp, outp):
            bad = [t for t in tensors(outp) if t.is_floating_point() and not finite(t)]
            if not bad:
                return
            ins = [t for t in tensors(inp) if t.is_floating_point()]
            inputs_finite = all(finite(t) for t in ins)
            params_finite = all(finite(q) for q in mod.parameters(recurse=False))
            rec = dict(iter=state['iter'], forward=state['forward'], module=name, type=type(mod).__name__,
                       inputs_finite=inputs_finite, params_finite=params_finite,
                       nan=int(sum(int(torch.isnan(t).sum()) for t in bad)), inf=int(sum(int(torch.isinf(t).sum()) for t in bad)),
                       shapes=[list(t.shape) for t in bad][:4],
                       input_absmax=[float(t.detach().abs().max()) for t in ins if t.numel()][:6] if inputs_finite else None)
            events.append(rec)
            if inputs_finite and state['saves'] < a.max_saves:
                state['saves'] += 1
                torch.save(dict(record=rec, inputs=[t.detach().cpu() for t in ins],
                                outputs=[t.detach().cpu() for t in tensors(outp) if t.is_floating_point()],
                                params={k: v.detach().cpu() for k, v in mod.state_dict().items()}),
                           out/f"origin_{state['saves']}_{name.replace('.', '_')}.pt")
        return fn

    def count_forward(mod, inp):
        state['forward'] += 1

    def encoder_stats(mod, inp, outp):
        absmax.append(dict(iter=state['iter'], forward=state['forward'],
                           absmax=[float(t.detach().abs().max()) for t in tensors(outp) if t.is_floating_point()][:4]))

    model.register_forward_pre_hook(count_forward)
    hooked = 0
    for name, mod in model.named_modules():
        top = name.split('.')[0]
        if name.startswith('encoder') or name in ('backbone', 'decoder', 'co_aux_head', 'dinov3_distiller.proj_head'):
            mod.register_forward_hook(hook(name))
            hooked += 1
        if top in ('co_aux_head',) and name.count('.') == 1:
            mod.register_forward_hook(hook(name))
            hooked += 1
    model.encoder.register_forward_hook(encoder_stats)
    print(f'[replay] resumed last_epoch={solver.last_epoch}; replaying epoch {epoch} for <= {a.stop_after} iterations; '
          f'{hooked} hooks', flush=True)

    t0, error = time.time(), None
    try:
        train_one_epoch(True, solver.lr_scheduler, solver.model, solver.criterion,
                        StopAfter(solver.train_dataloader, a.stop_after, state), solver.optimizer, solver.device, epoch,
                        max_norm=cfg.clip_max_norm, print_freq=cfg.print_freq, ema=solver.ema, scaler=solver.scaler,
                        writer=None, freeze_bn=cfg.freeze_bn, accum_steps=cfg.accum_steps, stop_on_nonfinite=True)
    except FloatingPointError as e:
        error = str(e)
    summary = dict(epoch=epoch, stopped_at_iter=state['iter'], error=error, seconds=round(time.time()-t0, 1),
                   resume=str(resume), framework=str(framework), config=a.config, events=len(events),
                   first_origin=next((e for e in events if e['inputs_finite']), None))
    (out/'events.json').write_text(json.dumps(dict(summary=summary, events=events), indent=1))
    (out/'encoder_absmax.json').write_text(json.dumps(absmax))
    print('[replay] summary', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
