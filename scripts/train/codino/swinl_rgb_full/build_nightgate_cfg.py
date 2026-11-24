"""Config of the night-gated fine-tune (aic_nightgate.py) from the pure-RGB Swin-L config: everything is the pure-RGB
recipe except
  model         AICFrozenCoDETR (all detector weights frozen, backbone in eval mode) with AICNightGateDataPreprocessor
                (image-level gate lo 0.02 / hi 0.05, fusion net hidden 32, 5 layers, last layer zero)
  pipelines     LoadImageFromFile → LoadAICGateInputs (+ thermal, window, raw depth) in train / val / test; training
                adds AICRandomNightBGR after LoadAnnotations (prob 1: a normal-light sample has G = 0 and gives the
                branch no gradient)
  data          splitA_train.json (1700 images), so the 300 split-A held-out images stay unseen by the branch and
                the night probes on them are clean for it (the frozen detector saw them in daylight)
  schedule      lr 1e-3 (AdamW, weight decay and grad clip as in the recipe), 200-iteration linear warmup, cosine to
                5 %, `epochs` epochs, no validation during training; load_from = the pure-RGB checkpoint
The unused top-level `train_pipeline` / `load_pipeline` variables of the recipe are left as they are.
Run in the teammate's env with PYTHONPATH=<mmdet root>:<his codino scripts>:<this dir>:<scripts/train/codino/aic_mm>.
"""
import argparse
import copy

from mmengine.config import Config


def swap_loader(pipeline: list, loader: dict) -> list:
    out = [copy.deepcopy(loader) if t.get('type') == 'LoadImageFromFile' else t for t in pipeline]
    if sum(t.get('type') == 'LoadAICGateInputs' for t in out) != 1:
        raise ValueError(f'expected exactly one LoadImageFromFile at the top of {pipeline}')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True, help='pure-RGB config (codino_aic_full_rgb_s20260918.py)')
    ap.add_argument('--load-from', required=True, help='pure-RGB checkpoint')
    ap.add_argument('--work-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--ir-dir', default='/root/autodl-tmp/AIC/dataset/train/infrared')
    ap.add_argument('--depth-dir', default='/root/autodl-tmp/AIC/dataset/train/depth')
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--train-ann', default='splitA_train.json')
    a = ap.parse_args()

    cfg = Config.fromfile(a.base)
    cfg.custom_imports = dict(allow_failed_imports=False,
                              imports=list(cfg.custom_imports['imports']) + ['aic_nightgate'])
    cfg.model.type = 'AICFrozenCoDETR'
    cfg.model.trainable = ('data_preprocessor.fuse.',)
    cfg.model.data_preprocessor.type = 'AICNightGateDataPreprocessor'
    cfg.model.data_preprocessor.gate = dict(lo=0.02, hi=0.05)
    cfg.model.data_preprocessor.update(hidden=32, layers=5, erode_px=5, feather_px=8)

    loader = dict(type='LoadAICGateInputs', ir_dir=a.ir_dir, depth_dir=a.depth_dir)
    train = swap_loader(cfg.train_dataloader.dataset.pipeline, loader)
    i = [t.get('type') for t in train].index('LoadAnnotations')
    train.insert(i + 1, dict(type='AICRandomNightBGR', prob=1.0, level_range=(0.003, 0.08), photon=70.0, read=0.012))
    cfg.train_dataloader.dataset.pipeline = train
    cfg.train_dataloader.dataset.ann_file = a.train_ann
    for key in ('val_dataloader', 'test_dataloader'):
        cfg[key].dataset.pipeline = swap_loader(cfg[key].dataset.pipeline, loader)
    cfg.test_pipeline = swap_loader(cfg.test_pipeline, loader)

    cfg.load_from = a.load_from
    cfg.optim_wrapper.optimizer.lr = a.lr
    cfg.train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=a.epochs, val_interval=a.epochs + 1)
    cfg.param_scheduler = [
        dict(type='LinearLR', start_factor=1e-3, by_epoch=False, begin=0, end=200),
        dict(type='CosineAnnealingLR', eta_min=a.lr * 0.05, begin=0, end=a.epochs, T_max=a.epochs, by_epoch=True,
             convert_to_iter_based=True),
    ]
    for h in cfg.custom_hooks:
        if h.get('type') == 'AicIterRecorderHook':
            h['out_file'] = a.work_dir.rstrip('/') + '/iter_log_first.jsonl'
    cfg.work_dir = a.work_dir
    cfg.dump(a.out)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
