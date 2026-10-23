"""
DEIMv2: Real-Time Object Detection Meets DINOv3
Copyright (c) 2025 The DEIMv2 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import sys
_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(_root, 'scripts/lib'))
sys.path.insert(0, os.path.join(_root, 'vendor/Depth-Anything-V2'))

import argparse
import torch
import torch.multiprocessing as mp
try:
    mp.set_sharing_strategy('file_system')
except Exception:
    pass

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    # Volta SM_70 (Tesla V100) Winograd FP16 bug workaround for 2.5K high-res multi-modal inputs
    if torch.cuda.get_device_capability() == (7, 0):
        torch.backends.cudnn.enabled = False
        print("[System Info] Volta SM_70 architecture detected: disabled cuDNN to use native cuBLAS/ATen GEMM for memory alignment safety.")

from engine.misc import dist_utils
from engine.core import YAMLConfig, yaml_utils
from engine.solver import TASKS

debug=False

if debug:
    import torch
    def custom_repr(self):
        return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'
    original_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = custom_repr

def main(args, ) -> None:
    """main
    """
    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'


    update_dict = yaml_utils.parse_cli(args.update)
    update_dict.update({k: v for k, v in args.__dict__.items() \
        if k not in ['update', ] and v is not None})

    cfg = YAMLConfig(args.config, **update_dict)
    # Resolve the YAML seed before seeding; an omitted CLI flag must not erase it.
    if cfg.seed is None:
        cfg.seed = 0
    cfg.yaml_cfg['seed'] = cfg.seed
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=cfg.seed, strict=cfg.clean_baseline)
    if cfg.clean_baseline:
        if torch.cuda.is_available():
            if torch.cuda.get_device_capability()[0] >= 8:
                # On SM >= 8.0 (Ampere/Ada/Blackwell), allow TF32 and BF16 AMP (has full FP32 exponent range)
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            else:
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                if cfg.use_amp:
                    raise ValueError('Clean baseline requires FP32 on legacy SM < 8.0 (use_amp=False)')
        if cfg.tuning and cfg.resume:
            raise ValueError('Select tuning or resume, not both')

    if cfg.resume or cfg.tuning:
        for bb_name in ['HGNetv2', 'HGNetv2MultiModal', 'HGNetv2ViTDualStream']:
            if bb_name in cfg.yaml_cfg:
                cfg.yaml_cfg[bb_name]['pretrained'] = False

    print('cfg: ', cfg.__dict__)

    solver = TASKS[cfg.yaml_cfg['task']](cfg)

    if args.test_only:
        solver.val()
    else:
        solver.fit()

    dist_utils.cleanup()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # priority 0
    parser.add_argument('-c', '--config', type=str, default='')
    parser.add_argument('-r', '--resume', type=str, help='resume from checkpoint')
    parser.add_argument('-t', '--tuning', type=str, help='tuning from checkpoint')
    parser.add_argument('-d', '--device', type=str, help='device',)
    parser.add_argument('--seed', type=int, default=None, help='Override YAML seed; otherwise use YAML or 0')
    parser.add_argument('--use-amp', action='store_true', default=None, help='auto mixed precision training')
    parser.add_argument('--output-dir', type=str, help='output directoy')
    parser.add_argument('--summary-dir', type=str, help='tensorboard summry')
    parser.add_argument('--test-only', action='store_true', default=False,)

    # priority 1
    parser.add_argument('-u', '--update', nargs='+', help='update yaml config')

    # env
    parser.add_argument('--print-method', type=str, default='builtin', help='print method')
    parser.add_argument('--print-rank', type=int, default=0, help='print rank id')

    parser.add_argument('--local-rank', type=int, help='local rank id')
    args = parser.parse_args()

    main(args)
