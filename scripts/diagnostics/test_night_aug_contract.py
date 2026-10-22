#!/usr/bin/env python3
"""CPU contracts for the dark-light arm of clean baseline v1: no training or parameter updates.

Needs the dataset. If the LBG-IR folder named in the configs is absent, set AIC_TRAIN_LBG_IR to an
equivalent folder (applied to both configs; it only feeds IR channel 4, which must stay untouched).
"""
import contextlib
import copy
import io
import os
from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
FW = ROOT/'vendor/DEIMv2'
sys.path[:0] = [str(FW), str(ROOT/'scripts/lib')]
os.chdir(FW)  # data paths in the configs are relative to vendor/DEIMv2
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import numpy as np
import torch
from engine.core import YAMLConfig
from engine.core.yaml_utils import load_config
import engine.data.transforms._transforms as tf

V1 = FW/'configs/deimv2/aic_clean_baseline_v1.yml'
NIGHT = FW/'configs/deimv2/aic_clean_baseline_v1_night.yml'
OP = dict(type='RandomDNDTNightify', p=0.5, bright_range=[0.0005, 0.5], log_bright=True, curriculum=False, seeded=True)


def seed(i):
    random.seed(i)
    np.random.seed(i)
    torch.manual_seed(i)


def lbg_override():
    configured = load_config(str(V1), cfg={})['train_dataloader']['dataset']['raw_ir_folder']
    if Path(configured).is_dir():
        return {}
    lbg = os.environ.get('AIC_TRAIN_LBG_IR')
    if not lbg:
        raise unittest.SkipTest(f'{configured} missing; set AIC_TRAIN_LBG_IR')
    return {s: {'dataset': {'raw_ir_folder': lbg}} for s in ('train_dataloader', 'val_dataloader')}


class NightContracts(unittest.TestCase):
    def test_config_is_v1_plus_one_op(self):
        a, b = load_config(str(V1), cfg={}), copy.deepcopy(load_config(str(NIGHT), cfg={}))
        ops = b['train_dataloader']['dataset']['transforms']['ops']
        found = [i for i, op in enumerate(ops) if op['type'] == 'RandomDNDTNightify']
        self.assertEqual(len(found), 1)
        self.assertEqual(ops[found[0]], OP)
        self.assertEqual(ops[found[0]-1]['type'], 'RandomHorizontalFlip')  # flip draws first: same flips as v1
        self.assertNotIn('RandomDNDTNightify', b['train_dataloader']['dataset']['transforms']['policy']['ops'])
        del ops[found[0]]
        self.assertEqual((b.pop('output_dir'), b.pop('summary_dir')),
                         ('outputs/aic_clean_baseline_v1_night', '../../logs/aic_clean_baseline_v1_night/tensorboard'))
        a.pop('output_dir'), a.pop('summary_dir')
        self.assertEqual(a, b)

    def test_op_darkens_rgb_only_and_is_seeded(self):
        self.assertIsNotNone(tf.nightify, 'nightify_aic not importable: add scripts/lib to PYTHONPATH')
        over = lbg_override()
        with contextlib.redirect_stdout(io.StringIO()):
            ds = YAMLConfig(str(NIGHT), **over).train_dataloader.dataset
        x, _ = ds.load_item(0)                      # raw [0,1] 6-channel tensor before transforms
        args = {k: v for k, v in OP.items() if k != 'type'}
        op = tf.RandomDNDTNightify(**dict(args, p=1.0))
        seed(3)
        y1 = op(x)
        seed(3)
        y2 = op(x)
        seed(4)
        y3 = op(x)
        self.assertTrue(torch.equal(y1, y2), 'same torch seed must give the same darkening')
        self.assertFalse(torch.equal(y1, y3))
        self.assertTrue(torch.equal(y1[3:], x[3:]), 'IR/depth must pass through bitwise')
        self.assertLess(float(y1[:3].mean()), float(x[:3].mean()))
        self.assertGreaterEqual(float(y1.min()), 0.)
        self.assertLessEqual(float(y1[:3].max()), 1.)
        unseeded = tf.RandomDNDTNightify(**dict(args, p=1.0, seeded=False))
        self.assertNotIn('seed', unseeded._get_params([x]))  # historical configs keep fresh entropy

    def test_train_samples_match_v1_except_rgb(self):
        over = lbg_override()
        with contextlib.redirect_stdout(io.StringIO()):
            d1 = YAMLConfig(str(V1), **over).train_dataloader.dataset
            dn = YAMLConfig(str(NIGHT), **over).train_dataloader.dataset
        darkened = 0
        for i in range(12):
            seed(100+i)
            x1, t1 = d1[i]
            seed(100+i)
            xn, tn = dn[i]
            self.assertEqual(x1.shape, xn.shape)
            self.assertTrue(torch.equal(x1[3:], xn[3:]), f'aux channels differ at sample {i}')
            for k in ('boxes', 'labels'):
                torch.testing.assert_close(torch.as_tensor(tn[k]), torch.as_tensor(t1[k]), rtol=0, atol=0)
            darkened += not torch.equal(x1[:3], xn[:3])
        self.assertTrue(0 < darkened < 12, darkened)
        print(f'\n[night arm] {darkened}/12 seeded training samples darkened (p=0.5)', file=sys.stderr)


if __name__ == '__main__':
    torch.set_num_threads(8)
    unittest.main(verbosity=2)
