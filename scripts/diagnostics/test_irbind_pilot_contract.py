#!/usr/bin/env python3
"""CPU contracts for the IR-binding pilot (aic_irbind_pilot_ws_v1.yml): no data, no training, no parameter updates.

  CUDA_VISIBLE_DEVICES='' $PY scripts/diagnostics/test_irbind_pilot_contract.py
"""
import contextlib
import io
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'vendor/DEIMv2'), str(ROOT/'scripts/lib')]
import torch
from torch import nn
from engine.core import BaseConfig
from engine.core.yaml_utils import load_config
from engine.solver._solver import BaseSolver
from engine.backbone.hgnetv2_mm import HGNetv2MultiModal
import engine.data.transforms._transforms as T

PILOT = ROOT/'vendor/DEIMv2/configs/deimv2/aic_irbind_pilot_ws_v1.yml'
PER_STAGE = ('ir_proj', 'ir_fuse', 'ir_res', 'ir_align', 'occlusion_gate', 'coord_lift')


def backbone(stages, seed):
    torch.manual_seed(seed)
    m = nn.Module()
    m.backbone = HGNetv2MultiModal(name='B5', ir_in_chans=2, inject_stages=stages, return_idx=[0, 1, 2, 3], pretrained=False,
                                   pretrained_ir_stem=False, use_dcda=True, use_dog=True, use_3d_lift=True, disable_fallback=True)
    return m


def solver(model, remap):
    cfg = BaseConfig()
    cfg.clean_baseline = True
    cfg.tuning_key_remap = remap
    s = BaseSolver(cfg)
    s.model = model
    return s


class PilotContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config(str(PILOT))
        cls.v1 = backbone([2, 3], 0)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.ckpt = Path(cls.tmp.name)/'v1_like.pth'
        torch.save({'model': cls.v1.state_dict()}, cls.ckpt)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_remap_keeps_each_stage_module_and_new_gate_is_identity(self):
        pilot = backbone([1, 2, 3], 1)
        s = solver(pilot, self.cfg['tuning_key_remap'])
        with contextlib.redirect_stdout(io.StringIO()):
            s.load_tuning_state(str(self.ckpt))
        old, new = self.v1.backbone, pilot.backbone
        for name in PER_STAGE:
            for k_old, k_new in ((0, 1), (1, 2)):
                for (n, a), (_, b) in zip(getattr(old, name)[k_old].state_dict().items(), getattr(new, name)[k_new].state_dict().items()):
                    torch.testing.assert_close(b, a, rtol=0, atol=0, msg=f'{name}[{k_new}].{n}')
        self.assertEqual(float(new.ir_res[0].weight.detach().abs().sum() + new.ir_res[0].bias.detach().abs().sum()), 0.0)
        torch.testing.assert_close(new.ir_stem.stem[0].weight, old.ir_stem.stem[0].weight, rtol=0, atol=0)
        torch.testing.assert_close(new.rgb_net.stem.stem1.conv.weight, old.rgb_net.stem.stem1.conv.weight, rtol=0, atol=0)
        self.assertTrue(all(n > 0 for n in s.tuning_load_report['remapped_keys'].values()))

    def test_without_remap_the_shifted_modules_are_refused(self):
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
            solver(backbone([1, 2, 3], 1), None).load_tuning_state(str(self.ckpt))

    def test_remap_fails_closed(self):
        for remap in ([['backbone.ir_proj.0', 'backbone.ir_proj.1']],            # no trailing dot
                      [['backbone.ir_nothing.0.', 'backbone.ir_nothing.1.']],    # matches no key
                      [['backbone.ir_proj.0.', 'backbone.ir_proj.1.']]):         # collides with the unmoved stage-3 keys
            with self.subTest(remap=remap), contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                solver(backbone([1, 2, 3], 1), remap).load_tuning_state(str(self.ckpt))

    def test_lr_groups_are_disjoint_and_gate_learns_with_ir_stream(self):
        groups = [(g['params'], g.get('lr', self.cfg['optimizer']['lr'])) for g in self.cfg['optimizer']['params']]
        names = ['backbone.' + n for n, _ in backbone([1, 2, 3], 1).backbone.named_parameters()]
        for n in names:
            hits = [lr for pat, lr in groups if re.findall(pat, n)]
            self.assertLessEqual(len(hits), 1, n)
            if re.search(r'\.ir_(fuse|res)\.', n):
                self.assertEqual(hits, [1e-4], n)

    def test_pilot_switches(self):
        c, hg = self.cfg, self.cfg['HGNetv2MultiModal']
        self.assertEqual(hg['inject_stages'], [1, 2, 3])
        self.assertEqual(hg['ir_drop_prob'], 0.0)
        self.assertFalse(c['init_ir_stem_after_tuning'])
        self.assertTrue(c['clean_baseline'])
        self.assertEqual(len(c['required_tuning_sha256']), 64)
        night = [o for o in c['train_dataloader']['dataset']['transforms']['ops'] if o['type'] == 'RandomDNDTNightify']
        self.assertEqual(len(night), 1)
        self.assertTrue(night[0]['strict'])
        self.assertNotIn('RandomDNDTNightify', c['train_dataloader']['dataset']['transforms']['policy']['ops'])

    def test_strict_nightify_refuses_without_backend(self):
        with patch.object(T, 'nightify', None):
            with self.assertRaises(ImportError):
                T.RandomDNDTNightify(strict=True)
            T.RandomDNDTNightify(strict=False)


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main(verbosity=2)
