#!/usr/bin/env python3
"""CPU contracts for the RGB-only control of clean baseline v1: no training or parameter updates.

Needs the dataset and the pinned weights under vendor/DEIMv2/ckpts. If the LBG-IR folder named in
aic_clean_baseline_v1.yml is absent, set AIC_TRAIN_LBG_IR to an equivalent folder; it is only used to
build the v1 dataset whose RGB channels are compared, never by the RGB control.
"""
import contextlib
import inspect
import io
import os
from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
FW = ROOT/'vendor/DEIMv2'
sys.path[:0] = [str(FW), str(ROOT/'scripts/lib')]
os.chdir(FW)  # data and checkpoint paths in both configs are relative to vendor/DEIMv2
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import numpy as np
import torch
import torch.nn.functional as F
from engine.core import YAMLConfig
from engine.core.yaml_utils import load_config
from engine.backbone.hgnetv2_mm import HGNetv2MultiModal
from engine.misc.numerical_contract import require_finite
from engine.solver._solver import BaseSolver

V1 = FW/'configs/deimv2/aic_clean_baseline_v1.yml'
RGB = FW/'configs/deimv2/aic_clean_baseline_v1_rgb_control.yml'
SPLITS = ('train_dataloader', 'val_dataloader')
AUX = ('ir_', 'occlusion_gate', 'coord_lift', 'dcda', 'cssa', 'trunk_bridge', 'bilateral', 'rfgm')
NEW_AT_TUNING = ('encoder.p2_refinement.', 'dinov3_distiller.', 'co_aux_head.', 'decoder.denoising_class_embed.')
MISSING = object()


def flat(d, p=''):
    if isinstance(d, dict):
        return {k: v for key, val in d.items() for k, v in flat(val, f'{p}{key}.').items()}
    if isinstance(d, list) and any(isinstance(v, dict) for v in d):
        return {k: v for i, val in enumerate(d) for k, v in flat(val, f'{p}{i}.').items()}
    return {p[:-1]: d}


def rgb_key(k):
    """v1 parameter name -> RGB-control name (the RGB trunk lives under backbone.rgb_net in v1)."""
    return 'backbone.' + k[len('backbone.rgb_net.'):] if k.startswith('backbone.rgb_net.') else k


def v1_overrides():
    configured = load_config(str(V1), cfg={})['val_dataloader']['dataset']['raw_ir_folder']
    if Path(configured).is_dir():
        return {}
    lbg = os.environ.get('AIC_TRAIN_LBG_IR')
    if not lbg:
        raise unittest.SkipTest(f'{configured} missing; set AIC_TRAIN_LBG_IR to compare against the v1 dataset')
    print(f'[v1 dataset] raw_ir_folder {configured} missing; using {lbg}', file=sys.stderr)
    return {s: {'dataset': {'raw_ir_folder': lbg}} for s in SPLITS}


def seed(i):
    random.seed(i)
    np.random.seed(i)
    torch.manual_seed(i)


class ConfigContracts(unittest.TestCase):
    def test_config_differs_from_v1_only_in_modality_keys(self):
        a, b = load_config(str(V1), cfg={}), load_config(str(RGB), cfg={})
        fa, fb = flat(a), flat(b)
        diff = {k for k in fa.keys() | fb.keys() if fa.get(k, MISSING) != fb.get(k, MISSING)}
        allowed = {'DEIM.backbone', 'output_dir', 'summary_dir', 'init_ir_stem_after_tuning'}
        for s in SPLITS:
            ops = a[s]['dataset']['transforms']['ops']
            i = next(j for j, op in enumerate(ops) if op['type'] == 'Normalize')
            allowed |= {f'{s}.dataset.{k}' for k in ('ir_folder', 'raw_ir_folder', 'depth_folder', 'destripe_ir', 'n_channels', 'modality')}
            allowed |= {f'{s}.dataset.transforms.ops.{i}.mean', f'{s}.dataset.transforms.ops.{i}.std'}
            norm = b[s]['dataset']['transforms']['ops'][i]
            self.assertEqual((norm['mean'], norm['std']), (ops[i]['mean'][:3], ops[i]['std'][:3]))
            self.assertEqual((b[s]['dataset']['n_channels'], b[s]['dataset']['modality']), (3, 'rgb'))
            self.assertTrue(b[s]['dataset']['strict_inputs'])
            for k in ('ir_folder', 'raw_ir_folder', 'depth_folder'):
                self.assertNotIn(k, b[s]['dataset'])
        extra = sorted(k for k in diff - allowed if not k.startswith(('HGNetv2.', 'HGNetv2MultiModal.')))
        self.assertFalse(extra, f'unexpected differences from v1: {extra}')
        self.assertEqual(b['DEIM']['backbone'], 'HGNetv2')
        self.assertNotIn('HGNetv2MultiModal', b)
        self.assertFalse(b['init_ir_stem_after_tuning'])
        self.assertNotEqual(b['output_dir'], a['output_dir'])
        # Every RGB-trunk argument equals what HGNetv2MultiModal passes to its rgb_net in v1.
        defaults = inspect.signature(HGNetv2MultiModal).parameters
        for k, v in b['HGNetv2'].items():
            self.assertEqual(v, a['HGNetv2MultiModal'].get(k, defaults[k].default), k)

    def test_strict_rgb_dataset_rejects_aux_folders(self):
        from engine.data.dataset.coco_dataset_mm import CocoDetectionMultiModal
        conf = load_config(str(RGB), cfg={})['val_dataloader']['dataset']
        args = dict(img_folder=conf['img_folder'], ann_file=conf['ann_file'], n_channels=3, modality='rgb', strict_inputs=True)
        with contextlib.redirect_stdout(io.StringIO()):
            CocoDetectionMultiModal(**args)
            for bad in (dict(n_channels=6), dict(depth_folder='x'), dict(ir_folder='x')):
                with self.assertRaises(ValueError):
                    CocoDetectionMultiModal(**{**args, **bad})


class DataContracts(unittest.TestCase):
    def test_rgb_inputs_and_targets_equal_v1_first_three_channels(self):
        over = v1_overrides()
        for split in SPLITS:
            with contextlib.redirect_stdout(io.StringIO()):
                d1 = getattr(YAMLConfig(str(V1), **over), split).dataset
                d3 = getattr(YAMLConfig(str(RGB)), split).dataset
            self.assertEqual(list(d3.ids), list(d1.ids))
            small = [i for i, im in enumerate(d3.ids) if d3.coco.imgs[im]['width'] < 1920][:1]
            for i in [0, 1] + small:
                seed(i)
                x1, t1 = d1[i]
                seed(i)
                x3, t3 = d3[i]
                self.assertEqual((tuple(x3.shape), x1.shape[0]), ((3, 1440, 2560), 6), (split, i))
                torch.testing.assert_close(x3, x1[:3], rtol=0, atol=0)
                self.assertEqual(set(t3), set(t1) - {'depth'})
                for k in t3:
                    torch.testing.assert_close(torch.as_tensor(t3[k]), torch.as_tensor(t1[k]), rtol=0, atol=0)


class ModelContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfgs, cls.models, cls.reports = {}, {}, {}
        for name, path in (('v1', V1), ('rgb', RGB)):
            cfg = YAMLConfig(str(path))
            seed(42)
            with contextlib.redirect_stdout(io.StringIO()):
                model = cfg.model.float()
                solver = BaseSolver(cfg)
                solver.model = model
                solver.load_tuning_state(cfg.tuning)
            cls.cfgs[name], cls.models[name], cls.reports[name] = cfg, model, solver.tuning_load_report

    def test_backbone_is_the_v1_rgb_trunk_without_aux_modules(self):
        v1, rgb = self.models['v1'], self.models['rgb']
        self.assertEqual(type(rgb.backbone).__name__, 'HGNetv2')
        self.assertEqual(rgb.backbone.stem.stem1.conv.weight.shape[1], 3)
        self.assertEqual(str(rgb.backbone), str(v1.backbone.rgb_net))
        names = [k for k, _ in rgb.named_parameters()] + [k for k, _ in rgb.named_buffers()]
        self.assertFalse([k for k in names if k.startswith('backbone.') and any(m in k for m in AUX)])
        self.assertTrue([k for k, _ in v1.named_parameters() if k.startswith('backbone.') and any(m in k for m in AUX)])

    def test_all_non_aux_parameters_match_v1(self):
        def shared(model, drop_aux):
            out = {}
            for kind, items in (('param', model.named_parameters()), ('buffer', model.named_buffers())):
                for k, v in items:
                    if drop_aux and k.startswith('backbone.') and not k.startswith('backbone.rgb_net.'):
                        continue
                    out[(kind, rgb_key(k))] = (tuple(v.shape), v.requires_grad)
            return out
        self.assertEqual(shared(self.models['rgb'], False), shared(self.models['v1'], True))
        count = lambda m: sum(p.numel() for k, p in m.named_parameters() if not k.startswith(('dinov3_distiller.', 'co_aux_head.')))
        print(f"\n[inference params] v1 {count(self.models['v1'])/1e6:.1f}M, rgb control {count(self.models['rgb'])/1e6:.1f}M",
              file=sys.stderr)

    def test_tuning_loads_the_same_coco_and_teacher_tensors(self):
        r1, r3 = self.reports['v1'], self.reports['rgb']
        s1, s3 = self.models['v1'].state_dict(), self.models['rgb'].state_dict()
        loaded = lambda s, r: set(s) - set(r['missed']) - set(r['unmatched'])
        l1, l3 = {rgb_key(k) for k in loaded(s1, r1)}, loaded(s3, r3)
        self.assertEqual(l3, l1)
        self.assertEqual(r3['matched_keys'], r1['matched_keys'])
        self.assertTrue(all(k.startswith(NEW_AT_TUNING) for k in r3['missed']), r3['missed'])
        self.assertTrue(r1['ir_stem_copied_after_rgb_load'])
        self.assertNotIn('ir_stem_copied_after_rgb_load', r3)
        inverse = {rgb_key(k): k for k in s1}
        # The frozen teacher comes from the pinned file; its projection head is new, randomly initialized.
        for k in l3 | {k for k in s3 if k.startswith('dinov3_distiller.teacher.')}:
            torch.testing.assert_close(s3[k], s1[inverse[k]], rtol=0, atol=0, msg=k)

    def test_cpu_train_step_and_full_resolution_eval_are_finite(self):
        cfg, model = self.cfgs['rgb'], self.models['rgb']
        with contextlib.redirect_stdout(io.StringIO()):
            train = YAMLConfig(str(RGB)).train_dataloader.dataset
            val = YAMLConfig(str(RGB)).val_dataloader.dataset
        seed(0)
        x, t = train[0]
        # Reduced-size training forward keeps the CPU check tractable; boxes are normalized cxcywh.
        x = F.interpolate(x[None], size=(640, 1152), mode='bilinear', align_corners=False)
        buffers = {k: v.clone() for k, v in model.named_buffers()}  # train-mode BN stats must not leak
        model.train()
        out = model(x, targets=[t])
        require_finite(out.items(), 'rgb control train forward')
        losses = cfg.criterion(out, [t], epoch=0, step=0, global_step=0, epoch_step=900)
        total = sum(losses.values())
        require_finite([*losses.items(), ('total_loss', total)], 'rgb control loss')
        self.assertTrue({'loss_distill', 'loss_co_cls', 'loss_co_reg'} <= set(losses))
        self.assertNotIn('loss_cmal', losses)  # DCDA alignment loss exists only in v1
        model.zero_grad(set_to_none=True)
        total.backward()
        grads = [(k, p.grad) for k, p in model.named_parameters() if p.grad is not None]
        require_finite(grads, 'rgb control gradients')
        self.assertTrue(any(k.startswith('backbone.') and float(g.abs().sum()) > 0 for k, g in grads))
        self.assertFalse([k for k, _ in grads if k.startswith('dinov3_distiller.teacher.')])
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            for k, v in model.named_buffers():
                v.copy_(buffers[k])
        model.eval()
        xv, tv = val[0]
        with torch.no_grad():
            out = model(xv[None])
            result = cfg.postprocessor(out, torch.stack([tv['orig_size']]))
        self.assertEqual(tuple(out['pred_logits'].shape), (1, 300, 12))
        require_finite(out.items(), 'rgb control eval forward')
        require_finite(result[0].items(), 'rgb control postprocessor')


if __name__ == '__main__':
    torch.set_num_threads(8)
    unittest.main(verbosity=2)
