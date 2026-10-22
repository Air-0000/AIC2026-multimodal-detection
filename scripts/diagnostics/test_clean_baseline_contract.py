#!/usr/bin/env python3
"""CPU contracts only: no real dataset/model training or parameter updates."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'vendor/DEIMv2'), str(ROOT/'scripts/lib')]
import numpy as np
import torch
from torch import nn
from engine.core import BaseConfig, YAMLConfig
from engine.core.yaml_utils import load_config
from engine.solver._solver import BaseSolver
from engine.solver.det_engine import train_one_epoch, evaluate
from engine.solver.clean_lifecycle import fit_clean, atomic_checkpoint
from engine.misc.numerical_contract import require_finite
from engine.backbone.dinov3_teacher import DINOv3TeacherDistiller


class NoUpdateOptimizer:
    """Observe gradients without making any parameter updates."""
    def __init__(self, parameter):
        self.p = parameter
        self.steps = 0
        self.param_groups = [{'lr':.1}]
        self.observed = None
    def zero_grad(self): self.p.grad = None
    def step(self):
        self.steps += 1
        self.observed = self.p.grad.clone()


class ScalarFixture(nn.Module):
    def __init__(self, invalid=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.invalid = invalid
    def forward(self, samples, targets=None):
        value = samples*self.weight
        if self.invalid: value = value*float('nan')
        return {'pred_logits':value, 'pred_boxes':value}


class ScalarLoss(nn.Module):
    def forward(self, outputs, targets, **kwargs):
        return {'loss_scalar': outputs['pred_logits'].square().mean()}


class CleanContracts(unittest.TestCase):
    def test_yaml_loads_are_independent(self):
        with tempfile.TemporaryDirectory() as t:
            a,b=Path(t)/'a.yml',Path(t)/'b.yml'
            a.write_text('num_classes: 80\nold_only: true\n');b.write_text('num_classes: 12\n')
            x=load_config(str(a));y=load_config(str(b))
            self.assertNotIn('old_only',y);self.assertEqual(x['num_classes'],80)

    def test_yaml_seed_survives_omitted_cli(self):
        spec=importlib.util.spec_from_file_location('baseline_train_entry',ROOT/'vendor/DEIMv2/train.py')
        module=importlib.util.module_from_spec(spec)
        with patch('torch.cuda.is_available',return_value=False): spec.loader.exec_module(module)
        seen={}
        class Solver:
            def __init__(self,cfg): seen['seed']=cfg.seed
            def val(self): pass
        args=SimpleNamespace(config=str(ROOT/'vendor/DEIMv2/configs/deimv2/aic_clean_baseline_v1.yml'),
              update=None,seed=None,use_amp=None,tuning=None,resume=None,print_rank=0,print_method='builtin',test_only=True)
        with patch.object(module.dist_utils,'setup_distributed') as setup, patch.object(module.dist_utils,'cleanup'),patch.dict(module.TASKS,{'detection':Solver}),contextlib.redirect_stdout(io.StringIO()):
            module.main(args)
        self.assertEqual(seen['seed'],42);self.assertEqual(setup.call_args.kwargs['seed'],42)

    def test_frozen_teacher_stays_eval(self):
        fake=nn.Sequential(nn.Dropout(.5),nn.Linear(32,32))
        with patch('engine.backbone.dinov3_teacher.DinoVisionTransformer',return_value=fake),patch.object(DINOv3TeacherDistiller,'_find_checkpoint',return_value=None),contextlib.redirect_stdout(io.StringIO()):
            m=DINOv3TeacherDistiller(student_dim=32,teacher_dim=32)
        m.train()
        self.assertTrue(m.proj_head.training);self.assertFalse(m.teacher.training)
        self.assertFalse(any(p.requires_grad for p in m.teacher.parameters()))
        with patch.object(DINOv3TeacherDistiller,'_find_checkpoint',return_value=None),self.assertRaises(FileNotFoundError):
            DINOv3TeacherDistiller(require_pretrained=True)

    def test_finite_guard_preserves_anchor_sentinel(self):
        anchors=torch.tensor([float('inf'),1.])
        require_finite([('decoder.anchors',anchors)],'fixture',allow_anchor_infinity=True)
        self.assertTrue(torch.isinf(anchors[0]))
        with self.assertRaises(FloatingPointError): require_finite([('bn.running_mean',anchors)],'fixture',True)
        with self.assertRaises(FloatingPointError): require_finite([('decoder.anchors',torch.tensor(float('nan')))],'fixture',True)
        with self.assertRaises(FloatingPointError): require_finite([('aux_outputs',[{'scores':torch.tensor(float('nan'))}])],'fixture')

    def test_wrong_teacher_hash_rejected_before_construction(self):
        with tempfile.TemporaryDirectory() as t:
            f=Path(t)/'teacher.pth';f.write_bytes(b'wrong source')
            with patch.object(DINOv3TeacherDistiller,'_find_checkpoint',return_value=str(f)),patch('engine.backbone.dinov3_teacher.DinoVisionTransformer') as create,self.assertRaises(ValueError):
                DINOv3TeacherDistiller(model_path=str(f),require_pretrained=True,pretrained_sha256='0'*64)
            create.assert_not_called()

    def test_nonfinite_forward_stops_before_update_and_ema(self):
        model=ScalarFixture(invalid=True);opt=NoUpdateOptimizer(model.weight)
        ema=SimpleNamespace(update=lambda _: self.fail('EMA must not update'))
        with contextlib.redirect_stdout(io.StringIO()),self.assertRaises(FloatingPointError):
            train_one_epoch(False,None,model,ScalarLoss(),[(torch.ones(1,1),[{}])],opt,torch.device('cpu'),0,ema=ema,stop_on_nonfinite=True)
        self.assertEqual(opt.steps,0);self.assertEqual(model.weight.item(),1.)

    def test_unequal_microbatches_weight_by_sample_count_without_updates(self):
        model=ScalarFixture();opt=NoUpdateOptimizer(model.weight)
        samples=torch.arange(1.,6.).view(-1,1)
        with contextlib.redirect_stdout(io.StringIO()):
            train_one_epoch(False,None,model,ScalarLoss(),[(samples,[{}]*5)],opt,torch.device('cpu'),0,accum_steps=2,stop_on_nonfinite=True)
        self.assertAlmostEqual(opt.observed.item(),2*samples.square().mean().item(),places=5)
        self.assertEqual(model.weight.item(),1.)

    def test_strict_resume_does_not_swallow_shape_mismatch(self):
        cfg=BaseConfig();cfg.strict_resume=True
        s=BaseSolver(cfg);s.model=nn.Linear(2,1);s.ema=None
        with self.assertRaises(RuntimeError): s.load_state_dict({'last_epoch':0,'model':{'weight':torch.ones(2,2),'bias':torch.zeros(1)}})
        with self.assertRaises(KeyError): s.load_state_dict({'last_epoch':0})

    def test_unrelated_class_head_mapping_is_not_coco_obj365(self):
        s=BaseSolver(BaseConfig())
        self.assertIsNone(s.map_class_weights(torch.ones(12,4),torch.ones(80,4)))

    def test_resume_restores_rng_and_optimizer_without_parameter_updates(self):
        cfg=BaseConfig();cfg.clean_baseline=True;cfg.strict_resume=True;cfg.seed=7;cfg.yaml_cfg={'seed':7}
        s=BaseSolver(cfg);s.model=nn.Linear(2,1);s.ema=None;s.last_epoch=3
        s.optimizer=torch.optim.SGD(s.model.parameters(),lr=.02,momentum=.9)
        random.seed(7);np.random.seed(7);torch.manual_seed(7)
        state=s.state_dict()
        expected=(random.random(),float(np.random.rand()),torch.rand(3))
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'resume.pth';torch.save(state,path)
            s.optimizer.param_groups[0]['lr']=.9
            with contextlib.redirect_stdout(io.StringIO()):s.load_resume_state(str(path))
        self.assertEqual(s.optimizer.param_groups[0]['lr'],.02)
        self.assertEqual(random.random(),expected[0]);self.assertEqual(float(np.random.rand()),expected[1])
        torch.testing.assert_close(torch.rand(3),expected[2],atol=0,rtol=0)

    def test_coatss_conflicting_labels_follow_single_regression_owner(self):
        from engine.deim.co_aux_head import CoATSSAuxHead
        with contextlib.redirect_stdout(io.StringIO()):head=CoATSSAuxHead(in_channels=32,num_classes=2,num_convs=1,strides=(8,))
        targets=[{'boxes':torch.tensor([[.5,.5,.4,.4],[.5,.5,.4,.4]]),'labels':torch.tensor([0,1])}]
        observed=[]
        def focal(logits,target,**kw):
            observed.append(target.clone());return logits.sum()*0
        with torch.no_grad(),patch('torchvision.ops.sigmoid_focal_loss',side_effect=focal):head([torch.ones(1,32,5,5)],targets,img_size=(40,40))
        self.assertTrue(observed);self.assertLessEqual(float(observed[0].sum(-1).max()),1.)
        self.assertGreater(float(observed[0][:,0].sum()),0.)
        self.assertEqual(float(observed[0][:,1].sum()),0.)

    def test_post_tuning_stem_uses_loaded_rgb_not_constructor_rgb(self):
        from engine.backbone.hgnetv2_mm import HGNetv2MultiModal
        torch.manual_seed(42)
        module=nn.Module();module.backbone=HGNetv2MultiModal(name='B5',ir_in_chans=2,inject_stages=[2,3],return_idx=[0,1,2,3],pretrained=False,pretrained_ir_stem=False,use_dcda=True,use_dog=True,use_3d_lift=True)
        cfg=BaseConfig();cfg.init_ir_stem_after_tuning=True
        s=BaseSolver(cfg);s.model=module
        with contextlib.redirect_stdout(io.StringIO()):s.load_tuning_state(str(ROOT/'vendor/DEIMv2/ckpts/dfine_x_obj2coco.pth'))
        rgb=module.backbone.rgb_net.stem.stem1.conv.weight.detach();ir=module.backbone.ir_stem.stem[0].weight.detach()
        mean=rgb.mean(1,keepdim=True);lap=torch.tensor([[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]]).view(1,1,3,3)*.5
        expected=torch.cat([mean.expand(-1,2,-1,-1),(mean*.5+lap*.5).expand(-1,2,-1,-1)])
        torch.testing.assert_close(ir,expected,rtol=0,atol=0)
        self.assertTrue(s.tuning_load_report['ir_stem_copied_after_rgb_load'])
        # A trained multimodal checkpoint must never be reinitialized this way.
        with tempfile.TemporaryDirectory() as t:
            f=Path(t)/'mm.pth';torch.save({'model':module.state_dict()},f)
            with contextlib.redirect_stdout(io.StringIO()),self.assertRaises(ValueError):s.load_tuning_state(str(f))

    def test_lifecycle_never_rewinds_and_final_is_final_epoch(self):
        class Loader:
            def __init__(self):self.epochs=[]
            def set_epoch(self,e):self.epochs.append(e)
            def __len__(self):return 2
        with tempfile.TemporaryDirectory() as t:
            loader=Loader();cfg=SimpleNamespace(use_amp=False,stop_on_nonfinite=True,strict_resume=True,lr_gamma=.5,epoches=4,warmup_iter=1,flat_epoch=1,no_aug_epoch=1,clip_max_norm=0.,print_freq=1,freeze_bn=False,accum_steps=1,checkpoint_freq=2,seed=42)
            s=SimpleNamespace(cfg=cfg,train=lambda:None,optimizer=SimpleNamespace(param_groups=[{'initial_lr':.1}]),train_dataloader=loader,last_epoch=-1,ema=None,model=None,criterion=None,postprocessor=None,val_dataloader=None,evaluator=None,device='cpu',scaler=None,writer=None,output_dir=Path(t))
            s.state_dict=lambda:dict(last_epoch=s.last_epoch,best_stat=s.best_stat)
            scores=iter([.2,.4,.3,.1]);evaluator=SimpleNamespace(coco_eval={'bbox':SimpleNamespace(eval={})})
            def ev(*a,**k):return {'coco_eval_bbox':[next(scores)]},evaluator
            with patch('engine.solver.clean_lifecycle.train_one_epoch',return_value={'loss':1.}),patch('engine.solver.clean_lifecycle.evaluate',side_effect=ev),contextlib.redirect_stdout(io.StringIO()):fit_clean(s)
            self.assertEqual(loader.epochs,[0,1,2,3])
            final=torch.load(Path(t)/'final.pth',weights_only=False);best=torch.load(Path(t)/'best.pth',weights_only=False)
            self.assertEqual(final['last_epoch'],3);self.assertEqual(best['last_epoch'],1)
            self.assertEqual(final['validation_metrics']['coco_eval_bbox'],[.1])

    def test_effective_baseline_is_explicit_and_has_no_experimental_switches(self):
        c=load_config(str(ROOT/'vendor/DEIMv2/configs/deimv2/aic_clean_baseline_v1.yml'))
        self.assertNotIn('__include__',c)
        self.assertEqual(c['num_classes'],12);self.assertFalse(c['use_amp'])
        self.assertEqual(c['DFINETransformer']['p3_quota_ratio'],.35)
        for key in ('use_sdqs','use_distinct_queries','use_anisotropic_routing','use_offset_clamp','use_spatial_attn_bias','use_layer_ensemble'):self.assertFalse(c['DFINETransformer'][key])
        self.assertTrue(c['stop_on_nonfinite']);self.assertEqual(c['train_dataloader']['total_batch_size'],2)


if __name__=='__main__':
    torch.set_num_threads(4)
    unittest.main(verbosity=2)
