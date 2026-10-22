#!/usr/bin/env python3
"""CPU regression checks for DOG parity, routing exclusivity, and loss gradients."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts/lib'), str(ROOT / 'vendor/DEIMv2')]
import torch
from engine.backbone.hgnetv2_mm import DepthOcclusionGating, HGNetv2MultiModal
from engine.deim.dfine_decoder import DFINETransformer
from engine.deim.deim_criterion import DEIMCriterion
from engine.deim.matcher import HungarianMatcher


def main():
    torch.manual_seed(42)
    torch.set_num_threads(2)
    # Compare to the actual remote training source, with a nonzero learned gate.
    source = ROOT / 'work_dir/audits/v36_mechanism_20261018/source/remote_hgnetv2_mm.py'
    # Extract only the standalone DOG class: importing a second registered
    # backbone would collide with the framework registry.
    import ast
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DepthOcclusionGating')
    namespace = {'torch': torch, 'nn': torch.nn, 'F': torch.nn.functional}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    old = namespace['DepthOcclusionGating'](4).eval()
    torch.nn.init.normal_(old.gate_net[-1].weight, std=0.1)
    torch.nn.init.constant_(old.gate_net[-1].bias, -0.3)
    dog = DepthOcclusionGating(4).eval()
    dog.load_state_dict(old.state_dict(), strict=True)
    ir = torch.randn(1, 4, 8, 8, requires_grad=True)
    depth = torch.rand(1, 1, 16, 16, requires_grad=True)
    torch.testing.assert_close(dog(ir, depth), old(ir, depth), rtol=0, atol=0)
    dog.disable_fallback = False
    changed = dog(ir, depth)
    assert not torch.equal(changed, old(ir, depth))
    changed.sum().backward()
    assert torch.isfinite(ir.grad).all() and torch.isfinite(depth.grad).all()
    # Verify YAML-visible parent option reaches every child DOG.
    parent = HGNetv2MultiModal(pretrained=False, use_dog=True, disable_fallback=False)
    assert all(not x.disable_fallback for x in parent.occlusion_gate)
    del parent
    decoder = DFINETransformer(num_classes=2, use_pure_topk=True, num_queries=6)
    memory = torch.arange(21).view(1, 21, 1).float()
    scores = torch.arange(21).view(1, 21, 1).float().repeat(1, 1, 2)
    anchors = torch.zeros(1, 21, 4)
    selected, _, _ = decoder._select_topk(memory, scores, anchors, 6, [[4, 4], [2, 2], [1, 1]])
    assert selected.flatten().tolist() == [20, 19, 18, 17, 16, 15]
    decoder.use_pure_topk = False
    decoder.use_anisotropic_routing = True
    decoder.p3_quota_ratio = decoder.p4_quota_ratio = 1 / 3
    selected, _, _ = decoder._select_topk(memory, scores, anchors, 6, [[4, 4], [2, 2], [1, 1]])
    assert len(set(selected.flatten().tolist())) == 6
    assert sum(x < 16 for x in selected.flatten().tolist()) >= 2
    matcher = HungarianMatcher(weight_dict={'cost_class': 2, 'cost_bbox': 5, 'cost_giou': 2})
    kwargs = dict(matcher=matcher, weight_dict={'loss_tadl': 1, 'loss_bbox': 1, 'loss_giou': 1},
                  losses=['tadl', 'boxes'], num_classes=2, use_uni_set=False)
    normal = DEIMCriterion(**kwargs)
    discounted = DEIMCriterion(**kwargs, use_discounted_aux_loss=True,
                               aux_loss_discount_losses=['tadl'])
    def prediction():
        return {'pred_logits': torch.randn(1, 3, 2, requires_grad=True),
                'pred_boxes': torch.tensor([[[.45,.5,.2,.2],[.2,.2,.1,.1],[.8,.8,.1,.1]]], requires_grad=True)}
    outputs = prediction()
    outputs.update(aux_outputs=[prediction() for _ in range(5)], pre_outputs=prediction(),
                   enc_aux_outputs=[], enc_meta={'class_agnostic': False})
    targets = [{'labels': torch.tensor([0]), 'boxes': torch.tensor([[.5,.5,.2,.2]])}]
    before, after = normal(outputs, targets), discounted(outputs, targets)
    for key in before:
        factor = 1.0
        if key.startswith('loss_tadl_aux_'):
            factor = [.4,.5,.6,.8,.9][int(key.rsplit('_', 1)[1])]
        elif key == 'loss_tadl_pre':
            factor = .4
        torch.testing.assert_close(after[key], before[key] * factor)
    # Classification gradient changes while geometric gradients remain exact.
    a = torch.autograd.grad(before['loss_bbox_aux_0'], outputs['aux_outputs'][0]['pred_boxes'], retain_graph=True)[0]
    b = torch.autograd.grad(after['loss_bbox_aux_0'], outputs['aux_outputs'][0]['pred_boxes'], retain_graph=True)[0]
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    a = torch.autograd.grad(before['loss_tadl_aux_0'], outputs['aux_outputs'][0]['pred_logits'], retain_graph=True)[0]
    b = torch.autograd.grad(after['loss_tadl_aux_0'], outputs['aux_outputs'][0]['pred_logits'])[0]
    torch.testing.assert_close(b, a * .4)
    print('PASS: remote DOG exact parity, fallback wiring/gradients, unique routing, classification-only loss gradients')


if __name__ == '__main__':
    main()
