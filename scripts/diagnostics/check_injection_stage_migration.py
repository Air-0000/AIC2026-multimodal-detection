#!/usr/bin/env python3
"""CPU contracts for stage identity remapping, fusion/residual coverage and errors."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/lib'))


def main():
    import torch
    from injection_stage_migration import STAGE_FAMILIES, migrate_injection_stages, load_migrated_state
    def state(stages):
        result = {'backbone.rgb_net.shared': torch.tensor([99.])}
        for index, stage in enumerate(stages):
            for family in STAGE_FAMILIES:
                result[f'backbone.{family}.{index}.weight'] = torch.full((stage, 1), float(stage))
        return result
    source = state([2, 3])
    for stages in ([2, 3], [1, 2], [1, 2, 3]):
        target = state(stages)
        mapped, report = migrate_injection_stages(source, [2, 3], target, stages)
        for stage in set(stages) & {2, 3}:
            for family in STAGE_FAMILIES:
                destination = f'backbone.{family}.{stages.index(stage)}.weight'
                assert torch.equal(mapped[destination], source[f'backbone.{family}.{[2,3].index(stage)}.weight'])
        assert len(report['new_initialization_keys']) == (len(STAGE_FAMILIES) if 1 in stages else 0)
        assert not report['unresolved_keys']
    def rejected(src, src_stages, dst, dst_stages):
        try:
            migrate_injection_stages(src, src_stages, dst, dst_stages)
        except ValueError:
            return
        raise AssertionError('Invalid migration accepted')
    bad = state([1, 2]);bad['backbone.ir_res.1.weight'] = torch.zeros(3, 1)
    rejected(source, [2, 3], bad, [1, 2])
    bad = dict(source);bad['backbone.ir_align.2.weight'] = torch.zeros(2, 1)
    rejected(bad, [2, 3], state([1, 2]), [1, 2])
    bad = state([1, 2]);bad['backbone.extra_shared'] = torch.zeros(1)
    rejected(source, [2, 3], bad, [1, 2])
    rejected(source, [2, 2], state([1, 2]), [1, 2])
    def batchnorm_model(stages):
        model = torch.nn.Module()
        model.backbone = torch.nn.Module()
        model.backbone.ir_align = torch.nn.ModuleList([torch.nn.BatchNorm1d(s) for s in stages])
        return model
    src, dst = batchnorm_model([2, 3]), batchnorm_model([1, 2])
    report = load_migrated_state(dst, src.state_dict(), [2, 3], [1, 2])
    assert report['new_stage_initialization_preserved']
    assert 'backbone.ir_align.0.num_batches_tracked' in report['new_initialization_keys']
    print('PASS: three topology mappings; all seven families; shape/index/shared-key/duplicate-stage rejection')


if __name__ == '__main__':
    main()
