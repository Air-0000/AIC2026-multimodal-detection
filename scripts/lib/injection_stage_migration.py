"""Fail-closed state migration for HGNetv2MultiModal injection topology changes."""
import re

STAGE_FAMILIES = ('ir_proj', 'ir_align', 'occlusion_gate', 'coord_lift',
                  'ir_fuse', 'ir_res', 'bilateral_pool')


def stage_key(key, stages, prefix='backbone.'):
    match = re.match(r'^' + re.escape(prefix) + '(' + '|'.join(STAGE_FAMILIES) + r')\.(\d+)\.(.+)$', key)
    if match is None:
        return None
    family, index, suffix = match.groups()
    index = int(index)
    if index >= len(stages):
        raise ValueError(f'Unresolved module index: {key}, stages={stages}')
    return family, stages[index], suffix


def validate_stages(stages):
    values = list(stages)
    if not values or values != sorted(set(values)) or any(type(s) is not int or not 0 <= s <= 3 for s in values):
        raise ValueError(f'Expected sorted unique integer stage IDs in [0,3], got {values}')
    return values


def migrate_injection_stages(source, source_stages, target, target_stages, prefix='backbone.'):
    """Return a partial state and audit; only genuinely new stages may be missing.

    No source/target tensors are modified. All non-stage keys must match exactly
    in name, shape and dtype; removed stages are dropped explicitly. Unexpected
    source keys, missing shared keys and shape/dtype mismatches raise ValueError.
    Includes parameters and buffers of fusion/residual modules, not just IR proj.
    """
    source_stages, target_stages = validate_stages(source_stages), validate_stages(target_stages)
    new_stages = sorted(set(target_stages) - set(source_stages))
    mapped, mapping, removed = {}, {}, []
    for key, value in source.items():
        parsed = stage_key(key, source_stages, prefix)
        if parsed:
            family, stage, suffix = parsed
            if stage not in target_stages:
                removed.append(key)
                continue
            destination = f'{prefix}{family}.{target_stages.index(stage)}.{suffix}'
        else:
            destination = key
        if destination not in target:
            raise ValueError(f'Unresolved source key: {key} -> {destination}')
        expected = target[destination]
        if expected.shape != value.shape or expected.dtype != value.dtype:
            raise ValueError(f'Shape/dtype mismatch {key} -> {destination}: '
                             f'{tuple(value.shape)}/{value.dtype} vs {tuple(expected.shape)}/{expected.dtype}')
        if destination in mapped:
            raise ValueError(f'Duplicate destination: {destination}')
        mapped[destination], mapping[destination] = value, key
    new_keys, unresolved = [], []
    for key in target.keys() - mapped.keys():
        parsed = stage_key(key, target_stages, prefix)
        (new_keys if parsed and parsed[1] in new_stages else unresolved).append(key)
    if unresolved:
        raise ValueError(f'Unresolved target keys: {sorted(unresolved)}')
    report = {'source_stages': source_stages, 'target_stages': target_stages,
              'migrated_stages': sorted(set(source_stages) & set(target_stages)),
              'new_stages': new_stages, 'removed_stages': sorted(set(source_stages) - set(target_stages)),
              'new_initialization_keys': sorted(new_keys), 'removed_source_keys': sorted(removed),
              'unresolved_keys': [], 'key_mapping': mapping, 'mapped_key_count': len(mapped)}
    return mapped, report


def load_migrated_state(model, source, source_stages, target_stages):
    import torch
    target = model.state_dict()
    mapped, report = migrate_injection_stages(source, source_stages, target, target_stages)
    initial = {k: target[k].clone() for k in report['new_initialization_keys']}
    incompatible = model.load_state_dict(mapped, strict=False)
    missing = set(incompatible.missing_keys)
    expected_missing = set(report['new_initialization_keys'])
    # PyTorch may silently supply a zero num_batches_tracked for a missing BN
    # checkpoint entry. Accept only actual BatchNorm counters, and still verify
    # every fresh tensor below against its pre-load constructor value.
    implicit = expected_missing - missing
    valid_implicit = all(k.endswith('.num_batches_tracked') and
                         isinstance(model.get_submodule(k.rsplit('.', 1)[0]), torch.nn.modules.batchnorm._BatchNorm)
                         for k in implicit)
    if not missing <= expected_missing or not valid_implicit or incompatible.unexpected_keys:
        raise ValueError(f'Unexpected load result: {incompatible}')
    actual = model.state_dict()
    if not all(torch.equal(actual[k], v) for k, v in mapped.items()):
        raise ValueError('A migrated tensor was not preserved exactly')
    if not all(torch.equal(actual[k], v) for k, v in initial.items()):
        raise ValueError('A new-stage initial tensor was overwritten')
    report.update(strict_false_missing_keys=list(incompatible.missing_keys),
                  strict_false_unexpected_keys=list(incompatible.unexpected_keys),
                  pytorch_implicit_new_batchnorm_defaults=sorted(implicit),
                  mapped_tensors_exact=True, new_stage_initialization_preserved=True)
    return report
