"""Finite-value checks for controlled runs; never repair tensors in place."""
import torch


def require_finite(named_tensors, context, allow_anchor_infinity=False):
    checks = []
    def walk(name, value):
        if isinstance(value, dict):
            for key, item in value.items():
                yield from walk(f'{name}.{key}', item)
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                yield from walk(f'{name}.{index}', item)
        else:
            yield name, value
    for name, value in (pair for key, value in named_tensors for pair in walk(key, value)):
        if not isinstance(value, torch.Tensor):
            continue
        # Invalid border anchors deliberately carry +inf until masked by the
        # decoder. NaN in an anchor is still a failure.
        valid = (~torch.isnan(value)).all() if allow_anchor_infinity and name.endswith('anchors') else torch.isfinite(value).all()
        checks.append((name, valid))
    by_device = {}
    for _, valid in checks:
        by_device.setdefault(valid.device, []).append(valid)
    if any(not bool(torch.stack(values).all()) for values in by_device.values()):
        bad = [name for name, valid in checks if not bool(valid)]
        raise FloatingPointError(f'{context}: nonfinite tensors {bad}; stopped without sanitizing or retrying')
