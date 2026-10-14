"""
torchvision v2 API compatibility shim.

DEIMv2 was written against torchvision ~0.20 where custom transforms override
``Transform._transform()`` / ``_get_params()``. torchvision >=0.22 renamed these
hooks to ``transform()`` / ``make_params()`` and the base ``Transform.forward``
now calls ``transform()`` directly, so custom classes only defining the old
``_transform`` hook crash with ``NotImplementedError``.

We patch the two base hooks so that:
  * if a subclass overrides the NEW ``transform`` hook -> use it (untouched);
  * otherwise fall back to the OLD ``_transform`` hook (DEIMv2's custom classes);
and likewise for ``make_params`` / ``_get_params``.
"""
from __future__ import annotations

import torchvision.transforms.v2 as _T


def _patch() -> None:
    if not hasattr(_T.Transform, "transform"):
        return
    _orig_transform = _T.Transform.transform
    _orig_make_params = getattr(_T.Transform, "make_params", None)

    def _compat_transform(self, inpt, params):
        # Prefer the OLD ``_transform`` hook if the subclass defines one (DEIMv2
        # custom transforms). torchvision's own transforms implement the NEW
        # ``transform`` hook, so for them we take the default path.
        if hasattr(self, "_transform"):
            return self._transform(inpt, params)
        return _orig_transform(self, inpt, params)

    def _compat_make_params(self, flat_inputs):
        if hasattr(self, "_get_params"):
            return self._get_params(flat_inputs)
        return _orig_make_params(self, flat_inputs)

    _T.Transform.transform = _compat_transform
    _T.Transform.make_params = _compat_make_params


_patch()
