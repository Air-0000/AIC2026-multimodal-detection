"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

# torchvision v2 API shim: must run before any custom transform is instantiated
from . import _tv_compat  # noqa: F401


from ._transforms import (
    EmptyTransform,
    RandomPhotometricDistort,
    RandomPhotometricDistort5ch,
    RandomZoomOut,
    RandomIoUCrop,
    RandomHorizontalFlip,
    RandomAffine,
    Resize,
    PadToSize,
    SanitizeBoundingBoxes,
    RandomCrop,
    Normalize,
    ConvertBoxes,
    ConvertPILImage,
    RandomRGBDegrade5ch,
    RandomNight5ch,
    RandomModalityFailure,
    RandomTriStateIllumination,
)
from .container import Compose
from .mosaic import Mosaic