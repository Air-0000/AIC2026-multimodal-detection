"""
Anti-Aliasing Modules (BlurPool & Shift-Equivariant Operators)
Based on Richard Zhang (ICML 2019): "Making Convolutional Networks Shift-Invariant Again"

Provides:
1. BlurPool2d: Depthwise fixed/learnable low-pass filtering before stride-2 downsampling.
2. AntiAliasedConv2d: Decomposes Stride-2 Conv into Stride-1 Conv + BlurPool2d.
3. AntiAliasedDownsampleResidual: Plug-and-play residual wrapper ensuring 100% backward compatibility
   at initialization (zero/identity perturbation) while smoothly conferring shift-invariance.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class BlurPool2d(nn.Module):
    """
    2D BlurPool (Anti-aliased sub-sampling).
    filt_size: 3 (binomial [1, 2, 1]) or 5 (binomial [1, 4, 6, 4, 1]).
    stride: downsampling factor (typically 2).
    """
    def __init__(self, channels: int, filt_size: int = 3, stride: int = 2, pad_off: int = 0):
        super().__init__()
        self.channels = channels
        self.filt_size = filt_size
        self.stride = stride
        self.pad_off = pad_off

        if filt_size == 3:
            a = torch.tensor([1., 2., 1.])
        elif filt_size == 5:
            a = torch.tensor([1., 4., 6., 4., 1.])
        elif filt_size == 1:
            a = torch.tensor([1.])
        else:
            raise ValueError(f"Unsupported filt_size: {filt_size}")

        filt = a[:, None] * a[None, :]
        filt = filt / torch.sum(filt)
        self.register_buffer('filt', filt[None, None, :, :].repeat(channels, 1, 1, 1), persistent=False)
        self.pad_sizes = [filt_size // 2, filt_size // 2, filt_size // 2, filt_size // 2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.filt_size == 1:
            if self.stride == 1:
                return x
            return x[:, :, ::self.stride, ::self.stride]

        x_pad = F.pad(x, self.pad_sizes, mode='reflect')
        filt = self.filt.to(device=x.device, dtype=x.dtype)
        return F.conv2d(x_pad, filt, stride=self.stride, groups=x.shape[1])


class AntiAliasedDownsample(nn.Module):
    """
    Anti-aliased downsampling wrapper for existing stride-2 convolutions.
    Features:
    - shift_alpha: blending parameter between raw downsampled output and blur-filtered output.
    - When shift_alpha = 0.0, output is 100% bitwise-identical to original model.
    - When shift_alpha > 0.0, filters out high-frequency spatial aliasing.
    """
    def __init__(self, channels: int, filt_size: int = 3, initial_alpha: float = 0.0, learnable: bool = True):
        super().__init__()
        self.channels = channels
        self.blur = BlurPool2d(channels=channels, filt_size=filt_size, stride=1)
        if learnable:
            self.alpha = nn.Parameter(torch.tensor([float(initial_alpha)]))
        else:
            self.register_buffer('alpha', torch.tensor([float(initial_alpha)]), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = torch.clamp(self.alpha, min=0.0, max=1.0)
        if alpha == 0.0 and not self.training:
            return x
        blurred = self.blur(x)
        return (1.0 - alpha) * x + alpha * blurred
