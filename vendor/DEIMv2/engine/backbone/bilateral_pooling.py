"""
Disparity-Bilateral Anti-Aliased Pooling Module (DisparityBilateralPooling2d).

Replaces standard stride-2 pooling or convolution by conditioning downsampling weights
on disparity differences:
  - Inside a contiguous 3D object, features are pooled normally;
  - Across large depth discontinuities (e.g. car boundary vs background trees),
    pooling weights drop to 0, strictly preventing cross-boundary feature bleeding.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DisparityBilateralPooling2d(nn.Module):
    def __init__(self, in_channels, stride=2, kernel_size=3, padding=1, sigma_disp=0.05):
        super().__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = padding
        self.sigma_disp = sigma_disp

        # Spatial Gaussian base kernel
        self.register_buffer("spatial_kernel", self._get_spatial_kernel(kernel_size))

    def _get_spatial_kernel(self, k):
        coords = torch.arange(k, dtype=torch.float32) - (k - 1) / 2.0
        x, y = torch.meshgrid(coords, coords, indexing="xy")
        dist_sq = x**2 + y**2
        kernel = torch.exp(-dist_sq / (2.0 * (k / 3.0)**2))
        return kernel / kernel.sum()

    def forward(self, x, disparity):
        """
        Args:
          x: (B, C, H, W) feature map
          disparity: (B, 1, H, W) normalized inverse depth in m^-1 (or relative disparity)
        Returns:
          out: (B, C, H // stride, W // stride) anti-aliased pooled feature map
        """
        B, C, H, W = x.shape
        k = self.kernel_size
        p = self.padding
        s = self.stride

        # Unfold disparity into local patches
        d_unfold = F.unfold(disparity, kernel_size=k, padding=p, stride=s) # (B, k*k, L)
        # Center pixel disparity
        center_idx = (k * k) // 2
        d_center = d_unfold[:, center_idx:center_idx+1, :] # (B, 1, L)
        
        # Bilateral disparity weight: exp(-|d_patch - d_center| / sigma)
        d_diff = torch.abs(d_unfold - d_center)
        bilateral_w = torch.exp(-torch.clamp(d_diff / (self.sigma_disp + 1e-4), max=20.0)) # (B, k*k, L)
        
        # Multiply with spatial Gaussian
        spatial_w = self.spatial_kernel.view(1, k * k, 1).to(device=d_diff.device, dtype=d_diff.dtype)
        total_w = bilateral_w * spatial_w
        denom = torch.sum(total_w, dim=1, keepdim=True)
        # Safe fallback: if all weights underflow or sum to zero, fall back to center pixel
        total_w = torch.where(denom > 1e-6, total_w / (denom + 1e-6), torch.zeros_like(total_w))
        total_w[:, center_idx:center_idx+1, :] = torch.where(denom <= 1e-6, torch.ones_like(d_center), total_w[:, center_idx:center_idx+1, :])

        # Unfold feature map and apply weights per channel
        # Use grouped convolution / unfolded bmm
        out_h = (H + 2 * p - k) // s + 1
        out_w = (W + 2 * p - k) // s + 1

        # Reshape for efficient batched execution
        x_unfold = F.unfold(x, kernel_size=k, padding=p, stride=s) # (B, C * k * k, L)
        x_unfold = x_unfold.view(B, C, k * k, -1) # (B, C, k*k, L)
        total_w = total_w.unsqueeze(1) # (B, 1, k*k, L)

        out = torch.sum(x_unfold * total_w, dim=2) # (B, C, L)
        out = out.view(B, C, out_h, out_w)
        return out
