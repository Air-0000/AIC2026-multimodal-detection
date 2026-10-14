"""
Physical Inductive Bias & Modality Grounding Module for Sovereign V32.

Addresses the fundamental failure of statistical pattern-matching DETRs:
1. Thermodynamic Radiation Law:
   Active powered devices (UAV motors/battery, operating vehicles) must exhibit
   a positive local thermal contrast (Delta T = T_peak - T_ambient > 0).
   Passive ambient structures (painted 2D posters on cold concrete walls) have Delta T ~ 0.
2. Depth Surface Coplanarity Relief:
   Real floating 3D objects exhibit surface discontinuity / relief relative to the background.
   Flat 2D wall printings are coplanar with the background wall.
3. Zero-Init Stability:
   The physical modulation layer is strictly zero-initialized, ensuring that upon checkpoint loading,
   the model output is 100% identical to the base model, guaranteeing seamless warm-start with zero perturbation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class PhysicalInductiveGating(nn.Module):
    def __init__(
        self,
        num_classes: int = 12,
        ir_channel_idx: int = 3,       # index 3 in 6-channel input (Aligned IR)
        depth_channel_idx: int = 5,    # index 5 in 6-channel input (Depth P0)
        uav_class_idx: int = 10,       # class 10 is UAV
        hidden_dim: int = 64,
        tau_cold_ambient: float = 0.06, # ~15/255 thermal contrast threshold
        tau_hot_boost: float = 0.20,    # ~50/255 thermal contrast threshold for boost
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ir_channel_idx = ir_channel_idx
        self.depth_channel_idx = depth_channel_idx
        self.uav_class_idx = uav_class_idx
        self.tau_cold_ambient = tau_cold_ambient
        self.tau_hot_boost = tau_hot_boost

        # Physics Feature Dimension:
        # [Delta_T_mean, Delta_T_max, T_peak, T_ambient, Delta_Z_relief, Depth_std] -> 6 dims
        in_dim = 6

        # Lightweight Physical Modulator MLP
        self.modulator = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes)
        )

        # STRICT Zero-Initialization:
        # Final layer weights and bias are initialized to ZERO so that at step 0,
        # Delta_Logits is exactly 0.0, preserving 100% baseline behavior.
        nn.init.zeros_(self.modulator[-1].weight)
        nn.init.zeros_(self.modulator[-1].bias)

    @torch.no_grad()
    def _extract_rois(self, boxes_cxcywh: torch.Tensor, H: int, W: int, scale_factor: float = 1.0):
        """Convert normalized cxcywh [B, N, 4] to torchvision format [B*N, 5] (batch_idx, x1, y1, x2, y2)."""
        B, N, _ = boxes_cxcywh.shape
        cx = boxes_cxcywh[..., 0] * W
        cy = boxes_cxcywh[..., 1] * H
        bw = boxes_cxcywh[..., 2] * W * scale_factor
        bh = boxes_cxcywh[..., 3] * H * scale_factor

        x1 = torch.clamp(cx - bw / 2.0, 0, W - 1)
        y1 = torch.clamp(cy - bh / 2.0, 0, H - 1)
        x2 = torch.clamp(cx + bw / 2.0, 0, W - 1)
        y2 = torch.clamp(cy + bh / 2.0, 0, H - 1)

        batch_idx = torch.arange(B, device=boxes_cxcywh.device).unsqueeze(1).expand(B, N).reshape(-1, 1).float()
        rois = torch.cat([batch_idx, x1.reshape(-1, 1), y1.reshape(-1, 1), x2.reshape(-1, 1), y2.reshape(-1, 1)], dim=1)
        return rois

    def compute_physics_features(self, samples_6ch: torch.Tensor, pred_boxes_cxcywh: torch.Tensor):
        """
        Extract physical thermal and depth features for each candidate box.
        
        Args:
            samples_6ch: [B, 6, H, W] normalized input tensor
            pred_boxes_cxcywh: [B, N, 4] normalized box predictions [0, 1]
        
        Returns:
            physics_feats: [B, N, 6] physical descriptor tensor
            delta_t: [B, N] thermal contrast tensor
        """
        B, C, H, W = samples_6ch.shape
        N = pred_boxes_cxcywh.shape[1]

        # Extract IR and Depth channels
        # Note: samples_6ch is normalized with MEAN_6CH and STD_6CH.
        # For ch3 (IR) and ch5 (Depth), mean=0.5, std=0.5, so raw [0, 1] is approx (val * 0.5 + 0.5)
        ir_map = torch.clamp(samples_6ch[:, self.ir_channel_idx:self.ir_channel_idx + 1] * 0.5 + 0.5, 0.0, 1.0)
        depth_map = torch.clamp(samples_6ch[:, self.depth_channel_idx:self.depth_channel_idx + 1] * 0.5 + 0.5, 0.0, 1.0)

        # 1. Inner RoIs (target bounding box)
        rois_inner = self._extract_rois(pred_boxes_cxcywh, H, W, scale_factor=1.0)

        # 2. Outer RoIs (local ambient context, expanded 1.8x)
        rois_outer = self._extract_rois(pred_boxes_cxcywh, H, W, scale_factor=1.8)

        # RoIAlign (7x7 for inner, 11x11 for outer)
        ir_inner = torchvision.ops.roi_align(ir_map, rois_inner, output_size=(7, 7), spatial_scale=1.0) # [B*N, 1, 7, 7]
        ir_outer = torchvision.ops.roi_align(ir_map, rois_outer, output_size=(11, 11), spatial_scale=1.0) # [B*N, 1, 11, 11]

        depth_inner = torchvision.ops.roi_align(depth_map, rois_inner, output_size=(7, 7), spatial_scale=1.0)
        depth_outer = torchvision.ops.roi_align(depth_map, rois_outer, output_size=(11, 11), spatial_scale=1.0)

        # Flatten features for statistics
        ir_inner_flat = ir_inner.view(B * N, 49)
        ir_outer_flat = ir_outer.view(B * N, 121)
        depth_inner_flat = depth_inner.view(B * N, 49)
        depth_outer_flat = depth_outer.view(B * N, 121)

        # --- Thermal Statistics ---
        # Peak thermal emission in target (top 5 pixels out of 49)
        t_peak_topk, _ = torch.topk(ir_inner_flat, k=5, dim=1)
        t_peak = t_peak_topk.mean(dim=1, keepdim=True) # [B*N, 1]
        t_target_mean = ir_inner_flat.mean(dim=1, keepdim=True)
        t_ambient_mean = ir_outer_flat.mean(dim=1, keepdim=True)

        delta_t_max = t_peak - t_ambient_mean
        delta_t_mean = t_target_mean - t_ambient_mean

        # --- Depth Relief Statistics ---
        depth_target_mean = depth_inner_flat.mean(dim=1, keepdim=True)
        depth_ambient_mean = depth_outer_flat.mean(dim=1, keepdim=True)
        delta_z_relief = torch.abs(depth_target_mean - depth_ambient_mean)
        depth_std = depth_inner_flat.std(dim=1, keepdim=True)

        # Assemble physical descriptor vector: [B*N, 6]
        phys_vec = torch.cat([
            delta_t_mean,
            delta_t_max,
            t_peak,
            t_ambient_mean,
            delta_z_relief,
            depth_std
        ], dim=1)

        physics_feats = phys_vec.view(B, N, 6)
        delta_t = delta_t_max.view(B, N)
        return physics_feats, delta_t

    def forward(self, pred_logits: torch.Tensor, pred_boxes: torch.Tensor, samples_6ch: torch.Tensor):
        """
        Modulate prediction logits with physical inductive bias.

        Args:
            pred_logits: [B, N, num_classes] raw classification logits from Transformer Decoder
            pred_boxes: [B, N, 4] normalized predicted boxes (cx, cy, w, h)
            samples_6ch: [B, 6, H, W] raw multi-channel input image
        
        Returns:
            modulated_logits: [B, N, num_classes] physically grounded classification logits
            aux_dict: auxiliary information for physics loss computation
        """
        physics_feats, delta_t = self.compute_physics_features(samples_6ch, pred_boxes.detach())

        # Forward through lightweight modulator MLP
        delta_logits = self.modulator(physics_feats) # [B, N, num_classes]

        # Apply modulation
        modulated_logits = pred_logits + delta_logits

        aux_dict = {
            'delta_t': delta_t,
            'physics_feats': physics_feats,
            'delta_logits': delta_logits
        }
        return modulated_logits, aux_dict


class PhysicalConsistencyLoss(nn.Module):
    """
    Penalizes physical violations:
    1. Cold Thermal Violation:
       When a query predicts high UAV probability (active power/motor device) on a cold background (Delta T < tau_cold),
       impose a strong quadratic penalty.
    """
    def __init__(self, tau_cold: float = 0.05, loss_weight: float = 2.0, uav_class_idx: int = 10):
        super().__init__()
        self.tau_cold = tau_cold
        self.loss_weight = loss_weight
        self.uav_class_idx = uav_class_idx

    def forward(self, modulated_logits: torch.Tensor, delta_t: torch.Tensor):
        """
        Args:
            modulated_logits: [B, N, num_classes]
            delta_t: [B, N] thermal contrast
        """
        uav_probs = torch.sigmoid(modulated_logits[..., self.uav_class_idx]) # [B, N]

        # Thermal violation: UAV predicted with significant probability but delta_t is below tau_cold
        cold_violation = F.relu(self.tau_cold - delta_t) # > 0 when delta_t < tau_cold
        
        # Penalty: uav_probs * cold_violation^2
        loss_thermal_contra = (uav_probs * (cold_violation ** 2)).mean()
        return self.loss_weight * loss_thermal_contra
