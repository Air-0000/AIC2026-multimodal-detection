"""
DINOv3 Online Frozen Teacher and Distillation Module
===================================================
Sovereign V29: Micro-Surgery Semantic Distillation from Meta DINOv3-ViT-Large
Anchored to Sovereign V23 Gold Baseline (88.83% mAP).

Features:
  - Frozen DINOv3-ViT-Large (300M, 1024-dim, torch.no_grad())
  - Dynamic 2x bilinear downsampling for teacher (576x1024, only 2,304 tokens, ~2GB VRAM)
  - Directional Cosine Similarity Loss on dense patch tokens
  - Inter-channel Gram Matrix Correlation Loss for topological relation anchoring
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .dinov3.vision_transformer import DinoVisionTransformer


class DINOv3TeacherDistiller(nn.Module):
    def __init__(
        self,
        model_path="ckpts/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        student_dim=384,        # Student Stage 2 (P4) feature channels in HGNetv2
        teacher_dim=1024,       # DINOv3-ViT-Large hidden dim
        loss_cos_weight=1.0,
        loss_gram_weight=0.5,
        downsample_scale=0.5,   # Downsample 1152x2048 to 576x1024 for teacher
    ):
        super().__init__()
        self.model_path = model_path
        self.student_dim = student_dim
        self.teacher_dim = teacher_dim
        self.loss_cos_weight = loss_cos_weight
        self.loss_gram_weight = loss_gram_weight
        self.downsample_scale = downsample_scale

        # Projection Head: maps student P4 features to teacher dimension
        self.proj_head = nn.Sequential(
            nn.Conv2d(student_dim, teacher_dim, kernel_size=1, bias=False),
            nn.GroupNorm(32, teacher_dim),
            nn.GELU(),
            nn.Conv2d(teacher_dim, teacher_dim, kernel_size=1, bias=True)
        )
        nn.init.kaiming_normal_(self.proj_head[0].weight)
        nn.init.kaiming_normal_(self.proj_head[3].weight)
        nn.init.zeros_(self.proj_head[3].bias)

        self.teacher = None
        self._device = None

    def _find_checkpoint(self):
        candidates = [
            self.model_path,
            os.path.join(os.path.dirname(__file__), "../../ckpts/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"),
            "/root/autodl-tmp/AIC/vendor/DEIMv2/ckpts/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
            "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/ckpts/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        ]
        for c in candidates:
            if c and os.path.exists(c) and os.path.isfile(c) and os.path.getsize(c) > 1_000_000_000:
                return os.path.abspath(c)
        return None

    def _ensure_teacher_loaded(self, device):
        if self.teacher is None or self._device != device:
            ckpt_path = self._find_checkpoint()
            print(f"[DINOv3Teacher] Initializing native DinoVisionTransformer (dinov3_vitl16)...")
            self.teacher = DinoVisionTransformer(name="dinov3_vitl16")
            if ckpt_path:
                print(f"[DINOv3Teacher] Loading official weights from {ckpt_path} onto {device}...")
                state_dict = torch.load(ckpt_path, map_location="cpu")
                res = self.teacher.load_state_dict(state_dict, strict=True)
                print(f"[DINOv3Teacher] Weights loaded successfully (strict=True): {res}")
            else:
                print(f"[DINOv3Teacher] WARNING: Checkpoint not found at {self.model_path}, using uninitialized weights!")

            self.teacher.to(device)
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad = False
            self._device = device
            print("[DINOv3Teacher] Teacher successfully frozen in eval mode.")

    @torch.no_grad()
    def get_teacher_features(self, rgb):
        """
        Extract dense spatial patch tokens from DINOv3.
        rgb: [B, 3, H, W] in ImageNet normalized space.
        """
        B, C, H, W = rgb.shape
        self._ensure_teacher_loaded(rgb.device)

        # Bilinear downsample for teacher: 1152x2048 -> 576x1024
        if self.downsample_scale != 1.0:
            target_h = int(H * self.downsample_scale)
            target_w = int(W * self.downsample_scale)
            # Ensure divisible by 16 (patch size)
            target_h = (target_h // 16) * 16
            target_w = (target_w // 16) * 16
            rgb_t = F.interpolate(rgb, size=(target_h, target_w), mode='bilinear', align_corners=False)
        else:
            target_h, target_w = (H // 16) * 16, (W // 16) * 16
            rgb_t = F.interpolate(rgb, size=(target_h, target_w), mode='bilinear', align_corners=False)

        # Forward through frozen DINOv3 ViT in FP16
        with torch.autocast(device_type=rgb.device.type, dtype=torch.float16):
            outputs = self.teacher.forward_features(rgb_t)
            patch_tokens = outputs["x_norm_patchtokens"]  # [B, N_patches, 1024]
            grid_h = target_h // 16
            grid_w = target_w // 16
            feat_map = patch_tokens.permute(0, 2, 1).reshape(B, self.teacher_dim, grid_h, grid_w)

        return feat_map.float()  # [B, 1024, grid_h, grid_w]

    def compute_distill_loss(self, rgb, student_feats):
        """
        rgb: [B, 3, H, W] (first 3 channels of 6-ch input)
        student_feats: list of multi-scale features from student backbone or encoder.
                       Specifically student_feats[2] is P4 (stride 16, C=384).
        """
        if isinstance(student_feats, (list, tuple)):
            # P4 is stage 2
            student_p4 = student_feats[2] if len(student_feats) > 2 else student_feats[-1]
        else:
            student_p4 = student_feats

        # 1. Project student P4 to teacher dimension
        proj_s = self.proj_head(student_p4) # [B, 1024, H_s, W_s]

        # 2. Extract frozen teacher features
        with torch.no_grad():
            feat_t = self.get_teacher_features(rgb) # [B, 1024, H_t, W_t]

        # Align spatial dimensions via bilinear interpolation
        if proj_s.shape[-2:] != feat_t.shape[-2:]:
            proj_s = F.interpolate(proj_s, size=feat_t.shape[-2:], mode='bilinear', align_corners=False)

        # 3. Directional Cosine Similarity Loss
        norm_s = F.normalize(proj_s, p=2, dim=1)
        norm_t = F.normalize(feat_t, p=2, dim=1)
        cos_sim = (norm_s * norm_t).sum(dim=1) # [B, H, W]
        loss_cos = (1.0 - cos_sim).mean()

        # 4. Gram Matrix Inter-channel Topological Correlation Loss
        # Gram matrix: G = F * F^T / (H * W)
        B, C, H_g, W_g = norm_s.shape
        flat_s = norm_s.view(B, C, H_g * W_g)
        flat_t = norm_t.view(B, C, H_g * W_g)
        gram_s = torch.bmm(flat_s, flat_s.transpose(1, 2)) / (H_g * W_g) # [B, C, C]
        gram_t = torch.bmm(flat_t, flat_t.transpose(1, 2)) / (H_g * W_g) # [B, C, C]
        loss_gram = F.mse_loss(gram_s, gram_t)

        total_loss = self.loss_cos_weight * loss_cos + self.loss_gram_weight * loss_gram
        return total_loss
