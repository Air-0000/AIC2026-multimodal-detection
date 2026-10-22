import torch
import torch.nn as nn
from ..core import register


__all__ = ['DEIM', ]


class SceneGeometryContextGate(nn.Module):
    """3D Scene-Geometry Continuous Depth Context Gate (SGMG).
    Extracts continuous statistical moments from 16-bit metric depth map
    (bottom ground/water plane mean, variance, percentiles, gradient)
    to softly modulate class logits (e.g. smoothly suppress boat false alarms on dry land).
    Zero-initialized to guarantee 100.00% exact numerical identity with pretrained weights at Step 0.
    """
    def __init__(self, num_classes=12, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes)
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, depth):
        # depth: [B, 1, H, W], normalized with mean=0.5, std=0.5 -> [-1, 1]
        # Reconstruct [0, 1] raw metric depth
        depth_raw = torch.clamp(depth * 0.5 + 0.5, 0.0, 1.0)
        B, _, H, W = depth_raw.shape
        bottom = depth_raw[:, :, int(H * 0.5):, :]
        # Area pool to (40, 80) for lightning-fast sub-millisecond quantile statistics
        bottom_down = torch.nn.functional.interpolate(bottom, size=(40, 80), mode='area')[:, 0]
        valid = bottom_down > 0.015

        stats = []
        for b in range(B):
            v_pix = bottom_down[b][valid[b]]
            if len(v_pix) > 10:
                mean = v_pix.mean()
                std = torch.sqrt(torch.var(v_pix, unbiased=False) + 1e-6)
                q10 = torch.quantile(v_pix, 0.10)
                q90 = torch.quantile(v_pix, 0.90)
                ratio = std / (mean + 1e-4)
                stats.append(torch.stack([mean, std, q10, q90, ratio]))
            else:
                stats.append(torch.zeros(5, device=depth.device))
        x = torch.stack(stats, dim=0)
        return self.mlp(x) # [B, num_classes]


class PlanarRigidityContextGate(nn.Module):
    """Option 2: Planar Rigidity Context Gate (架构内生化刚体平面度门控).
    Measures 3D local physical planarity (least-squares planar fit residual) on depth RoIs.
    Differentiates 2D rigid planar sheets (e.g. signboards, placards) from 3D volumetric
    biological entities (animal).
    When a candidate query exhibits strict 2D planar rigidity elevated in near-field ground:
      - Boosts Class 4 (sign) logits
      - Suppresses Class 2 (animal) false positives
    Zero-initialized / parameterized with physical priors.
    """
    def __init__(self,
                 num_classes=12,
                 planarity_thresh=0.015,
                 near_depth_max=0.15,
                 boost_logit=3.5,
                 suppress_logit=3.5):
        super().__init__()
        self.num_classes = num_classes
        self.register_buffer('planarity_thresh', torch.tensor(float(planarity_thresh)))
        self.register_buffer('near_depth_max', torch.tensor(float(near_depth_max)))
        self.register_buffer('boost_logit', torch.tensor(float(boost_logit)))
        self.register_buffer('suppress_logit', torch.tensor(float(suppress_logit)))

        K = 8
        y, x = torch.meshgrid(torch.linspace(-1, 1, K), torch.linspace(-1, 1, K), indexing='ij')
        A = torch.stack([x.flatten(), y.flatten(), torch.ones_like(x.flatten())], dim=1)
        self.register_buffer('A', A)
        self.K = K

    def forward(self, depth, pred_boxes, pred_logits):
        if depth is None:
            return pred_logits
        import torchvision.ops as ops

        B, Q, num_cls = pred_logits.shape
        depth_raw = torch.clamp(depth * 0.5 + 0.5, 0.0, 1.0)
        Hd, Wd = depth_raw.shape[-2:]

        out_logits = pred_logits.clone()
        scores = torch.sigmoid(pred_logits)

        for b in range(B):
            animal_scores = scores[b, :, 2]
            # Genuine flocks in AIC (swans, flamingos, ducks) have >= 4 animals
            has_animal_flock = ((animal_scores >= 0.70).sum() >= 4)

            cand_mask = (scores[b, :, 2] >= 0.15) & (pred_boxes[b, :, 2] > 0.05) & (pred_boxes[b, :, 3] > 0.05)
            if not cand_mask.any():
                continue

            cand_indices = torch.where(cand_mask)[0]
            boxes = pred_boxes[b, cand_indices]
            cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

            x1 = (cx - bw/2).clamp(0, 1) * (Wd - 1)
            y1 = (cy - bh/2).clamp(0, 1) * (Hd - 1)
            x2 = (cx + bw/2).clamp(0, 1) * (Wd - 1)
            y2 = (cy + bh/2).clamp(0, 1) * (Hd - 1)

            batch_col = torch.zeros(len(cand_indices), 1, device=depth.device)
            rois = torch.cat([batch_col, x1.unsqueeze(1), y1.unsqueeze(1), x2.unsqueeze(1), y2.unsqueeze(1)], dim=1)

            roi_d = ops.roi_align(depth_raw[b:b+1], rois, output_size=(self.K, self.K), spatial_scale=1.0)
            roi_flat = roi_d.view(len(cand_indices), -1)

            c_idx = (self.K // 2) * self.K + (self.K // 2)
            d_center = roi_flat[:, c_idx]

            for m, q_idx in enumerate(cand_indices):
                d_c = d_center[m].item()
                if not (0.06 < d_c < self.near_depth_max.item()):
                    continue

                w_m = bw[m].item()
                h_m = bh[m].item()
                ar_m = w_m / max(h_m, 1e-5)
                if not (0.40 <= ar_m <= 1.20):
                    continue

                cy_m = cy[m].item()
                if cy_m > 0.55:
                    continue

                y_ground = min(1.0, cy_m + h_m / 2.0 + 0.02)
                cx_m = cx[m].item()
                x_d_int = int(cx_m * (Wd - 1))
                y_d_int = int(y_ground * (Hd - 1))
                d_ground = depth_raw[b, 0, y_d_int, x_d_int].item()

                relief = abs(d_c - d_ground)
                # Elevated off ground by mounting pole: relief between 0.020 and 0.080 (~40cm to 1.6m)
                is_pole_elevated = (0.020 <= relief <= 0.080)

                patch_m = roi_flat[m]
                fg_mask = torch.abs(patch_m - d_c) < 0.03
                if fg_mask.sum() >= 10:
                    xs_fg = self.A[fg_mask, 0]
                    ys_fg = self.A[fg_mask, 1]
                    zs_fg = patch_m[fg_mask]
                    A_sub = torch.stack([xs_fg, ys_fg, torch.ones_like(xs_fg)], dim=1)
                    pinv_sub = torch.linalg.pinv(A_sub)
                    params = pinv_sub @ zs_fg
                    pred_z = A_sub @ params
                    rmse = torch.sqrt(torch.mean((zs_fg - pred_z)**2)).item()
                else:
                    rmse = 999.0

                is_planar = (rmse < self.planarity_thresh.item())

                if is_planar and is_pole_elevated and not has_animal_flock:
                    out_logits[b, q_idx, 4] += self.boost_logit
                    out_logits[b, q_idx, 2] -= self.suppress_logit

        return out_logits


class P2CascadeZoomGate(nn.Module):
    """Option 2: Feature-Level Cascade RoI Zoom Gate (特征层级级联内化).
    Operates on Stage 0 (P2, stride 4, C=128, 288x512) feature map.
    Extracts 7x7 RoIAlign patches on candidate queries.
    Measures spatial center-to-surround contrast and high-frequency saliency.
    Calibrates ambiguous queries (e.g. scores in [score_min, score_max]) with positive
    saliency boost to recover weak/small targets without hallucinating noise.
    """
    def __init__(self,
                 roi_size=7,
                 score_min=0.15,
                 score_max=0.45,
                 max_area=0.025,
                 saliency_thresh=0.35,
                 alpha_boost=0.45,
                 beta_suppress=0.0,
                 num_classes=12):
        super().__init__()
        self.roi_size = roi_size
        self.register_buffer('score_min', torch.tensor(float(score_min)))
        self.register_buffer('score_max', torch.tensor(float(score_max)))
        self.register_buffer('max_area', torch.tensor(float(max_area)))
        self.register_buffer('saliency_thresh', torch.tensor(float(saliency_thresh)))
        self.register_buffer('alpha_boost', torch.tensor(float(alpha_boost)))
        self.register_buffer('beta_suppress', torch.tensor(float(beta_suppress)))
        self.num_classes = num_classes

    def compute_saliency(self, roi_feats):
        N, C, H, W = roi_feats.shape
        energy = torch.norm(roi_feats, dim=1) # [N, 7, 7]
        center = energy[:, 2:5, 2:5].mean(dim=(-2, -1)) # [N]
        total_sum = energy.sum(dim=(-2, -1)) # [N]
        center_sum = energy[:, 2:5, 2:5].sum(dim=(-2, -1)) # [N]
        surround = (total_sum - center_sum) / 40.0 # [N]
        std = energy.std(dim=(-2, -1)) + 1e-5 # [N]
        contrast = (center - surround) / std # [N]
        return contrast

    def forward(self, p2_feat, pred_boxes, pred_logits):
        from torchvision.ops import roi_align
        B, Q, num_cls = pred_logits.shape
        H_p2, W_p2 = p2_feat.shape[-2:]

        scores = torch.sigmoid(pred_logits)
        max_scores, best_cls = scores.max(dim=-1)
        box_area = pred_boxes[..., 2] * pred_boxes[..., 3]

        target_mask = (max_scores >= self.score_min) & (max_scores <= self.score_max) & (box_area <= self.max_area)
        if not target_mask.any():
            return pred_logits

        cx = pred_boxes[..., 0] * W_p2
        cy = pred_boxes[..., 1] * H_p2
        w = pred_boxes[..., 2] * W_p2
        h = pred_boxes[..., 3] * H_p2

        x1 = torch.clamp(cx - w / 2, min=0.0, max=W_p2 - 1.0)
        y1 = torch.clamp(cy - h / 2, min=0.0, max=H_p2 - 1.0)
        x2 = torch.clamp(cx + w / 2, min=1.0, max=W_p2)
        y2 = torch.clamp(cy + h / 2, min=1.0, max=H_p2)

        batch_idx = torch.arange(B, device=p2_feat.device).unsqueeze(1).repeat(1, Q)
        rois = torch.stack([batch_idx[target_mask].float(), x1[target_mask], y1[target_mask], x2[target_mask], y2[target_mask]], dim=1)

        roi_feats = roi_align(
            p2_feat,
            rois,
            output_size=(self.roi_size, self.roi_size),
            spatial_scale=1.0,
            sampling_ratio=2,
            aligned=True
        )

        saliency = self.compute_saliency(roi_feats)
        delta_logits = torch.zeros_like(pred_logits)
        best_cls_sel = best_cls[target_mask]

        diff = saliency - self.saliency_thresh
        boost = torch.where(
            diff > 0,
            self.alpha_boost * torch.clamp(diff, max=1.2),
            -self.beta_suppress * torch.clamp(-diff, max=1.5)
        )

        target_indices = torch.nonzero(target_mask, as_tuple=True)
        delta_logits[target_indices[0], target_indices[1], best_cls_sel] += boost
        return pred_logits + delta_logits
class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class NAFBlock(nn.Module):
    """Nonlinear Activation Free Block for Multi-Modal Quality Restoration (CVPR 2022)."""
    def __init__(self, in_channels=6, mid_channels=32):
        super().__init__()
        self.norm1 = LayerNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.dwconv = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, groups=mid_channels)
        self.sg_channels = mid_channels // 2
        self.sca_pool = nn.AdaptiveAvgPool2d(1)
        self.sca_conv = nn.Conv2d(self.sg_channels, self.sg_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.sg_channels, in_channels, kernel_size=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        h = self.norm1(x)
        h = self.conv1(h)
        h = self.dwconv(h)
        h1, h2 = torch.split(h, self.sg_channels, dim=1)
        h = h1 * h2
        attn = self.sca_conv(self.sca_pool(h))
        h = h * torch.sigmoid(attn)
        h = self.conv2(h)
        return x + h


class FrontendNAFRestoration(nn.Module):
    """Stacked 2x NAFBlock Multi-Modal Denoising and Deblurring Head (0.002M params)."""
    def __init__(self, in_channels=6, mid_channels=32, num_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([NAFBlock(in_channels, mid_channels) for _ in range(num_blocks)])

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


class WaveletHighFreqStarInjection(nn.Module):
    """
    P2 Feature-Level Wavelet High-Frequency StarOperation Injection (1.25x Super-Resolution).
    Separates P2 into LL and [LH, HL, HH], modulates and magnifies details by 1.25x via StarOperation,
    and applies high-frequency residual sharpening to Stage 0 features.
    """
    def __init__(self, channels=128, scale=1.25, blend_weight=0.15):
        super().__init__()
        from .sovereign_v13_modules import StarOperationConv2d
        self.channels = channels
        self.scale = scale
        self.blend_weight = blend_weight
        self.high_channels = channels * 3 # LH, HL, HH
        self.star_conv = StarOperationConv2d(dim=self.high_channels, mlp_ratio=1.5)

    def dwt_haar(self, x):
        x01 = x[:, :, 0::2, :] / 2.0
        x02 = x[:, :, 1::2, :] / 2.0
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        
        ll = x1 + x2 + x3 + x4
        lh = -x1 - x3 + x2 + x4
        hl = -x1 + x3 - x2 + x4
        hh = x1 - x3 - x2 + x4
        return ll, lh, hl, hh

    def idwt_haar(self, ll, lh, hl, hh):
        B, C, H, W = ll.shape
        x1 = (ll - lh - hl + hh) / 2.0
        x2 = (ll + lh - hl - hh) / 2.0
        x3 = (ll - lh + hl - hh) / 2.0
        x4 = (ll + lh + hl + hh) / 2.0

        out = torch.zeros(B, C, H * 2, W * 2, device=ll.device, dtype=ll.dtype)
        out[:, :, 0::2, 0::2] = x1
        out[:, :, 1::2, 0::2] = x2
        out[:, :, 0::2, 1::2] = x3
        out[:, :, 1::2, 1::2] = x4
        return out

    def forward(self, p2):
        ll, lh, hl, hh = self.dwt_haar(p2)
        high = torch.cat([lh, hl, hh], dim=1)
        high_mod = self.star_conv(high) * self.scale
        lh_m, hl_m, hh_m = torch.split(high_mod, self.channels, dim=1)
        p2_enhanced = self.idwt_haar(ll, lh_m, hl_m, hh_m)
        return p2 + self.blend_weight * (p2_enhanced - p2)


@register()
class DEIM(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, \
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
        use_scene_gate: bool = False,
        use_internal_sahi: bool = False,
        use_internal_tta: bool = False,
        use_p2_cascade_zoom: bool = False,
        p2_zoom_alpha: float = 0.45,
        p2_zoom_thresh: float = 0.35,
        p2_zoom_max_area: float = 0.025,
        p2_zoom_score_min: float = 0.15,
        p2_zoom_score_max: float = 0.45,
        use_frontend_naf: bool = False,
        naf_mid_channels: int = 32,
        naf_num_blocks: int = 2,
        use_p2_wavelet_sr: bool = False,
        p2_wavelet_scale: float = 1.25,
        p2_wavelet_blend: float = 0.15,
        use_planar_rigidity_gate: bool = False,
        planar_gate_boost: float = 3.5,
        planar_gate_suppress: float = 3.5,
        use_dinov3_distill: bool = False,
        dinov3_teacher_path: str = "/home/constant/CodeBuddy/Dev/AIC/weights/dinov3_large",
        distill_cos_weight: float = 1.0,
        distill_gram_weight: float = 0.5,
        distill_downsample_scale: float = 1.0,
        require_teacher_weights: bool = False,
        teacher_weights_sha256: str = None,
        use_co_aux: bool = False,
        co_aux_loss_weight_cls: float = 1.0,
        co_aux_loss_weight_reg: float = 2.0,
        use_physics_grounding: bool = False,
        physics_loss_weight: float = 2.0,
        num_classes: int = 12,
        num_queries: int = 300,
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.use_scene_gate = use_scene_gate
        self.use_internal_sahi = use_internal_sahi
        self.use_internal_tta = use_internal_tta
        self.use_p2_cascade_zoom = use_p2_cascade_zoom
        self.use_frontend_naf = use_frontend_naf
        self.use_p2_wavelet_sr = use_p2_wavelet_sr
        self.use_planar_rigidity_gate = use_planar_rigidity_gate
        self.use_dinov3_distill = use_dinov3_distill
        self.use_co_aux = use_co_aux
        self.use_physics_grounding = use_physics_grounding
        self.num_classes = num_classes
        self.num_queries = num_queries
        if use_dinov3_distill:
            from ..backbone.dinov3_teacher import DINOv3TeacherDistiller
            self.dinov3_distiller = DINOv3TeacherDistiller(
                model_path=dinov3_teacher_path,
                student_dim=384,
                teacher_dim=1024,
                loss_cos_weight=distill_cos_weight,
                loss_gram_weight=distill_gram_weight,
                downsample_scale=distill_downsample_scale,
                require_pretrained=require_teacher_weights,
                pretrained_sha256=teacher_weights_sha256,
            )
            print(f"[DEIM] DINOv3-ViT-Large Online Semantic Distillation: ON (scale={distill_downsample_scale}, cos={distill_cos_weight}, gram={distill_gram_weight})")
        else:
            self.dinov3_distiller = None
        if use_scene_gate:
            self.scene_gate = SceneGeometryContextGate(num_classes=num_classes)
            print("[DEIM] 3D Scene-Geometry Context Gate (SGMG) -> Soft Prior: ON (zero-init)")
        else:
            self.scene_gate = None
        if use_planar_rigidity_gate:
            self.planar_rigidity_gate = PlanarRigidityContextGate(
                num_classes=num_classes,
                boost_logit=planar_gate_boost,
                suppress_logit=planar_gate_suppress
            )
            print("[DEIM] Planar Rigidity Context Gate (PRCG) -> Rigid 2D Plane vs 3D Bio Prior: ON")
        else:
            self.planar_rigidity_gate = None
        if use_frontend_naf:
            self.frontend_naf = FrontendNAFRestoration(
                in_channels=6, mid_channels=naf_mid_channels, num_blocks=naf_num_blocks
            )
            print(f"[DEIM] Combo 3: Frontend NAFBlock Quality Restoration (mid={naf_mid_channels}, blocks={naf_num_blocks}): ON (zero-init)")
        else:
            self.frontend_naf = None
        if use_p2_wavelet_sr:
            self.p2_wavelet_sr = WaveletHighFreqStarInjection(
                channels=128, scale=p2_wavelet_scale, blend_weight=p2_wavelet_blend
            )
            print(f"[DEIM] Combo 3: P2 Wavelet StarConv Super-Resolution Injection (scale={p2_wavelet_scale}x, blend={p2_wavelet_blend}): ON (zero-init)")
        else:
            self.p2_wavelet_sr = None
        if use_p2_cascade_zoom:
            self.p2_cascade_gate = P2CascadeZoomGate(
                roi_size=7,
                score_min=p2_zoom_score_min,
                score_max=p2_zoom_score_max,
                max_area=p2_zoom_max_area,
                saliency_thresh=p2_zoom_thresh,
                alpha_boost=p2_zoom_alpha,
                beta_suppress=0.0,
                num_classes=num_classes
            )
            print(f"[DEIM] Option 2: P2 Cascade RoI Zoom Gate (Stage 0 Zoom, alpha={p2_zoom_alpha}, thresh={p2_zoom_thresh}, area<={p2_zoom_max_area}): ON")
        else:
            self.p2_cascade_gate = None
        if use_internal_sahi:
            print("[DEIM] Internalized CAA-SAHI 3-View High-Res Slicing: ON (Zero Post-Processing)")
        if use_internal_tta:
            print("[DEIM] Internalized Horizontal Flip TTA Engine: ON (Zero Post-Processing)")

        if use_co_aux:
            from .co_aux_head import CoATSSAuxHead
            enc_dim = getattr(encoder, 'hidden_dim', 384)
            self.co_aux_head = CoATSSAuxHead(
                in_channels=enc_dim,
                num_classes=num_classes,
                loss_weight_cls=co_aux_loss_weight_cls,
                loss_weight_reg=co_aux_loss_weight_reg,
            )
        else:
            self.co_aux_head = None

        if use_physics_grounding:
            from .physics_grounding import PhysicalInductiveGating, PhysicalConsistencyLoss
            self.physics_gating = PhysicalInductiveGating(num_classes=num_classes)
            self.physics_loss_fn = PhysicalConsistencyLoss(loss_weight=physics_loss_weight)
            print("[DEIM] Physical Inductive Bias & Modality Grounding (PIG): ON (zero-init)")
        else:
            self.physics_gating = None
            self.physics_loss_fn = None

    def forward(self, x, targets=None):
        if self.use_frontend_naf and self.frontend_naf is not None:
            x = self.frontend_naf(x)

        if not self.training and self.use_internal_tta:
            return self._forward_internal_tta(x, targets)

        if not self.training and self.use_internal_sahi and x.shape[-2] == 640 and x.shape[-1] == 640:
            return self._forward_internal_sahi(x)

        depth = x[:, 5:6] if (x.shape[1] >= 6) else None
        feats = self.backbone(x)
        for i, f in enumerate(feats):
            if torch.isnan(f).any() or torch.isinf(f).any():
                print(f'[DEIM TRACE] backbone output {i} has NaN: {torch.isnan(f).sum().item()}, Inf: {torch.isinf(f).sum().item()}')
        if self.use_p2_wavelet_sr and self.p2_wavelet_sr is not None and len(feats) >= 1:
            feats[0] = self.p2_wavelet_sr(feats[0])
        p2_feat = feats[0] if (len(feats) == 4 and self.use_p2_cascade_zoom) else None
        feats = self.encoder(feats, depth=depth)
        for i, f in enumerate(feats):
            if torch.isnan(f).any() or torch.isinf(f).any():
                print(f'[DEIM TRACE] encoder output {i} has NaN: {torch.isnan(f).sum().item()}, Inf: {torch.isinf(f).sum().item()}')
        out = self.decoder(feats, targets)
        for k in ('pred_boxes', 'pred_logits'):
            if k in out and (torch.isnan(out[k]).any() or torch.isinf(out[k]).any()):
                print(f'[DEIM TRACE] decoder output {k} has NaN: {torch.isnan(out[k]).sum().item()}')

        if self.use_scene_gate and self.scene_gate is not None and depth is not None:
            gate = self.scene_gate(depth)  # [B, num_classes]
            out['pred_logits'] = out['pred_logits'] + gate.unsqueeze(1)
            if 'aux_outputs' in out and out['aux_outputs'] is not None:
                for aux in out['aux_outputs']:
                    aux['pred_logits'] = aux['pred_logits'] + gate.unsqueeze(1)
            if 'dn_outputs' in out and out['dn_outputs'] is not None:
                for dn in out['dn_outputs']:
                    dn['pred_logits'] = dn['pred_logits'] + gate.unsqueeze(1)

        if self.use_p2_cascade_zoom and self.p2_cascade_gate is not None and p2_feat is not None:
            out['pred_logits'] = self.p2_cascade_gate(p2_feat, out['pred_boxes'], out['pred_logits'])

        if self.use_planar_rigidity_gate and self.planar_rigidity_gate is not None and depth is not None:
            out['pred_logits'] = self.planar_rigidity_gate(depth, out['pred_boxes'], out['pred_logits'])

        if self.training and hasattr(self.backbone, 'get_align_loss'):
            align_l = self.backbone.get_align_loss()
            if isinstance(align_l, torch.Tensor) and align_l.requires_grad:
                out['loss_cmal'] = align_l

        if self.training and self.use_dinov3_distill and self.dinov3_distiller is not None:
            # feats contains multi-scale features: [P2, P3, P4, P5]
            # Stage 2 (P4, stride 16) matches DINOv3 patch representation
            distill_l = self.dinov3_distiller.compute_distill_loss(x[:, 0:3], feats)
            out['loss_distill'] = distill_l

        if self.use_physics_grounding and self.physics_gating is not None and x.shape[1] >= 6:
            mod_logits, aux_phys = self.physics_gating(out['pred_logits'], out['pred_boxes'], x)
            out['pred_logits'] = mod_logits
            if self.training and self.physics_loss_fn is not None:
                out['loss_physics'] = self.physics_loss_fn(mod_logits, aux_phys['delta_t'])

        if self.training and self.use_co_aux and self.co_aux_head is not None and targets is not None:
            co_losses = self.co_aux_head(feats, targets, img_size=(x.shape[-2], x.shape[-1]))
            out.update(co_losses)

        return out

    def _forward_internal_sahi(self, x):
        import torchvision
        from torchvision.ops import batched_nms, box_convert
        B = x.shape[0]

        # 1. Generate 16:9 native canvas (640 x 1152)
        canvas = torch.nn.functional.interpolate(x, size=(640, 1152), mode='bilinear', align_corners=False)
        p1 = canvas[:, :, :, 0:640]
        p2 = canvas[:, :, :, 512:1152]
        p0 = x

        x_all = torch.cat([p1, p2, p0], dim=0) # [3*B, 6, 640, 640]
        depth_all = x_all[:, 5:6] if x_all.shape[1] >= 6 else None

        feats_all = self.backbone(x_all)
        if self.use_p2_wavelet_sr and self.p2_wavelet_sr is not None and len(feats_all) >= 1:
            feats_all[0] = self.p2_wavelet_sr(feats_all[0])
        p2_all = feats_all[0] if (len(feats_all) == 4 and self.use_p2_cascade_zoom) else None
        feats_all = self.encoder(feats_all, depth=depth_all)
        out_all = self.decoder(feats_all)

        if self.use_scene_gate and self.scene_gate is not None and depth_all is not None:
            gate_all = self.scene_gate(depth_all)
            out_all['pred_logits'] = out_all['pred_logits'] + gate_all.unsqueeze(1)

        if self.use_p2_cascade_zoom and self.p2_cascade_gate is not None and p2_all is not None:
            out_all['pred_logits'] = self.p2_cascade_gate(p2_all, out_all['pred_boxes'], out_all['pred_logits'])

        if self.use_planar_rigidity_gate and self.planar_rigidity_gate is not None and depth_all is not None:
            out_all['pred_logits'] = self.planar_rigidity_gate(depth_all, out_all['pred_boxes'], out_all['pred_logits'])

        logits_p1, logits_p2, logits_p0 = out_all['pred_logits'].chunk(3, dim=0)
        boxes_p1, boxes_p2, boxes_p0 = out_all['pred_boxes'].chunk(3, dim=0)

        W_canvas = 1152.0
        scale_w = 640.0 / W_canvas
        offset_w = 512.0 / W_canvas

        b1_canvas = boxes_p1.clone()
        b1_canvas[..., 0] = boxes_p1[..., 0] * scale_w
        b1_canvas[..., 2] = boxes_p1[..., 2] * scale_w

        b2_canvas = boxes_p2.clone()
        b2_canvas[..., 0] = offset_w + boxes_p2[..., 0] * scale_w
        b2_canvas[..., 2] = boxes_p2[..., 2] * scale_w

        b0_canvas = boxes_p0.clone()

        all_boxes = []
        all_logits = []

        P0_ONLY_CATS = {7, 8, 11} # garbage can (7), ball (8), tricycle (11) (easily bisected by patch seams, kept from global P0)

        for b in range(B):
            xyxy1 = box_convert(b1_canvas[b], in_fmt="cxcywh", out_fmt="xyxy")
            xyxy2 = box_convert(b2_canvas[b], in_fmt="cxcywh", out_fmt="xyxy")

            keep1 = (xyxy1[:, 2] < (615.0 / W_canvas)) & (b1_canvas[b, :, 2] < (320.0 / W_canvas))
            keep2 = (xyxy2[:, 0] > (537.0 / W_canvas)) & (b2_canvas[b, :, 2] < (320.0 / W_canvas))

            labels1 = logits_p1[b].argmax(dim=-1)
            labels2 = logits_p2[b].argmax(dim=-1)
            for c in P0_ONLY_CATS:
                keep1 = keep1 & (labels1 != c)
                keep2 = keep2 & (labels2 != c)

            boxes_cand = torch.cat([b1_canvas[b][keep1], b2_canvas[b][keep2], b0_canvas[b]], dim=0)
            logits_cand = torch.cat([logits_p1[b][keep1], logits_p2[b][keep2], logits_p0[b]], dim=0)

            xyxy_cand = box_convert(boxes_cand, in_fmt="cxcywh", out_fmt="xyxy")
            scores_cand, labels_cand = torch.sigmoid(logits_cand).max(dim=-1)

            sahi_thresh = getattr(self, 'sahi_iou_threshold', 0.65)
            keep_nms = batched_nms(xyxy_cand, scores_cand, labels_cand, iou_threshold=sahi_thresh)
            keep_nms = keep_nms[:self.num_queries]

            if len(keep_nms) < self.num_queries:
                pad_num = self.num_queries - len(keep_nms)
                b_pad = torch.zeros(pad_num, 4, device=boxes_cand.device)
                l_pad = torch.full((pad_num, self.num_classes), -10.0, device=logits_cand.device)
                b_res = torch.cat([boxes_cand[keep_nms], b_pad], dim=0)
                l_res = torch.cat([logits_cand[keep_nms], l_pad], dim=0)
            else:
                b_res = boxes_cand[keep_nms]
                l_res = logits_cand[keep_nms]

            all_boxes.append(b_res)
            all_logits.append(l_res)

        return {
            'pred_logits': torch.stack(all_logits, dim=0),
            'pred_boxes': torch.stack(all_boxes, dim=0)
        }

    def _forward_internal_tta(self, x, targets=None):
        from torchvision.ops import batched_nms, box_convert
        B = x.shape[0]

        # 1. Batch [original, horizontal_flip] in a single combined tensor
        x_flip = torch.flip(x, dims=[-1])
        batch_2x = torch.cat([x, x_flip], dim=0) # [2*B, C, H, W]

        depth_all = batch_2x[:, 5:6] if batch_2x.shape[1] >= 6 else None
        feats_all = self.backbone(batch_2x)
        if self.use_p2_wavelet_sr and self.p2_wavelet_sr is not None and len(feats_all) >= 1:
            feats_all[0] = self.p2_wavelet_sr(feats_all[0])
        p2_all = feats_all[0] if (len(feats_all) == 4 and self.use_p2_cascade_zoom) else None
        feats_all = self.encoder(feats_all, depth=depth_all)
        out_all = self.decoder(feats_all, targets)

        if self.use_scene_gate and self.scene_gate is not None and depth_all is not None:
            gate_all = self.scene_gate(depth_all)
            out_all['pred_logits'] = out_all['pred_logits'] + gate_all.unsqueeze(1)

        if self.use_p2_cascade_zoom and self.p2_cascade_gate is not None and p2_all is not None:
            out_all['pred_logits'] = self.p2_cascade_gate(p2_all, out_all['pred_boxes'], out_all['pred_logits'])

        if self.use_planar_rigidity_gate and self.planar_rigidity_gate is not None and depth_all is not None:
            out_all['pred_logits'] = self.planar_rigidity_gate(depth_all, out_all['pred_boxes'], out_all['pred_logits'])

        logits_orig, logits_flip = out_all['pred_logits'].chunk(2, dim=0) # [B, num_queries, num_classes]
        boxes_orig, boxes_flip = out_all['pred_boxes'].chunk(2, dim=0)   # [B, num_queries, 4]

        # 2. Mirror-restore flipped coordinates: cx -> 1.0 - cx
        boxes_flip_restored = boxes_flip.clone()
        boxes_flip_restored[..., 0] = 1.0 - boxes_flip[..., 0]

        all_boxes = []
        all_logits = []

        for b in range(B):
            b_orig = boxes_orig[b] # [num_queries, 4]
            b_flip = boxes_flip_restored[b]
            l_orig = logits_orig[b] # [num_queries, num_classes]
            l_flip = logits_flip[b]

            boxes_cand = torch.cat([b_orig, b_flip], dim=0)   # [2*num_queries, 4]
            logits_cand = torch.cat([l_orig, l_flip], dim=0) # [2*num_queries, num_classes]

            xyxy_cand = box_convert(boxes_cand, in_fmt="cxcywh", out_fmt="xyxy")
            scores_cand, labels_cand = torch.sigmoid(logits_cand).max(dim=-1)

            tta_nms_iou = getattr(self, 'tta_nms_iou', 0.60)
            keep_nms = batched_nms(xyxy_cand, scores_cand, labels_cand, iou_threshold=tta_nms_iou)
            keep_nms = keep_nms[:self.num_queries]

            if len(keep_nms) < self.num_queries:
                pad_num = self.num_queries - len(keep_nms)
                b_pad = torch.zeros(pad_num, 4, device=boxes_cand.device)
                l_pad = torch.full((pad_num, self.num_classes), -10.0, device=logits_cand.device)
                b_res = torch.cat([boxes_cand[keep_nms], b_pad], dim=0)
                l_res = torch.cat([logits_cand[keep_nms], l_pad], dim=0)
            else:
                b_res = boxes_cand[keep_nms]
                l_res = logits_cand[keep_nms]

            all_boxes.append(b_res)
            all_logits.append(l_res)

        return {
            'pred_logits': torch.stack(all_logits, dim=0),
            'pred_boxes': torch.stack(all_boxes, dim=0)
        }

    def deploy(self, ):
        self.eval()
        self.use_internal_sahi = False
        self.use_internal_tta = False
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self

