"""
Co-DETR Style One-to-Many Auxiliary Supervision Head for DEIMv2.
=============================================================================
设计哲学 (Philosophical & Mathematical Rationale):
  1. 根治 One-to-One 匹配的过度保守性:
     DETR 纯匈牙利匹配时每个真实目标每轮仅被分配 1 个 Query，导致周围大量高质量特征
     被强行判为背景，分类头为了减小损失方差被迫压低置信度 (0.2 ~ 0.4)。
  2. 训练端引入 One-to-Many 密集监督 (ATSS):
     在 HybridEncoder 输出的多尺度特征 (P3, P4, P5) 上挂载轻量级卷积辅助头。
     每个真实目标自适应匹配 6 ~ 18 个正样本锚点，反向传播共享 Backbone 与 Encoder，
     向特征层注入密集、高对比度的前景语义梯度，极大激发特征判别力与前景峰值响应。
  3. 推理期零额外开销 (Zero-Overhead at Inference):
     仅在 training == True 且传入 targets 时计算辅助损失；在 model.eval() 推理期完全
     跳过，不增加任何推理计算量与显存占用，产出严格单模型端到端预测。
=============================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class Scale(nn.Module):
    """Learnable scale layer per feature level."""
    def __init__(self, init_value=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([init_value], dtype=torch.float32))

    def forward(self, x):
        return x * self.scale


def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Generalized IoU between pairs of boxes of shape (N, 4) with FP32 safety."""
    assert boxes1.shape == boxes2.shape
    b1 = boxes1.float()
    b2 = boxes2.float()
    area1 = (b1[:, 2] - b1[:, 0]).clamp(min=0) * (b1[:, 3] - b1[:, 1]).clamp(min=0)
    area2 = (b2[:, 2] - b2[:, 0]).clamp(min=0) * (b2[:, 3] - b2[:, 1]).clamp(min=0)

    lt = torch.max(b1[:, :2], b2[:, :2])
    rb = torch.min(b1[:, 2:], b2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    union = area1 + area2 - inter
    iou = inter / union.clamp(min=1e-5)

    # Convex hull
    c_lt = torch.min(b1[:, :2], b2[:, :2])
    c_rb = torch.max(b1[:, 2:], b2[:, 2:])
    c_wh = (c_rb - c_lt).clamp(min=0)
    c_area = c_wh[:, 0] * c_wh[:, 1]

    giou = iou - (c_area - union) / c_area.clamp(min=1e-5)
    return giou.clamp(min=-1.0, max=1.0).to(boxes1.dtype)



class CoATSSAuxHead(nn.Module):
    """
    Co-DETR style ATSS (Adaptive Training Sample Selection) auxiliary head.
    Operates on multi-scale feature pyramid [P3, P4, P5].
    """
    def __init__(
        self,
        in_channels: int = 384,
        num_classes: int = 12,
        num_convs: int = 2,
        strides=(8, 16, 32),
        topk: int = 9,
        gamma: float = 1.5,
        alpha: float = 0.25,
        loss_weight_cls: float = 1.0,
        loss_weight_reg: float = 1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.strides = list(strides)
        self.topk = topk
        self.gamma = gamma
        self.alpha = alpha
        self.loss_weight_cls = loss_weight_cls
        self.loss_weight_reg = loss_weight_reg

        # Shared lightweight convolutional towers
        cls_layers = []
        reg_layers = []
        for _ in range(num_convs):
            cls_layers.append(nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False))
            cls_layers.append(nn.GroupNorm(32, in_channels))
            cls_layers.append(nn.ReLU(inplace=True))

            reg_layers.append(nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False))
            reg_layers.append(nn.GroupNorm(32, in_channels))
            reg_layers.append(nn.ReLU(inplace=True))

        self.cls_tower = nn.Sequential(*cls_layers)
        self.reg_tower = nn.Sequential(*reg_layers)

        self.cls_head = nn.Conv2d(in_channels, num_classes, 3, padding=1)
        self.reg_head = nn.Conv2d(in_channels, 4, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in range(len(self.strides))])

        self._init_weights()
        print(f"[CoATSSAuxHead] Initialized One-to-Many Auxiliary Supervision Head (classes={num_classes}, strides={self.strides}, topk={topk}, gamma={gamma})")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # Prior probability initialization for focal loss stability
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_head.bias, bias_value)

    def _generate_grid_anchors(self, feats, img_size, device):
        H_img, W_img = img_size
        anchors_list = []
        centers_list = []
        strides_list = []
        grid_info = []

        for lvl, (feat, stride) in enumerate(zip(feats, self.strides)):
            h, w = feat.shape[-2:]
            gy, gx = torch.meshgrid(
                torch.arange(h, device=device),
                torch.arange(w, device=device),
                indexing='ij'
            )
            xc = (gx.float() + 0.5) / float(w)
            yc = (gy.float() + 0.5) / float(h)
            wa = 4.0 * float(stride) / float(W_img)
            ha = 4.0 * float(stride) / float(H_img)

            centers = torch.stack([xc, yc], dim=-1).reshape(-1, 2)
            boxes = torch.stack([
                (xc - wa * 0.5).clamp(min=0.0),
                (yc - ha * 0.5).clamp(min=0.0),
                (xc + wa * 0.5).clamp(max=1.0),
                (yc + ha * 0.5).clamp(max=1.0),
            ], dim=-1).reshape(-1, 4)

            anchors_list.append(boxes)
            centers_list.append(centers)
            strides_list.append(torch.full((h * w,), stride, dtype=torch.float32, device=device))
            grid_info.append((h, w, stride))

        all_anchors = torch.cat(anchors_list, dim=0) # (N_total, 4)
        all_centers = torch.cat(centers_list, dim=0) # (N_total, 2)
        all_strides = torch.cat(strides_list, dim=0) # (N_total,)
        return all_anchors, all_centers, all_strides, grid_info

    def forward(self, feats, targets=None, img_size=(1440, 2560)):
        if not self.training or targets is None:
            return {}

        B = feats[0].shape[0]
        device = feats[0].device
        H_img, W_img = img_size

        all_anchors, all_centers, all_strides, grid_info = self._generate_grid_anchors(feats, img_size, device)
        N_total = len(all_anchors)

        cls_logits_all = []
        reg_preds_all = []

        for lvl, (feat, stride) in enumerate(zip(feats, self.strides)):
            cls_feat = self.cls_tower(feat)
            reg_feat = self.reg_tower(feat)

            cls_l = self.cls_head(cls_feat) # (B, num_classes, h, w)
            reg_l = self.scales[lvl](self.reg_head(reg_feat)) # (B, 4, h, w)

            cls_logits_all.append(cls_l.permute(0, 2, 3, 1).reshape(B, -1, self.num_classes))
            reg_preds_all.append(reg_l.permute(0, 2, 3, 1).reshape(B, -1, 4))

        cls_logits_cat = torch.cat(cls_logits_all, dim=1) # (B, N_total, num_classes)
        reg_preds_cat = torch.cat(reg_preds_all, dim=1)   # (B, N_total, 4)

        # Decode predicted boxes from distance offsets (ltrb) in normalized coordinates
        # Clamp in log-space [-8.0, 5.0] to prevent exp overflow in FP16
        dist_ltrb = torch.exp(torch.clamp(reg_preds_cat.float(), min=-8.0, max=5.0))
        all_centers_f = all_centers.float()
        all_strides_f = all_strides.float()
        # Maintain strictly positive width (x2 > x1) and height (y2 > y1) to prevent zero-area box collapse
        offset_x = dist_ltrb[..., [0, 2]] * (all_strides_f.unsqueeze(0).unsqueeze(-1) / float(W_img))
        offset_y = dist_ltrb[..., [1, 3]] * (all_strides_f.unsqueeze(0).unsqueeze(-1) / float(H_img))
        pred_x1 = all_centers_f[:, 0].unsqueeze(0) - offset_x[..., 0]
        pred_y1 = all_centers_f[:, 1].unsqueeze(0) - offset_y[..., 0]
        pred_x2 = all_centers_f[:, 0].unsqueeze(0) + offset_x[..., 1]
        pred_y2 = all_centers_f[:, 1].unsqueeze(0) + offset_y[..., 1]
        pred_boxes_cat = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=-1)



        # Target Assignment per Image
        total_loss_cls = torch.tensor(0.0, device=device)
        total_loss_reg = torch.tensor(0.0, device=device)
        total_num_pos = 0

        for b in range(B):
            tgt = targets[b]
            gt_boxes_raw = tgt['boxes'] # cxcywh normalized
            gt_labels = tgt['labels']   # (M,)
            M = len(gt_boxes_raw)

            logits_b = cls_logits_cat[b]     # (N_total, num_classes)
            pred_boxes_b = pred_boxes_cat[b] # (N_total, 4)

            if M == 0:
                # All background focal loss
                target_onehot = torch.zeros_like(logits_b)
                loss_cls_b = torchvision.ops.sigmoid_focal_loss(
                    logits_b, target_onehot, alpha=self.alpha, gamma=self.gamma, reduction='sum'
                )
                total_loss_cls = total_loss_cls + loss_cls_b
                continue

            gt_xyxy = box_cxcywh_to_xyxy(gt_boxes_raw) # (M, 4)
            gt_cx = gt_boxes_raw[:, 0]
            gt_cy = gt_boxes_raw[:, 1]

            # Fast O(1) Candidate Anchor Indexing (3x3 grid around center on each level)
            cand_indices = []
            offset_idx = 0
            for h, w, stride in grid_info:
                gj = (gt_cx * float(w)).long().clamp(1, max(1, w - 2))
                gi = (gt_cy * float(h)).long().clamp(1, max(1, h - 2))
                # 3x3 offsets
                dx = torch.tensor([-1, 0, 1], device=device)
                dy = torch.tensor([-1, 0, 1], device=device)
                mesh_dy, mesh_dx = torch.meshgrid(dy, dx, indexing='ij')
                gx = (gj.unsqueeze(1).unsqueeze(2) + mesh_dx.unsqueeze(0)).clamp(0, w - 1) # (M, 3, 3)
                gy = (gi.unsqueeze(1).unsqueeze(2) + mesh_dy.unsqueeze(0)).clamp(0, h - 1) # (M, 3, 3)
                flat_idx = (gy * w + gx).reshape(M, 9) + offset_idx      # (M, 9)
                cand_indices.append(flat_idx)
                offset_idx += (h * w)

            cand_idx = torch.cat(cand_indices, dim=1).clamp(0, N_total - 1) # (M, K_cands)
            K_cands = cand_idx.shape[1]
            cand_anchors = all_anchors[cand_idx]      # (M, K_cands, 4)
            cand_centers = all_centers[cand_idx]      # (M, K_cands, 2)

            # IoU between candidate anchors and corresponding GT
            gt_expand = gt_xyxy.unsqueeze(1).expand(-1, K_cands, -1)
            inter_x1 = torch.max(cand_anchors[..., 0], gt_expand[..., 0])
            inter_y1 = torch.max(cand_anchors[..., 1], gt_expand[..., 1])
            inter_x2 = torch.min(cand_anchors[..., 2], gt_expand[..., 2])
            inter_y2 = torch.min(cand_anchors[..., 3], gt_expand[..., 3])
            inter_w = (inter_x2 - inter_x1).clamp(min=0.0)
            inter_h = (inter_y2 - inter_y1).clamp(min=0.0)
            inter_area = inter_w * inter_h
            area_cand = (cand_anchors[..., 2] - cand_anchors[..., 0]) * (cand_anchors[..., 3] - cand_anchors[..., 1])
            area_gt = (gt_expand[..., 2] - gt_expand[..., 0]) * (gt_expand[..., 3] - gt_expand[..., 1])
            union = (area_cand + area_gt - inter_area).clamp(min=1e-6)
            cand_ious = inter_area / union # (M, K_cands)

            iou_mean = cand_ious.mean(dim=1, keepdim=True)
            iou_std = cand_ious.std(dim=1, keepdim=True)
            iou_thresh = (iou_mean + iou_std).clamp(min=0.1)

            inside_gt = (
                (cand_centers[..., 0] >= gt_expand[..., 0]) &
                (cand_centers[..., 0] <= gt_expand[..., 2]) &
                (cand_centers[..., 1] >= gt_expand[..., 1]) &
                (cand_centers[..., 1] <= gt_expand[..., 3])
            )

            is_pos = (cand_ious >= iou_thresh) & inside_gt # (M, K_cands)

            # Fallback: if no anchor passed threshold for a GT, take top-1 candidate with highest IoU
            for m in range(M):
                if not is_pos[m].any():
                    best_cand = cand_ious[m].argmax()
                    is_pos[m, best_cand] = True

            # Assign targets with conflict resolution
            target_onehot = torch.zeros_like(logits_b)
            anchor_to_gt = torch.full((N_total,), -1, dtype=torch.long, device=device)
            anchor_to_iou = torch.full((N_total,), -1.0, dtype=torch.float32, device=device)

            for m in range(M):
                pos_cand_mask = is_pos[m]
                selected_anchor_idx = cand_idx[m][pos_cand_mask].clamp(0, N_total - 1)
                selected_ious = cand_ious[m][pos_cand_mask]
                # Resolve anchor conflict: GT with highest IoU claims this anchor for regression
                better_mask = selected_ious > anchor_to_iou[selected_anchor_idx]
                win_anchors = selected_anchor_idx[better_mask]
                anchor_to_gt[win_anchors] = m
                anchor_to_iou[win_anchors] = selected_ious[better_mask]

            pos_mask = anchor_to_gt >= 0
            if pos_mask.any():
                pos_anchor_idx = torch.nonzero(pos_mask, as_tuple=True)[0]
                assigned_gts = anchor_to_gt[pos_anchor_idx]
                # Classification and regression must describe the same owner.
                target_onehot[pos_anchor_idx, gt_labels[assigned_gts]] = 1.0
                pos_gt_boxes_cat = gt_xyxy[assigned_gts]

                # Regression Loss on Positive Anchors (GIoU Loss via torchvision native kernel)
                pred_boxes_pos = pred_boxes_b[pos_anchor_idx]
                loss_reg_b = torchvision.ops.generalized_box_iou_loss(
                    pred_boxes_pos.float(), pos_gt_boxes_cat.float(), reduction='sum'
                )
                num_pos_b = len(pos_anchor_idx)
            else:
                loss_reg_b = torch.tensor(0.0, device=device)
                num_pos_b = 0

            # Classification Loss (Sigmoid Focal Loss on all anchors)
            loss_cls_b = torchvision.ops.sigmoid_focal_loss(
                logits_b, target_onehot, alpha=self.alpha, gamma=self.gamma, reduction='sum'
            )

            total_loss_cls = total_loss_cls + loss_cls_b
            total_loss_reg = total_loss_reg + loss_reg_b
            total_num_pos += num_pos_b

        num_pos_norm = max(float(total_num_pos), 4.0)
        # Raw normalized losses; weighting is uniformly governed by Criterion's weight_dict
        norm_loss_cls = total_loss_cls / num_pos_norm
        norm_loss_reg = total_loss_reg / num_pos_norm

        return {
            'loss_co_cls': norm_loss_cls,
            'loss_co_reg': norm_loss_reg,
        }
