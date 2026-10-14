"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE/)
Copyright (c) 2024 D-FINE Authors. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.distributed
import torch.nn.functional as F
import torchvision

import copy

from .dfine_utils import bbox2distance
from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou, box_nwd
from ..misc.dist_utils import get_world_size, is_dist_available_and_initialized
from ..core import register


@register()
class DEIMCriterion(nn.Module):
    """ This class computes the loss for DEIM.
    """
    __share__ = ['num_classes', ]
    __inject__ = ['matcher', ]

    def __init__(self, \
        matcher,
        weight_dict,
        losses,
        alpha=0.2,
        gamma=2.0,
        num_classes=80,
        reg_max=32,
        boxes_weight_format=None,
        share_matched_indices=False,
        mal_alpha=None,
        use_uni_set=True,
        rs_margin=1.0,
        rs_sort_w=1.0,
        use_depth_prior=False,
        depth_bins=(0.33, 0.66),
        depth_weights=(1.0, 2.0, 4.0),
        depth_invert=False,
        cls_mal_scale=None,
        cls_loss_weights=None,
        tadl_beta=0.5,
        tadl_min_iou=0.35,
        cls_neg_scale=None,
        use_ndsl=False,
        ndsl_min_iou=0.35,
        ndsl_factor=2.5,
        ):
        """Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals.
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            num_classes: number of object categories, omitting the special no-object category.
            reg_max (int): Max number of the discrete bins in D-FINE.
            boxes_weight_format: format for boxes weight (iou, ).
            use_depth_prior: DepthPrior switch. When True the box/cls losses are
                weighted by the GT depth so that far/small targets (where RGB is
                unreliable at night and depth is the dominant cue) contribute more
                -- DLS (depth-stratified loss) + DLW (depth-weighted loss).
            depth_bins / depth_weights: bin boundaries (in raw depth [0,1]) and the
                weight applied to each bin; len(depth_weights) == len(depth_bins)+1.
            depth_invert: if True, depth is inverted before binning. AIC depth is
                now read as ABSOLUTE depth (mm/19999): near = LOW value, far =
                HIGH value, and depth_weights [1,2,4] must up-weight FAR boxes ->
                keep this False. Set True only when re-scoring a legacy checkpoint
                that was trained on the old clipped (binary) depth maps.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.boxes_weight_format = boxes_weight_format
        self.share_matched_indices = share_matched_indices
        self.alpha = alpha
        self.gamma = gamma
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.own_targets, self.own_targets_dn = None, None
        self.reg_max = reg_max
        self.num_pos, self.num_neg = None, None
        self.mal_alpha = mal_alpha
        self.use_uni_set = use_uni_set
        self.rs_margin = rs_margin        # RS Loss ascending-rank hinge margin
        self.rs_sort_w = rs_sort_w        # weight of the IoU-order sort term
        self.use_depth_prior = use_depth_prior
        self.depth_bins = list(depth_bins)
        self.depth_weights = list(depth_weights)
        self.depth_invert = depth_invert
        self.cls_mal_scale = list(cls_mal_scale) if cls_mal_scale is not None else None
        self.cls_loss_weights = list(cls_loss_weights) if cls_loss_weights is not None else None
        self.tadl_beta = tadl_beta
        self.tadl_min_iou = tadl_min_iou
        self.use_ndsl = use_ndsl
        self.ndsl_min_iou = float(ndsl_min_iou)
        self.ndsl_factor = float(ndsl_factor)
        if use_ndsl:
            print(f"[DEIMCriterion] Near-Duplicate Suppression Loss (NDSL): ON (min_iou={self.ndsl_min_iou}, factor={self.ndsl_factor})")
        self.cls_neg_scale = list(cls_neg_scale) if cls_neg_scale is not None else [
            1.0,  # 0: person
            1.5,  # 1: boat
            1.0,  # 2: animal
            1.0,  # 3: seat
            1.0,  # 4: sign
            1.0,  # 5: bicycle
            1.0,  # 6: car
            2.0,  # 7: ball (anti-hallucination)
            1.0,  # 8: light
            1.2,  # 9: garbage can
            1.5,  # 10: uav
            2.0,  # 11: tricycle (anti-hallucination)
        ]

    def loss_labels_focal(self, outputs, targets, indices, num_boxes, depth_weight=None):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes+1)[..., :-1]
        loss = torchvision.ops.sigmoid_focal_loss(src_logits, target, self.alpha, self.gamma, reduction='none')
        if depth_weight is not None:
            loss = loss * depth_weight.view(src_logits.shape[0], src_logits.shape[1], 1)
        if self.cls_loss_weights is not None and src_logits.shape[-1] == len(self.cls_loss_weights):
            w_cls = torch.as_tensor(self.cls_loss_weights, dtype=loss.dtype, device=loss.device)
            loss = loss * w_cls.view(1, 1, -1)
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_focal': loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, values=None, depth_weight=None):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            sb_xyxy = box_cxcywh_to_xyxy(src_boxes)
            tb_xyxy = box_cxcywh_to_xyxy(target_boxes)
            ious, _ = box_iou(sb_xyxy, tb_xyxy)
            diag_iou = torch.diag(ious).detach()
            nwd = box_nwd(sb_xyxy, tb_xyxy).detach()
            ious = 0.6 * diag_iou + 0.4 * nwd
        else:
            ious = values

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score

        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        if depth_weight is not None:
            loss = loss * depth_weight.view(src_logits.shape[0], src_logits.shape[1], 1)
        if self.cls_loss_weights is not None and src_logits.shape[-1] == len(self.cls_loss_weights):
            w_cls = torch.as_tensor(self.cls_loss_weights, dtype=loss.dtype, device=loss.device)
            loss = loss * w_cls.view(1, 1, -1)
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_vfl': loss}

    def loss_labels_mal(self, outputs, targets, indices, num_boxes, values=None, depth_weight=None):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            sb_xyxy = box_cxcywh_to_xyxy(src_boxes)
            tb_xyxy = box_cxcywh_to_xyxy(target_boxes)
            ious, _ = box_iou(sb_xyxy, tb_xyxy)
            diag_iou = torch.diag(ious).detach()
            nwd = box_nwd(sb_xyxy, tb_xyxy).detach()
            ious = 0.6 * diag_iou + 0.4 * nwd
        else:
            ious = values

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        target_score = target_score.pow(self.gamma)
        if self.mal_alpha != None:
            weight = self.mal_alpha * pred_score.pow(self.gamma) * (1 - target) + target
        else:
            weight = pred_score.pow(self.gamma) * (1 - target) + target

        if getattr(self, 'cls_mal_scale', None) is not None:
            cls_w = torch.as_tensor(self.cls_mal_scale, device=src_logits.device, dtype=weight.dtype)
            weight = weight * cls_w.view(1, 1, -1)

        # print(" ### DEIM-gamma{}-alpha{} ### ".format(self.gamma, self.mal_alpha))
        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        if depth_weight is not None:
            loss = loss * depth_weight.view(src_logits.shape[0], src_logits.shape[1], 1)
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_mal': loss}

    def loss_labels_tadl(self, outputs, targets, indices, num_boxes, values=None, depth_weight=None):
        """Task-Aligned Decoupled Loss (TADL).
        
        Key Mathematical Properties:
        1. Decoupled Ceiling: The positive target is strictly 1.0 (decoupled from IoU).
           Unlike MAL/VFL where target = IoU^1.5 (capping weak/distant objects at 0.35),
           TADL allows ALL true objects to reach high confidence (0.80 ~ 0.95), aligning
           with F5's hard Focal Loss confidence calibration.
        2. Task-Aligned Quality Weighting: Positives are weighted by q^beta where
           q = clamp(IoU_quality, min=tadl_min_iou, max=1.0). Higher quality proposals
           receive proportionally stronger positive gradients, maintaining optimal ranking.
        3. Anti-Hallucination Rare Class Negative Suppression: Background queries are
           weighted by alpha * p^gamma, with class-specific multipliers (e.g. 2.0x for ball
           and tricycle), preventing false positives in long-tail categories.
        """
        assert 'pred_boxes' in outputs and 'pred_logits' in outputs
        idx = self._get_src_permutation_idx(indices)
        batch_idx, src_idx = idx

        if values is None:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            sb_xyxy = box_cxcywh_to_xyxy(src_boxes)
            tb_xyxy = box_cxcywh_to_xyxy(target_boxes)
            ious, _ = box_iou(sb_xyxy, tb_xyxy)
            diag_iou = torch.diag(ious).detach()
            nwd = box_nwd(sb_xyxy, tb_xyxy).detach()
            ious = 0.6 * diag_iou + 0.4 * nwd
        else:
            ious = values

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1].float()

        # 1. Quality factor q for positive matches
        q_clamped = torch.clamp(ious, min=self.tadl_min_iou, max=1.0).pow(self.tadl_beta)

        # 2. Positive weighting matrix (only scales positive entries)
        pos_scale = torch.ones_like(target)
        pos_scale[batch_idx, src_idx, target_classes_o] = q_clamped.to(pos_scale.dtype)

        # 3. Rare-class negative suppression multipliers
        if self.cls_neg_scale is not None and len(self.cls_neg_scale) == self.num_classes:
            neg_cls_tensor = torch.as_tensor(self.cls_neg_scale, dtype=src_logits.dtype, device=src_logits.device).view(1, 1, -1)
            neg_scale = (1.0 - target) * (neg_cls_tensor - 1.0) + 1.0
        else:
            neg_scale = 1.0

        # 3b. Near-Duplicate Suppression Loss (NDSL):
        # Heavily penalize unmatched queries that substantially overlap with GT boxes
        dup_scale = 1.0
        if getattr(self, 'use_ndsl', False) and 'pred_boxes' in outputs:
            dup_scale_mat = torch.ones_like(target)
            matched_mask = torch.zeros(src_logits.shape[:2], dtype=torch.bool, device=src_logits.device)
            matched_mask[batch_idx, src_idx] = True

            for b_i in range(src_logits.shape[0]):
                unmatched_q = (~matched_mask[b_i]).nonzero(as_tuple=True)[0]
                tgt_boxes_b = targets[b_i]['boxes']
                if len(unmatched_q) > 0 and len(tgt_boxes_b) > 0:
                    pred_b_unm = outputs['pred_boxes'][b_i, unmatched_q]
                    sb_xy = box_cxcywh_to_xyxy(pred_b_unm)
                    tb_xy = box_cxcywh_to_xyxy(tgt_boxes_b)
                    ious_unm, union_unm = box_iou(sb_xy, tb_xy)
                    
                    # Compute Part-Whole Containment: fraction of query area inside a GT box
                    sb_area = (sb_xy[:, 2] - sb_xy[:, 0]).clamp(min=0) * (sb_xy[:, 3] - sb_xy[:, 1]).clamp(min=0)
                    inter_mat = ious_unm * union_unm
                    containment = inter_mat / (sb_area.unsqueeze(1) + 1e-6)
                    max_containment = containment.max(dim=1).values
                    max_iou_unm = ious_unm.max(dim=1).values
                    
                    # Suppress near-duplicates and partial appendages (e.g. peacock tail/body duplicate)
                    dup_mask = (max_iou_unm >= self.ndsl_min_iou) | (max_containment >= 0.40)
                    if dup_mask.any():
                        dup_q_idx = unmatched_q[dup_mask]
                        dup_penalty_val = torch.maximum(max_iou_unm[dup_mask], max_containment[dup_mask])
                        boost = 1.0 + self.ndsl_factor * dup_penalty_val.clamp(0.0, 1.0).pow(2)
                        dup_scale_mat[b_i, dup_q_idx, :] = boost.unsqueeze(-1).to(dup_scale_mat.dtype)
            dup_scale = dup_scale_mat

        # 4. Focal modulation
        p = torch.sigmoid(src_logits)
        p_t = p * target + (1.0 - p) * (1.0 - target)
        modulating = (1.0 - p_t.detach()).pow(self.gamma)
        alpha_factor = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)

        weight = alpha_factor * modulating * pos_scale * neg_scale * dup_scale

        # 5. Decoupled binary cross entropy with target = 1.0 for positives, 0.0 for negatives
        bce = F.binary_cross_entropy_with_logits(src_logits, target, reduction='none')
        loss = weight * bce

        if depth_weight is not None:
            loss = loss * depth_weight.view(src_logits.shape[0], src_logits.shape[1], 1)

        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_tadl': loss}

    def loss_labels_rs(self, outputs, targets, indices, num_boxes, values=None, depth_weight=None):
        """Rank & Sort Loss (Oksuz et al., ICCV 2021) adapted to DETR sigmoid heads.

        Directly optimizes the AP-relevant ordering: every TP must rank above
        every FP (ascending-rank hinge), and positives must be ordered by IoU
        (sort term). This targets the failure mode where high-scoring FPs
        outrank low-scoring TPs (e.g. sign: TP floor 0.062, 2161 FPs above it).

        Simplifications for multi-class sigmoid channels:
          - positives = matched queries, score = sigmoid logit of assigned class
          - negatives = unmatched queries, score = max over class channels
          - per-image computation, normalized by #FPs above each TP (paper Eq.)
        """
        src_logits = outputs['pred_logits']          # [B, Q, C]
        B, Q, _ = src_logits.shape
        scores = src_logits.sigmoid()
        total = src_logits.new_zeros(())
        total_pos = 0
        for b in range(B):
            src_idx, tgt_idx = indices[b]
            if src_idx.numel() == 0:
                continue
            pos_cls = targets[b]['labels'][tgt_idx]
            pos_score = scores[b, src_idx, pos_cls]                    # [P], grad flows
            if values is not None:
                # flat IoU tensor aligned with indices order (same as MAL)
                pos_iou = values[total_pos:total_pos + src_idx.numel()].detach()
            else:
                src_boxes = outputs['pred_boxes'][b, src_idx]
                tgt_boxes = targets[b]['boxes'][tgt_idx]
                iou, _ = box_iou(box_cxcywh_to_xyxy(src_boxes.detach()),
                                 box_cxcywh_to_xyxy(tgt_boxes))
                pos_iou = torch.diag(iou).detach()
            neg_mask = torch.ones(Q, dtype=torch.bool, device=scores.device)
            neg_mask[src_idx] = False
            neg_score = scores[b][neg_mask].max(dim=1).values          # [N], grad flows
            if neg_score.numel() == 0:
                total_pos += pos_score.numel()
                continue

            # ---- ascending rank loss: each TP penalized by FPs ranked above ----
            hinge = (self.rs_margin - pos_score.unsqueeze(1)
                     + neg_score.unsqueeze(0)).clamp(min=0)             # [P, N]
            ranks = (neg_score.unsqueeze(0) > pos_score.unsqueeze(1)).sum(dim=1).float()
            l_rank = torch.where(ranks > 0, hinge.sum(dim=1) / ranks.clamp(min=1),
                                 torch.zeros_like(ranks))
            l_rank = l_rank.sum() / pos_score.numel()

            # ---- sort term: IoU order must follow score order ----
            l_sort = src_logits.new_zeros(())
            if pos_score.numel() > 1:
                order = torch.argsort(pos_score, descending=True)
                s_sorted = pos_score[order]
                iou_sorted = pos_iou[order]
                w = (iou_sorted[:-1] - iou_sorted[1:]).clamp(min=0)     # IoU inversions
                hinge_s = (s_sorted[:-1] - s_sorted[1:]).clamp(min=0)   # push down low-IoU-above
                l_sort = (w * hinge_s).sum() / (pos_score.numel() - 1)

            total = total + l_rank + self.rs_sort_w * l_sort
            total_pos += pos_score.numel()

        if total_pos == 0:
            return {'loss_rs': src_logits.sum() * 0.0}
        loss = total * src_logits.shape[1] / max(num_boxes, 1)
        return {'loss_rs': loss}


    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None, depth_weight=None):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        dw = depth_weight[idx[0] * outputs['pred_boxes'].shape[1] + idx[1]] if depth_weight is not None else None
        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        if dw is not None:
            loss_bbox = loss_bbox * dw.unsqueeze(-1)
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        raw_giou = 1 - torch.diag(generalized_box_iou(\
            box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)))
        # NWD enhancement: Gaussian Wasserstein distance provides smooth non-vanishing gradients for small targets
        nwd_sim = box_nwd(src_boxes, target_boxes, C=0.5)
        loss_nwd = 1.0 - nwd_sim
        loss_giou = 0.6 * raw_giou + 0.4 * loss_nwd

        loss_giou = loss_giou if boxes_weight is None else loss_giou * boxes_weight
        if dw is not None:
            loss_giou = loss_giou * dw
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses

    def loss_dssm(self, outputs, targets, indices, num_boxes, boxes_weight=None, depth_weight=None):
        """Deformation-Sensitive Similarity Metric Loss (paper [4] Tang et al.,
        Information Fusion 2026, Eq.1-11). Replaces GIoU as the box reg loss.
        Boxes are (cx,cy,w,h) normalized. NOTE: the paper does NOT report the
        values of alpha (shape weight) and K (balance constant) -- paywalled; we
        use SIoU-style defaults alpha=1.0, K=1.0.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src = outputs['pred_boxes'][idx]
        tgt = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        x1, y1, w1, h1 = src[:, 0], src[:, 1], src[:, 2], src[:, 3]
        x2, y2, w2, h2 = tgt[:, 0], tgt[:, 1], tgt[:, 2], tgt[:, 3]
        eps = 1e-7
        # Eq.1 center distance, Eq.2 width/height distance
        d_center2 = (x1 - x2) ** 2 + (y1 - y2) ** 2
        d_wandh2 = (w1 - w2) ** 2 + (h1 - h2) ** 2
        # Eq.3-6 diagonal cosine similarity (angle between box vectors)
        v1 = torch.sqrt(w1 ** 2 + h1 ** 2 + eps)
        v2 = torch.sqrt(w2 ** 2 + h2 ** 2 + eps)
        r_cos = (w1 * w2 + h1 * h2) / (v1 * v2 + eps)
        d_angle = 1.0 * (1.0 - r_cos)              # Eq.7, alpha=1.0 (paper value unknown)
        # Eq.8 convex-hull diagonal normalisation
        dx = torch.max(x1 + w1 / 2, x2 + w2 / 2) - torch.min(x1 - w1 / 2, x2 - w2 / 2)
        dy = torch.max(y1 + h1 / 2, y2 + h2 / 2) - torch.min(y1 - h1 / 2, y2 - h2 / 2)
        d_diag2 = dx ** 2 + dy ** 2 + eps
        # Eq.9 composite, Eq.10 loss (negative-exponential similarity mapping)
        r_comp = (d_center2 + d_wandh2 + d_angle ** 2) / d_diag2
        loss = r_comp * torch.exp(-1.0 * r_comp)    # K=1.0 (paper value unknown)
        if boxes_weight is not None:
            loss = loss * boxes_weight
        return {'loss_dssm': loss.sum() / num_boxes}

    def loss_local(self, outputs, targets, indices, num_boxes, T=5, depth_weight=None):
        """Compute Fine-Grained Localization (FGL) Loss
            and Decoupled Distillation Focal (DDF) Loss. """

        losses = {}
        if 'pred_corners' in outputs:
            idx = self._get_src_permutation_idx(indices)
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

            pred_corners = outputs['pred_corners'][idx].reshape(-1, (self.reg_max+1)).clamp(-20.0, 20.0)
            ref_points = outputs['ref_points'][idx].detach()
            with torch.no_grad():
                if self.fgl_targets_dn is None and 'is_dn' in outputs:
                        self.fgl_targets_dn= bbox2distance(ref_points, box_cxcywh_to_xyxy(target_boxes),
                                                        self.reg_max, outputs['reg_scale'], outputs['up'])
                if self.fgl_targets is None and 'is_dn' not in outputs:
                        self.fgl_targets = bbox2distance(ref_points, box_cxcywh_to_xyxy(target_boxes),
                                                        self.reg_max, outputs['reg_scale'], outputs['up'])

            target_corners, weight_right, weight_left = self.fgl_targets_dn if 'is_dn' in outputs else self.fgl_targets

            ious = torch.diag(box_iou(\
                        box_cxcywh_to_xyxy(outputs['pred_boxes'][idx]), box_cxcywh_to_xyxy(target_boxes))[0])
            weight_targets = ious.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()

            losses['loss_fgl'] = self.unimodal_distribution_focal_loss(
                pred_corners, target_corners, weight_right, weight_left, weight_targets, avg_factor=num_boxes)

            if 'teacher_corners' in outputs:
                pred_corners = outputs['pred_corners'].reshape(-1, (self.reg_max+1))
                target_corners = outputs['teacher_corners'].reshape(-1, (self.reg_max+1))
                pred_corners = torch.nan_to_num(pred_corners, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
                target_corners = torch.nan_to_num(target_corners, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
                if not torch.equal(pred_corners, target_corners):
                    weight_targets_local = outputs['teacher_logits'].sigmoid().max(dim=-1)[0]

                    mask = torch.zeros_like(weight_targets_local, dtype=torch.bool)
                    mask[idx] = True
                    mask = mask.unsqueeze(-1).repeat(1, 1, 4).reshape(-1)

                    weight_targets_local[idx] = ious.reshape_as(weight_targets_local[idx]).to(weight_targets_local.dtype)
                    weight_targets_local = weight_targets_local.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()

                    loss_match_local = weight_targets_local * (T ** 2) * (nn.KLDivLoss(reduction='none')
                    (F.log_softmax(pred_corners / T, dim=1), F.softmax(target_corners.detach() / T, dim=1))).sum(-1)
                    if 'is_dn' not in outputs:
                        batch_scale = 8 / outputs['pred_boxes'].shape[0]  # Avoid the influence of batch size per GPU
                        self.num_pos, self.num_neg = (mask.sum() * batch_scale) ** 0.5, ((~mask).sum() * batch_scale) ** 0.5
                    loss_match_local1 = loss_match_local[mask].mean() if mask.any() else 0
                    loss_match_local2 = loss_match_local[~mask].mean() if (~mask).any() else 0
                    losses['loss_ddf'] = (loss_match_local1 * self.num_pos + loss_match_local2 * self.num_neg) / (self.num_pos + self.num_neg)

        return losses

    def loss_fcr(self, outputs, targets, indices, num_boxes, **kwargs):
        """Fourier Contrastive Regularization Loss (V1-054 / ECCV 2024).
        Computes 1D FFT amplitude spectra of matched foreground queries vs unmatched background queries.
        Pushes foreground frequency representations away from background noise representations.
        Helps discern low-contrast poles and faint boats in nighttime / glare scenes.
        """
        if 'pred_corners' not in outputs or len(indices) == 0:
            return {'loss_fcr': torch.tensor(0.0, device=outputs['pred_logits'].device)}
        
        corners = outputs['pred_corners'] # (B, Q, 4 * (reg_max + 1))
        B, Q, C_c = corners.shape
        corners_4d = corners.view(B, Q, 4, -1) # (B, Q, 4, 33)
        
        # 1D FFT along distribution bins (dim=-1) - cast to float32 for cuFFT under AMP
        fft_amp = torch.abs(torch.fft.rfft(corners_4d.float(), dim=-1)) # (B, Q, 4, 17)
        fft_flat = fft_amp.view(B, Q, -1) # (B, Q, 68)
        
        losses = []
        for b in range(B):
            src_idx, _ = indices[b]
            if len(src_idx) == 0:
                continue
            all_idx = torch.arange(Q, device=corners.device)
            neg_mask = torch.ones(Q, dtype=torch.bool, device=corners.device)
            neg_mask[src_idx] = False
            neg_idx = all_idx[neg_mask]
            
            if len(neg_idx) == 0:
                continue
                
            pos_fft = fft_flat[b, src_idx] # (N_pos, 68)
            neg_fft = fft_flat[b, neg_idx] # (N_neg, 68)
            
            pos_norm = F.normalize(pos_fft, p=2, dim=-1, eps=1e-5)
            neg_norm = F.normalize(neg_fft, p=2, dim=-1, eps=1e-5)
            
            neg_proto = neg_norm.mean(dim=0, keepdim=True) # (1, 68)
            sim = torch.sum(pos_norm * neg_proto, dim=-1) # (N_pos,)
            loss_b = F.relu(sim - 0.3).mean()
            losses.append(loss_b)
            
        if len(losses) == 0:
            loss_fcr = torch.tensor(0.0, device=corners.device)
        else:
            loss_fcr = torch.stack(losses).mean()
            
        return {'loss_fcr': loss_fcr}

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def _get_go_indices(self, indices, indices_aux_list):
        """Get a matching union set across all decoder layers. """
        results = []
        for indices_aux in indices_aux_list:
            indices = [(torch.cat([idx1[0], idx2[0]]), torch.cat([idx1[1], idx2[1]]))
                        for idx1, idx2 in zip(indices.copy(), indices_aux.copy())]

        for ind in [torch.cat([idx[0][:, None], idx[1][:, None]], 1) for idx in indices]:
            unique, counts = torch.unique(ind, return_counts=True, dim=0)
            count_sort_indices = torch.argsort(counts, descending=True)
            unique_sorted = unique[count_sort_indices]
            column_to_row = {}
            for idx in unique_sorted:
                row_idx, col_idx = idx[0].item(), idx[1].item()
                if row_idx not in column_to_row:
                    column_to_row[row_idx] = col_idx
            final_rows = torch.tensor(list(column_to_row.keys()), device=ind.device)
            final_cols = torch.tensor(list(column_to_row.values()), device=ind.device)
            results.append((final_rows.long(), final_cols.long()))
        return results

    def _clear_cache(self):
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.own_targets, self.own_targets_dn = None, None
        self.num_pos, self.num_neg = None, None

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'boxes': self.loss_boxes,
            'dssm': self.loss_dssm,
            'focal': self.loss_labels_focal,
            'vfl': self.loss_labels_vfl,
            'mal': self.loss_labels_mal,
            'tadl': self.loss_labels_tadl,
            'rs': self.loss_labels_rs,
            'local': self.loss_local,
            'fcr': self.loss_fcr,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, epoch=0, **kwargs):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets, epoch=epoch)['indices']
        self._clear_cache()

        # Get the matching union set across all decoder layers.
        if 'aux_outputs' in outputs:
            indices_aux_list, cached_indices, cached_indices_enc = [], [], []
            aux_outputs_list = outputs['aux_outputs']
            if 'pre_outputs' in outputs:
                aux_outputs_list = outputs['aux_outputs'] + [outputs['pre_outputs']]
            for i, aux_outputs in enumerate(aux_outputs_list):
                indices_aux = self.matcher(aux_outputs, targets, epoch=epoch)['indices']
                cached_indices.append(indices_aux)
                indices_aux_list.append(indices_aux)
            for i, aux_outputs in enumerate(outputs['enc_aux_outputs']):
                indices_enc = self.matcher(aux_outputs, targets, epoch=epoch)['indices']
                cached_indices_enc.append(indices_enc)
            indices_go = self._get_go_indices(indices, indices_aux_list)

            num_boxes_go = sum(len(x[0]) for x in indices_go)
            num_boxes_go = torch.as_tensor([num_boxes_go], dtype=torch.float, device=next(iter(outputs.values())).device)
            if is_dist_available_and_initialized():
                torch.distributed.all_reduce(num_boxes_go)
            num_boxes_go = torch.clamp(num_boxes_go / get_world_size(), min=1).item()
        else:
            assert 'aux_outputs' in outputs, ''

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses, main loss
        losses = {}
        for loss in self.losses:
            use_uni_set = self.use_uni_set and (loss in ['boxes', 'local'])
            indices_in = indices_go if use_uni_set else indices
            num_boxes_in = num_boxes_go if use_uni_set else num_boxes
            meta = self.get_loss_meta_info(loss, outputs, targets, indices_in)
            l_dict = self.get_loss(loss, outputs, targets, indices_in, num_boxes_in, **meta)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        if 'loss_cmal' in outputs:
            w_cmal = self.weight_dict.get('loss_cmal', 0.5)
            losses['loss_cmal'] = outputs['loss_cmal'] * w_cmal

        if 'loss_distill' in outputs:
            w_distill = self.weight_dict.get('loss_distill', 1.0)
            losses['loss_distill'] = outputs['loss_distill'] * w_distill

        if 'loss_co_cls' in outputs:
            w_co_cls = self.weight_dict.get('loss_co_cls', 1.0)
            losses['loss_co_cls'] = outputs['loss_co_cls'] * w_co_cls

        if 'loss_co_reg' in outputs:
            w_co_reg = self.weight_dict.get('loss_co_reg', 2.0)
            losses['loss_co_reg'] = outputs['loss_co_reg'] * w_co_reg

        if 'loss_physics' in outputs:
            w_physics = self.weight_dict.get('loss_physics', 1.0)
            losses['loss_physics'] = outputs['loss_physics'] * w_physics

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                if 'local' in self.losses:      # only work for local loss
                    aux_outputs['up'], aux_outputs['reg_scale'] = outputs['up'], outputs['reg_scale']
                for loss in self.losses:
                    use_uni_set = self.use_uni_set and (loss in ['boxes', 'local'])
                    indices_in = indices_go if use_uni_set else cached_indices[i]
                    num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_in)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_in, num_boxes_in, **meta)

                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of auxiliary traditional head output at first decoder layer. just for dfine
        if 'pre_outputs' in outputs:
            aux_outputs = outputs['pre_outputs']
            for loss in self.losses:
                use_uni_set = self.use_uni_set and (loss in ['boxes', 'local'])
                indices_in = indices_go if use_uni_set else cached_indices[-1]
                num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_in)
                l_dict = self.get_loss(loss, aux_outputs, targets, indices_in, num_boxes_in, **meta)

                l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                l_dict = {k + '_pre': v for k, v in l_dict.items()}
                losses.update(l_dict)

        # In case of encoder auxiliary losses.
        if 'enc_aux_outputs' in outputs:
            assert 'enc_meta' in outputs, ''
            class_agnostic = outputs['enc_meta']['class_agnostic']
            if class_agnostic:
                orig_num_classes = self.num_classes
                self.num_classes = 1
                enc_targets = copy.deepcopy(targets)
                for t in enc_targets:
                    t['labels'] = torch.zeros_like(t["labels"])
            else:
                enc_targets = targets

            for i, aux_outputs in enumerate(outputs['enc_aux_outputs']):
                for loss in self.losses:
                    indices_in = cached_indices_enc[i]
                    num_boxes_in = num_boxes
                    meta = self.get_loss_meta_info(loss, aux_outputs, enc_targets, indices_in)
                    l_dict = self.get_loss(loss, aux_outputs, enc_targets, indices_in, num_boxes_in, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

            if class_agnostic:
                self.num_classes = orig_num_classes

        # In case of cdn auxiliary losses.
        if 'dn_outputs' in outputs:
            assert 'dn_meta' in outputs, ''
            indices_dn = self.get_cdn_matched_indices(outputs['dn_meta'], targets)
            dn_num_boxes = num_boxes * outputs['dn_meta']['dn_num_group']

            for i, aux_outputs in enumerate(outputs['dn_outputs']):
                if 'local' in self.losses:      # only work for local loss
                    aux_outputs['is_dn'] = True
                    aux_outputs['up'], aux_outputs['reg_scale'] = outputs['up'], outputs['reg_scale']
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_dn)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

            # In case of auxiliary traditional head output at first decoder layer, just for dfine
            if 'dn_pre_outputs' in outputs:
                aux_outputs = outputs['dn_pre_outputs']
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_dn)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + '_dn_pre': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # For debugging Objects365 pre-train.
        losses = {k:torch.nan_to_num(v, nan=0.0) for k, v in losses.items()}
        return losses

    def _depth_weights(self, targets, indices, num_queries):
        """Per-query depth weight for DepthPrior (DLS + DLW).

        Returns a flat [B*num_queries] tensor: matched queries get the depth-bin
        weight of their target (far/small -> heavier), unmatched queries get 1.0.
        The classification losses are computed over all queries, so the weight
        must be per-query; the box losses gather it via the matched indices.
        Returns None when depth prior is off or no depth is available.
        """
        if not self.use_depth_prior:
            return None
        dev = None
        for t in targets:
            d = t.get('depth', None)
            if d is not None and d.numel() > 0:
                dev = d.device
                break
        if dev is None:
            return None
        parts = []
        for t, (qidx, gidx) in zip(targets, indices):
            wq = torch.ones(num_queries, device=dev, dtype=torch.float32)
            d = t.get('depth', None)
            if d is not None and gidx.numel() > 0:
                dlen = d.shape[0]
                if gidx.max() >= dlen:
                    # gidx are GT indices and must be < dlen (= num GT boxes); if not,
                    # something is misaligned -- report once and clamp defensively.
                    if not getattr(type(self), '_depth_dbg', False):
                        print(f"[DEPTHPRIOR] gidx.max={int(gidx.max())} dlen={int(dlen)} "
                              f"qidx.max={int(qidx.max())} num_queries={num_queries} "
                              f"num_gt={int(t['labels'].shape[0])}", flush=True)
                        type(self)._depth_dbg = True
                    gidx = gidx.clamp(0, dlen - 1)
                raw_d = d[gidx].float().to(dev)
                dt = 1.0 - raw_d if self.depth_invert else raw_d
                wt = torch.full_like(dt, float(self.depth_weights[0]))
                for thr, wval in zip(self.depth_bins, self.depth_weights[1:]):
                    wt = torch.where(dt >= float(thr), torch.full_like(dt, float(wval)), wt)
                # raw depth == 0 usually means "no depth data" -> keep neutral
                wt = torch.where(raw_d < 1e-3, torch.ones_like(wt), wt)
                # CDN (denoising) query counts can occasionally disagree with the
                # matcher's index range; clamp to stay in-bounds (a few auxiliary
                # queries get a slightly wrong weight, the main output is untouched).
                wq[qidx.clamp(0, num_queries - 1)] = wt
            parts.append(wq)
        return torch.cat(parts)   # [B*num_queries]

    def get_loss_meta_info(self, loss, outputs, targets, indices):
        meta = {}
        if self.use_depth_prior:
            # The matcher's src indices address either the main query set
            # (outputs['pred_logits'], num_queries) or the denoising query set
            # (outputs['pred_logits_dn'], a larger count). Size the per-query
            # weight tensor to the set these indices actually reference, else
            # the scatter `wq[src] = wt` goes out of bounds on DN queries.
            main_n = outputs['pred_logits'].shape[1]
            dn_n = outputs['pred_logits_dn'].shape[1] if 'pred_logits_dn' in outputs else 0
            max_src = -1
            for (_, j) in indices:
                if j.numel() > 0:
                    max_src = max(max_src, int(j.max().item()))
            nq = dn_n if (dn_n > 0 and max_src >= main_n) else main_n
            meta['depth_weight'] = self._depth_weights(targets, indices, nq)

        if self.boxes_weight_format is None:
            return meta

        src_boxes = outputs['pred_boxes'][self._get_src_permutation_idx(indices)]
        target_boxes = torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, indices)], dim=0)

        if self.boxes_weight_format == 'iou':
            iou, _ = box_iou(box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes))
            iou = torch.diag(iou)
        elif self.boxes_weight_format == 'giou':
            iou = torch.diag(generalized_box_iou(\
                box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes)))
        else:
            raise AttributeError()

        if loss in ('boxes', ):
            meta['boxes_weight'] = iou
        elif loss in ('vfl', 'mal', 'rs', 'tadl'):
            meta['values'] = iou

        return meta

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        """get_cdn_matched_indices
        """
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        num_gts = [len(t['labels']) for t in targets]
        device = targets[0]['labels'].device

        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros(0, dtype=torch.int64, device=device), \
                    torch.zeros(0, dtype=torch.int64,  device=device)))

        return dn_match_indices


    def feature_loss_function(self, fea, target_fea):
        loss = (fea - target_fea) ** 2 * ((fea > 0) | (target_fea > 0)).float()
        return torch.abs(loss)


    def unimodal_distribution_focal_loss(self, pred, label, weight_right, weight_left, weight=None, reduction='sum', avg_factor=None):
        n_classes = pred.shape[1]
        label = torch.nan_to_num(label, nan=0.0, posinf=float(n_classes - 1), neginf=0.0)
        dis_left = label.clamp(0, n_classes - 1).long().clamp(0, n_classes - 1)
        dis_right = (dis_left + 1).clamp(0, n_classes - 1)

        w_l = torch.nan_to_num(weight_left, nan=0.0).reshape(-1)
        w_r = torch.nan_to_num(weight_right, nan=0.0).reshape(-1)

        pred_f32 = pred.float().clamp(-20.0, 20.0).contiguous()
        dis_left = dis_left.contiguous()
        dis_right = dis_right.contiguous()
        loss = F.cross_entropy(pred_f32, dis_left, reduction='none') * w_l \
             + F.cross_entropy(pred_f32, dis_right, reduction='none') * w_r

        if weight is not None:
            weight = torch.nan_to_num(weight.float(), nan=0.0)
            loss = loss * weight

        if avg_factor is not None:
            loss = loss.sum() / avg_factor
        elif reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()

        return loss

    def get_gradual_steps(self, outputs):
        num_layers = len(outputs['aux_outputs']) + 1 if 'aux_outputs' in outputs else 1
        step = .5 / (num_layers - 1)
        opt_list = [.5  + step * i for i in range(num_layers)] if num_layers > 1 else [1]
        return opt_list
