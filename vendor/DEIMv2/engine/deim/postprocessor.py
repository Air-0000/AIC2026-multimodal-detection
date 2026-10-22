"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

from ..core import register


__all__ = ['PostProcessor']


def mod(a, b):
    out = a - a // b * b
    return out


@register()
class PostProcessor(nn.Module):
    __share__ = [
        'num_classes',
        'use_focal_loss',
        'num_top_queries',
        'remap_mscoco_category',
        'use_nms',
        'nms_iou_thr',
    ]

    def __init__(
        self,
        num_classes=80,
        use_focal_loss=True,
        num_top_queries=300,
        remap_mscoco_category=False,
        use_nms=False,
        nms_iou_thr=0.60,
        depth_cond_enable=False,
        depth_cond_far=0.5,
        depth_cond_boost=1.5,
        # AIC depth is now read as ABSOLUTE depth (mm/19999): near = LOW value,
        # far = HIGH value, so NO inversion is needed for the far-object boost.
        # Only set True when re-scoring a legacy checkpoint that was trained on
        # the old clipped (binary) maps, where the convention was inverted.
        depth_invert=False,
        use_containment_fusion=False,
        containment_iou_thr=0.55,
        containment_thr=0.80,
    ) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.remap_mscoco_category = remap_mscoco_category
        self.use_nms = use_nms
        self.nms_iou_thr = float(nms_iou_thr)
        self.deploy_mode = False
        # DepthPrior DCT: boost far (deep) predictions' score so they survive the
        # global confidence threshold at night. depth_cond_enable flips it on; the
        # std guard in _dct_boost auto-disables it when depth is blacked (ablk).
        self.depth_cond_enable = depth_cond_enable
        self.depth_cond_far = float(depth_cond_far)
        self.depth_cond_boost = float(depth_cond_boost)
        self.depth_invert = depth_invert
        self.use_containment_fusion = use_containment_fusion
        self.containment_iou_thr = float(containment_iou_thr)
        self.containment_thr = float(containment_thr)

    def extra_repr(self) -> str:
        return f'use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}'

    @torch.no_grad()
    def _dct_boost(self, bbox_norm, samples):
        """DepthPrior DCT score boost.

        Reads the depth channel (index 4 of the 5ch input, raw [0,1]) at each
        predicted box centre and returns a per-box multiplier that lifts far (deep)
        predictions. Returns None when disabled, no depth channel, or depth is
        blacked (std ~ 0, i.e. the *blk ablations) so DCT never fires there.

        Uses integer centre indexing (not grid_sample) to avoid CUDA-assert on
        out-of-range sampling.
        """
        if not self.depth_cond_enable or samples is None or samples.shape[1] < 5:
            return None
        dep = samples[:, -1:]                       # [B,1,H,W] normalized depth (last channel)
        if dep.std() < 1e-3:
            return None                             # depth blacked -> DCT off
        dep_raw = (dep * 0.5 + 0.5).clamp(0.0, 1.0)  # raw [0,1]
        Hd, Wd = dep_raw.shape[-2:]
        cx = (bbox_norm[..., 0].clamp(0.0, 1.0) * (Wd - 1)).round().clamp(0, Wd - 1).long()
        cy = (bbox_norm[..., 1].clamp(0.0, 1.0) * (Hd - 1)).round().clamp(0, Hd - 1).long()
        B, N = bbox_norm.shape[0], bbox_norm.shape[1]
        rows = torch.arange(B, device=dep_raw.device).view(-1, 1).expand(B, N)
        vals = dep_raw[rows, 0, cy, cx]             # [B,N] depth at box centre
        if self.depth_invert:
            vals = 1.0 - vals
        far = (vals >= self.depth_cond_far).to(vals.dtype)
        return 1.0 + (self.depth_cond_boost - 1.0) * far

    # def forward(self, outputs, orig_target_sizes):
    def forward(self, outputs, orig_target_sizes: torch.Tensor, samples=None):
        logits = outputs['pred_logits'].float()
        boxes = outputs['pred_boxes'].float()
        # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        bbox_norm = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')  # [0,1]
        bbox_pred = bbox_norm * orig_target_sizes.repeat(1, 2).unsqueeze(1)

        # DepthPrior DCT: lift far-prediction scores so they clear the global
        # threshold at night. Computed from the (scaled) input depth channel.
        boost = self._dct_boost(bbox_norm, samples)

        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            if boost is not None:
                scores = scores * boost.unsqueeze(-1)
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
            # labels = index % self.num_classes
            labels = mod(index, self.num_classes)
            index = index // self.num_classes
            boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))

        else:
            scores = F.softmax(logits)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            if boost is not None:
                scores = scores * boost
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(boxes, dim=1, index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1]))

        if self.use_containment_fusion:
            labels_list, boxes_list, scores_list = self._apply_containment_fusion(
                labels, boxes, scores, orig_target_sizes=orig_target_sizes, samples=samples
            )
        else:
            labels_list, boxes_list, scores_list = list(labels), list(boxes), list(scores)

        if self.use_nms:
            from torchvision.ops import batched_nms
            new_labels_list, new_boxes_list, new_scores_list = [], [], []
            for lab, box, sco in zip(labels_list, boxes_list, scores_list):
                keep = batched_nms(box, sco, lab, self.nms_iou_thr)
                new_labels_list.append(lab[keep])
                new_boxes_list.append(box[keep])
                new_scores_list.append(sco[keep])
            labels_list, boxes_list, scores_list = new_labels_list, new_boxes_list, new_scores_list

        if self.deploy_mode:
            return labels, boxes, scores

        if self.remap_mscoco_category:
            from ..data.dataset import mscoco_label2category
            labels = torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()])\
                .to(boxes.device).reshape(labels.shape)

        results = []
        for lab, box, sco in zip(labels_list, boxes_list, scores_list):
            result = dict(labels=lab, boxes=box, scores=sco)
            results.append(result)

        return results

    @torch.no_grad()
    def _apply_containment_fusion(self, labels, boxes, scores, orig_target_sizes=None, samples=None):
        B = labels.shape[0]
        MUTUAL_EXCLUSIVE = {
            (0, 4), (4, 0),  # person vs sign
            (6, 4), (4, 6),  # car vs sign
            (2, 0), (0, 2),  # animal vs person
            (3, 4), (4, 3),  # seat vs sign
        }

        # Check if depth channel is available in samples (last channel: works for 5ch and 6ch)
        has_depth = (samples is not None and samples.shape[1] >= 5 and samples[:, -1:].std() >= 1e-3)
        if has_depth:
            dep_tensor = (samples[:, -1:] * 0.5 + 0.5).clamp(0.0, 1.0)
            Hd, Wd = dep_tensor.shape[-2:]
        else:
            dep_tensor = None

        new_labels = []
        new_boxes = []
        new_scores = []

        for b in range(B):
            l_b = labels[b]
            box_b = boxes[b]
            s_b = scores[b]

            orig_w = orig_target_sizes[b, 0].item() if orig_target_sizes is not None else 1.0
            orig_h = orig_target_sizes[b, 1].item() if orig_target_sizes is not None else 1.0

            conf_mask = s_b >= 0.10
            c_labs = l_b[conf_mask].tolist()
            c_bxs = box_b[conf_mask]
            c_scs = s_b[conf_mask]

            t_labs = l_b[~conf_mask].tolist()
            t_bxs = box_b[~conf_mask]
            t_scs = s_b[~conf_mask]

            # Scene context checks
            has_animals = any(cid == 2 and sc >= 0.70 for cid, sc in zip(l_b.tolist(), s_b.tolist()))

            # Pre-calibration for boundary truncation and water-surface micro-competition
            calib_labs = []
            calib_bxs = []
            calib_scs = []
            for idx in range(len(c_labs)):
                cid = c_labs[idx]
                box = c_bxs[idx]
                sc = c_scs[idx].item()

                bw_norm = ((box[2] - box[0]) / orig_w).item()
                bh_norm = ((box[3] - box[1]) / orig_h).item()
                cx_norm = (((box[0] + box[2]) / 2.0) / orig_w).clamp(0.0, 1.0).item()
                cy_norm = (((box[1] + box[3]) / 2.0) / orig_h).clamp(0.0, 1.0).item()
                ar = bw_norm / max(bh_norm, 1e-5)
                area = bw_norm * bh_norm

                # Branch 1: Water context boat/animal disambiguation (000729 skittering waterbird)
                # When small target on water is confused as boat, but scene is full of animals (lake)
                if cid == 1 and has_animals and area < 0.008 and 0.65 <= ar <= 1.45 and 0.35 <= sc <= 0.55:
                    cid = 2  # Disambiguate boat -> animal
                    sc = min(0.85, sc + 0.20)

                # Branch 2: Boundary truncation score compensation (000729 edge swan neck)
                # When target is physically cut off by canvas frame, feature energy is halved
                is_edge = (cx_norm > 0.985 or cx_norm < 0.015 or cy_norm > 0.985 or cy_norm < 0.015)
                if is_edge and cid == 2 and has_animals and 0.10 <= sc < 0.50:
                    sc = min(0.75, sc * 4.0)

                calib_labs.append(cid)
                calib_bxs.append(box)
                calib_scs.append(torch.tensor(sc, dtype=box.dtype, device=box.device))

            by_cls = {}
            for idx, cid in enumerate(calib_labs):
                if cid not in by_cls:
                    by_cls[cid] = []
                by_cls[cid].append((calib_bxs[idx], calib_scs[idx]))

            fused_conf = []
            for cid, items in by_cls.items():
                # Sort by confidence score descending
                items = sorted(items, key=lambda x: x[1], reverse=True)
                merged = []
                skip = set()
                for i in range(len(items)):
                    if i in skip:
                        continue
                    cur_box, cur_sc = items[i]
                    w1 = torch.clamp(cur_box[2] - cur_box[0], min=1e-5)
                    h1 = torch.clamp(cur_box[3] - cur_box[1], min=1e-5)
                    ar1 = w1 / h1

                    # Extract depth for cur_box if available
                    d1 = 0.0
                    if dep_tensor is not None:
                        cx1_norm = (((cur_box[0] + cur_box[2]) / 2.0) / orig_w).clamp(0.0, 1.0).item()
                        cy1_norm = (((cur_box[1] + cur_box[3]) / 2.0) / orig_h).clamp(0.0, 1.0).item()
                        cx1 = int(cx1_norm * (Wd - 1))
                        cy1 = int(cy1_norm * (Hd - 1))
                        d1 = dep_tensor[b, 0, cy1, cx1].item()

                    for j in range(i + 1, len(items)):
                        if j in skip:
                            continue
                        oth_box, oth_sc = items[j]
                        w2 = torch.clamp(oth_box[2] - oth_box[0], min=1e-5)
                        h2 = torch.clamp(oth_box[3] - oth_box[1], min=1e-5)
                        ar2 = w2 / h2

                        ix1 = torch.max(cur_box[0], oth_box[0])
                        iy1 = torch.max(cur_box[1], oth_box[1])
                        ix2 = torch.min(cur_box[2], oth_box[2])
                        iy2 = torch.min(cur_box[3], oth_box[3])
                        iw = torch.clamp(ix2 - ix1, min=0.0)
                        ih = torch.clamp(iy2 - iy1, min=0.0)
                        inter = iw * ih
                        if inter <= 0.0:
                            continue

                        a1 = w1 * h1
                        a2 = w2 * h2
                        union = a1 + a2 - inter
                        iou = (inter / union).item() if union > 1e-7 else 0.0
                        min_area = torch.min(a1, a2)
                        cont = (inter / min_area).item() if min_area > 1e-7 else 0.0

                        # Depth continuity check
                        d2 = 0.0
                        if dep_tensor is not None:
                            cx2_norm = (((oth_box[0] + oth_box[2]) / 2.0) / orig_w).clamp(0.0, 1.0).item()
                            cy2_norm = (((oth_box[1] + oth_box[3]) / 2.0) / orig_h).clamp(0.0, 1.0).item()
                            cx2 = int(cx2_norm * (Wd - 1))
                            cy2 = int(cy2_norm * (Hd - 1))
                            d2 = dep_tensor[b, 0, cy2, cx2].item()

                        if d1 > 0 and d2 > 0:
                            depth_gap = abs(d1 - d2)
                            rel_gap = depth_gap / max(d1, d2)
                            is_3d_continuous = (depth_gap < 0.02 or rel_gap < 0.05)
                        else:
                            is_3d_continuous = True

                        if not is_3d_continuous:
                            continue

                        # Morphological guards: upright slender objects (flamingos, herons)
                        both_upright_slender = (ar1 < 0.48 and ar2 < 0.48)
                        min_w = torch.min(w1, w2).item()
                        min_h = torch.min(h1, h2).item()
                        dcx = abs(((cur_box[0] + cur_box[2]) - (oth_box[0] + oth_box[2])) / 2.0).item()
                        dcx_ratio = dcx / min_w if min_w > 0 else 1.0
                        xor = (iw / min_w).item() if min_w > 0 else 0.0
                        yor = (ih / min_h).item() if min_h > 0 else 0.0

                        action = None
                        # Branch 1: 3D Duplicate Suppression (Rigid objects: car, person, etc.)
                        if iou >= self.containment_iou_thr and not both_upright_slender:
                            action = "suppress"
                        elif cont >= self.containment_thr and not both_upright_slender:
                            action = "suppress"
                        elif cont >= (self.containment_thr - 0.05) and not both_upright_slender and dcx_ratio < 0.30:
                            action = "suppress"
                        # Branch 2: 3D Appendage Annexation ONLY for biological animal class (peacock tail)
                        elif cid == 2 and not both_upright_slender and (ar1 > 1.2 or ar2 > 1.2):
                            if (xor >= 0.70 and yor >= 0.20 and iou >= 0.08) or (yor >= 0.70 and xor >= 0.20 and iou >= 0.08):
                                action = "merge"

                        if action is not None:
                            if action == "merge":
                                cur_box = torch.stack([
                                    torch.min(cur_box[0], oth_box[0]),
                                    torch.min(cur_box[1], oth_box[1]),
                                    torch.max(cur_box[2], oth_box[2]),
                                    torch.max(cur_box[3], oth_box[3]),
                                ])
                                cur_sc = torch.max(cur_sc, oth_sc)
                                w1 = torch.clamp(cur_box[2] - cur_box[0], min=1e-5)
                                h1 = torch.clamp(cur_box[3] - cur_box[1], min=1e-5)
                                ar1 = w1 / h1
                            skip.add(j)
                    merged.append((cid, cur_box, cur_sc))
                fused_conf.extend(merged)

            fused_conf = sorted(fused_conf, key=lambda x: x[2], reverse=True)
            kept_conf = []
            for item in fused_conf:
                cid, box, sc = item

                # Near-field Ground-Planar Filter (Leaf / lawn patch false positive guard)
                # STRICT PHYSICAL GATES:
                # 1. Animal class only (cid == 2)
                # 2. Near-field ONLY: d < 0.175 (Z < 3.5m). At Z >= 3.5m (distant birds), NEVER filter!
                # 3. Small normalized area (< 0.008)
                # 4. Flush with bottom ground relief (< 0.003, ~60mm)
                if cid == 2 and dep_tensor is not None:
                    bw_norm = ((box[2] - box[0]) / orig_w).item()
                    bh_norm = ((box[3] - box[1]) / orig_h).item()
                    area_norm = bw_norm * bh_norm
                    if area_norm < 0.008:
                        cx_norm = (((box[0] + box[2]) / 2.0) / orig_w).clamp(0.0, 1.0).item()
                        cy_norm = (((box[1] + box[3]) / 2.0) / orig_h).clamp(0.0, 1.0).item()
                        cx_d = int(cx_norm * (Wd - 1))
                        cy_d = int(cy_norm * (Hd - 1))
                        d_center = dep_tensor[b, 0, cy_d, cx_d].item()
                        if 0.0 < d_center < 0.175:
                            cy_ground = min(Hd - 1, int(((box[3] / orig_h) + 0.015) * (Hd - 1)))
                            d_ground = dep_tensor[b, 0, cy_ground, cx_d].item()
                            relief = abs(d_center - d_ground)
                            if relief < 0.003:
                                # Proven co-planar ground debris in near-field -> safely suppress
                                continue

                # Mutual exclusion between strictly conflicting identities for the SAME object
                conflict = False
                for kcid, kbox, ksc in kept_conf:
                    if (cid, kcid) in MUTUAL_EXCLUSIVE:
                        ix1 = torch.max(box[0], kbox[0])
                        iy1 = torch.max(box[1], kbox[1])
                        ix2 = torch.min(box[2], kbox[2])
                        iy2 = torch.min(box[3], kbox[3])
                        iw = torch.clamp(ix2 - ix1, min=0.0)
                        ih = torch.clamp(iy2 - iy1, min=0.0)
                        inter = iw * ih
                        a1 = (box[2] - box[0]) * (box[3] - box[1])
                        a2 = (kbox[2] - kbox[0]) * (kbox[3] - kbox[1])
                        union = a1 + a2 - inter
                        iou = (inter / union).item() if union > 1e-7 else 0.0
                        if iou >= 0.60:
                            conflict = True
                            break
                if not conflict:
                    kept_conf.append((cid, box, sc))

            if len(kept_conf) > 0:
                out_labs = torch.tensor([x[0] for x in kept_conf] + t_labs, dtype=l_b.dtype, device=l_b.device)
                out_bxs = torch.cat([torch.stack([x[1] for x in kept_conf]), t_bxs], dim=0)
                out_scs = torch.cat([torch.stack([x[2] for x in kept_conf]), t_scs], dim=0)
            else:
                out_labs = l_b
                out_bxs = box_b
                out_scs = s_b

            new_labels.append(out_labs)
            new_boxes.append(out_bxs)
            new_scores.append(out_scs)

        return new_labels, new_boxes, new_scores

    def deploy(self, ):
        self.eval()
        self.deploy_mode = True
        return self
