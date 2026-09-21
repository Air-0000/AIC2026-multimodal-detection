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
        'remap_mscoco_category'
    ]

    def __init__(
        self,
        num_classes=80,
        use_focal_loss=True,
        num_top_queries=300,
        remap_mscoco_category=False,
        depth_cond_enable=False,
        depth_cond_far=0.5,
        depth_cond_boost=1.5,
        # AIC depth is now read as ABSOLUTE depth (mm/19999): near = LOW value,
        # far = HIGH value, so NO inversion is needed for the far-object boost.
        # Only set True when re-scoring a legacy checkpoint that was trained on
        # the old clipped (binary) maps, where the convention was inverted.
        depth_invert=False,
    ) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.remap_mscoco_category = remap_mscoco_category
        self.deploy_mode = False
        # DepthPrior DCT: boost far (deep) predictions' score so they survive the
        # global confidence threshold at night. depth_cond_enable flips it on; the
        # std guard in _dct_boost auto-disables it when depth is blacked (ablk).
        self.depth_cond_enable = depth_cond_enable
        self.depth_cond_far = float(depth_cond_far)
        self.depth_cond_boost = float(depth_cond_boost)
        self.depth_invert = depth_invert

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
        dep = samples[:, 4:5]                       # [B,1,H,W] normalized depth
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
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
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

        if self.deploy_mode:
            return labels, boxes, scores

        if self.remap_mscoco_category:
            from ..data.dataset import mscoco_label2category
            labels = torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()])\
                .to(boxes.device).reshape(labels.shape)

        results = []
        for lab, box, sco in zip(labels, boxes, scores):
            result = dict(labels=lab, boxes=box, scores=sco)
            results.append(result)

        return results


    def deploy(self, ):
        self.eval()
        self.deploy_mode = True
        return self
