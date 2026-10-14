"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/util/box_ops.py
"""

import torch
from torch import Tensor
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w.clamp(min=0.0)), (y_c - 0.5 * h.clamp(min=0.0)),
         (x_c + 0.5 * w.clamp(min=0.0)), (y_c + 0.5 * h.clamp(min=0.0))]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x: Tensor) -> Tensor:
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def _sanitize_boxes(boxes: Tensor) -> Tensor:
    """Make boxes safe for IoU/GIoU: drop NaN/Inf and enforce x0<=x1, y0<=y1.

    Degenerate predicted boxes (negative width/height, or NaN from random init /
    denoising queries) would otherwise trigger assertions or produce inf/nan in
    the matching cost at the very first training step. Clamping here lets
    training proceed; the matcher already converts any residual NaN cost via
    nan_to_num.
    """
    boxes = torch.nan_to_num(boxes, nan=0.0, posinf=0.0, neginf=0.0)
    x0, y0 = boxes[:, 0:1], boxes[:, 1:2]
    x1, y1 = boxes[:, 2:3], boxes[:, 3:4]
    xmin = torch.min(x0, x1)
    xmax = torch.max(x0, x1)
    ymin = torch.min(y0, y1)
    ymax = torch.max(y0, y1)
    return torch.cat([xmin, ymin, xmax, ymax], dim=-1)


def box_iou(boxes1: Tensor, boxes2: Tensor):
    boxes1 = _sanitize_boxes(boxes1)
    boxes2 = _sanitize_boxes(boxes2)
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union.clamp(min=1e-6)
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate / NaN boxes (e.g. from random init at step 1 or denoising
    # queries) used to assert here and abort training; clamp instead so the
    # matching cost stays finite and training can proceed.
    boxes1 = _sanitize_boxes(boxes1)
    boxes2 = _sanitize_boxes(boxes2)
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = (wh[:, :, 0] * wh[:, :, 1]).clamp(min=1e-6)

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)


def box_nwd(boxes1, boxes2, C=0.5):
    """
    Normalized Gaussian Wasserstein Distance (NWD) for tiny and small object detection.
    Reference: Wang et al., 'A Normalized Gaussian Wasserstein Distance for Tiny Object Detection', NeurIPS 2021.
    Supports:
      - Pairwise: boxes1 [N, 4], boxes2 [M, 4] -> returns [N, M]
      - Element-wise: boxes1 [N, 4], boxes2 [N, 4] -> returns [N]
      - Arbitrary broadcastable shapes [..., 4]
    Boxes must be in (cx, cy, w, h) normalized format.
    """
    if boxes1.ndim == 2 and boxes2.ndim == 2 and boxes1.shape[0] != boxes2.shape[0]:
        cx1, cy1, w1, h1 = boxes1[:, 0:1], boxes1[:, 1:2], boxes1[:, 2:3], boxes1[:, 3:4]
        cx2, cy2, w2, h2 = boxes2[None, :, 0], boxes2[None, :, 1], boxes2[None, :, 2], boxes2[None, :, 3]
        w2_dist = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2 + ((w1 - w2) ** 2 + (h1 - h2) ** 2) / 4.0
        return torch.exp(-torch.sqrt(w2_dist.clamp(min=1e-8)) / C)
    else:
        cx1, cy1, w1, h1 = boxes1.unbind(-1)
        cx2, cy2, w2, h2 = boxes2.unbind(-1)
        w2_dist = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2 + ((w1 - w2) ** 2 + (h1 - h2) ** 2) / 4.0
        return torch.exp(-torch.sqrt(w2_dist.clamp(min=1e-8)) / C)