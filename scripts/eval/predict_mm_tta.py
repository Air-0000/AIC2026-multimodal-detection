"""Multi-modal (6ch) DEIMv2 test-set inference with TTA (hflip x multiscale) + WBF.

Channel layout MUST match training (see CocoDetectionMultiModal.load_item):
    ch0..2  RGB      (visible)
    ch3     heat     (work_dir/deimv2_heat/<split>  -- IR residual, 1ch)
    ch4     raw_ir   (work_dir/deimv2_ir/<split>    -- aligned raw IR, 1ch)
    ch5     depth    (<test_root>/depth             -- 16-bit mm / 19999, or 8-bit / 255)

Depth is read with the SAME convention as training (`depth_read='raw16'`):
Pillow mode 'I;16'/'I' -> /19999 (millimetres); anything else -> /255. Never use
PIL convert("L") here -- it clips 16-bit mm depth and collapses the map to a
binary mask (the historic "depth is dead weight" bug).

Views: scales x flips; boxes from every view are merged with WBF and written in
the official AIC format (<class_id> <cx> <cy> <w> <h> <conf>, normalized, CRLF,
<=100 boxes/image).

Usage:
  python predict_mm_tta.py --test-root dataset/test/AIC2026_PHASE_2_1000 \
      --out work_dir/submission_phase2_l4u3_tta --scales 640,800 --thresh 0.3
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
DEIMV2_ROOT = os.path.join(ROOT, "vendor/DEIMv2")
# engine.data.transforms imports the top-level module `nightify_aic`, which lives
# in scripts/lib -- DEIMv2 alone is not enough on sys.path.
sys.path.insert(0, os.path.join(ROOT, "scripts", "lib"))
sys.path.insert(0, DEIMV2_ROOT)
os.chdir(DEIMV2_ROOT)
from engine.core import YAMLConfig

LINESEP = "\r\n"
MAX_BOXES = 100
# must match the training config's Normalize op (6 channels)
MEAN = [0.485, 0.456, 0.406, 0.5, 0.5, 0.5]
STD = [0.229, 0.224, 0.225, 0.5, 0.5, 0.5]


def read_gray_01(folder, name):
    """1ch float32 in [0,1]; mirrors CocoDetectionMultiModal._read_channel(folder,name,1)."""
    base = os.path.splitext(name)[0]
    p_png = os.path.join(folder, base + ".png")
    path = p_png if os.path.exists(p_png) else os.path.join(folder, name)
    im = Image.open(path).convert("L")
    return TF.pil_to_tensor(im).float() / 255.0  # [1,H,W]


def read_depth_01(folder, name):
    """1ch float32 in [0,1]; mirrors _read_depth_raw16 (16-bit mm / 19999, else /255)."""
    base = os.path.splitext(name)[0]
    for ext in (".png", ".jpg", ".jpeg"):
        p = os.path.join(folder, base + ext)
        if os.path.exists(p):
            im = Image.open(p)
            arr = np.array(im).astype(np.float32)
            if arr.ndim == 3:
                arr = arr.mean(axis=2)
            scale = 19999.0 if getattr(im, "mode", "") in ("I;16", "I") else 255.0
            return torch.from_numpy(np.clip(arr / scale, 0.0, 1.0))[None]
    raise FileNotFoundError(os.path.join(folder, base + ".(png|jpg)"))


def build_6ch(rgb_pil, heat_dir, rawir_dir, depth_dir, name):
    """Returns [6,H,W] float in [0,1]; aux channels are resized to the RGB size
    exactly like the training dataset does."""
    size_hw = rgb_pil.size[::-1]  # (H, W)
    rgb = TF.pil_to_tensor(rgb_pil.convert("RGB")).float() / 255.0
    heat = read_gray_01(heat_dir, name)
    raw_ir = read_gray_01(rawir_dir, name)
    dep = read_depth_01(depth_dir, name)
    heat = TF.resize(heat, size_hw)
    raw_ir = TF.resize(raw_ir, size_hw)
    dep = TF.resize(dep, size_hw)
    return torch.cat([rgb, heat, raw_ir, dep], dim=0)


def to_model_input(x, size):
    """Resize to (size,size) then apply the 6-channel Normalize."""
    x = TF.resize(x, [size, size])
    mean = torch.tensor(MEAN, dtype=x.dtype).view(-1, 1, 1)
    std = torch.tensor(STD, dtype=x.dtype).view(-1, 1, 1)
    return (x - mean) / std


def iou(b1, b2):
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (b1[2] - b1[0]) * (b1[3] - b1[1]) + (b2[2] - b2[0]) * (b2[3] - b2[1]) - inter
    return inter / ua if ua > 0 else 0.0


def wbf(all_boxes, iou_thr=0.55, n_views=1, conf_mode="avg"):
    """all_boxes: list of (cls, [x1,y1,x2,y2], conf). Weighted box fusion.

    conf_mode:
      avg -- standard WBF mean confidence, damped by view coverage.
      sat -- the legacy ``sum / sqrt(n)`` rule, kept ONLY as a negative control:
             it saturates to 1.0 as soon as a box is seen in more than one view
             (4 views x conf 0.9 -> 1.8), flattening every score and destroying
             the confidence ranking that mAP depends on.
    """
    fused = []
    for cls in sorted(set(b[0] for b in all_boxes)):
        members = sorted([b for b in all_boxes if b[0] == cls], key=lambda b: b[2], reverse=True)
        clusters = []
        for box in members:
            for cl in clusters:
                if iou(box[1], cl[0][1]) >= iou_thr:
                    cl.append(box)
                    break
            else:
                clusters.append([box])
        for cl in clusters:
            confs = np.array([b[2] for b in cl])
            ws = confs / confs.sum()
            cx = sum(b[1][0] * w for b, w in zip(cl, ws))
            cy = sum(b[1][1] * w for b, w in zip(cl, ws))
            x2 = sum(b[1][2] * w for b, w in zip(cl, ws))
            y2 = sum(b[1][3] * w for b, w in zip(cl, ws))
            if conf_mode == "sat":
                c = float(np.clip(confs.sum() / (len(cl) ** 0.5), 0, 1))
            else:
                # avg confidence, damped by how many of the views actually saw the box
                c = float(np.clip(confs.mean() * (len(cl) / max(n_views, 1)) ** 0.25, 0, 1))
            fused.append((cls, [cx, cy, x2, y2], c))
    return fused


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/deimv2/"
                    "deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir_midboost_l4u3.yml")
    ap.add_argument("--ckpt", default=os.path.join(
        ROOT, "vendor/DEIMv2/outputs/"
        "deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir_midboost_l4u3/best_stg1.pth"))
    ap.add_argument("--test-root", default=os.path.join(ROOT, "dataset/test/AIC2026_PHASE_2_1000"))
    ap.add_argument("--split", default="phase2",
                    help="sub-folder name of work_dir/deimv2_heat and deimv2_ir")
    ap.add_argument("--out", default=os.path.join(ROOT, "work_dir/submission_phase2_l4u3_tta"))
    ap.add_argument("--thresh", type=float, default=0.1,
                    help="drop FUSED boxes below this (validated optimum 0.1)")
    ap.add_argument("--pre-thresh", type=float, default=0.05,
                    help="drop per-view boxes below this BEFORE fusion")
    ap.add_argument("--scales", default="640,800")
    ap.add_argument("--no-flip", action="store_true",
                    help="disable the horizontal-flip view (use with --scales 640 for a no-TTA run)")
    ap.add_argument("--iou-thr", type=float, default=0.7,
                    help="WBF cluster IoU; 0.7 validated better than 0.55")
    ap.add_argument("--conf-mode", choices=["avg", "max", "sat"], default="max",
                    help="fused confidence rule; 'max' beat 'avg' by +0.0095 AP on val")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    # the module-level os.chdir(DEIMV2_ROOT) already happened, so a relative
    # --out would land inside vendor/DEIMv2 -- resolve it against ROOT.
    args.out = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    args.test_root = (args.test_root if os.path.isabs(args.test_root)
                      else os.path.join(ROOT, args.test_root))
    scales = [int(s) for s in args.scales.split(",")]
    flips = [False] if args.no_flip else [False, True]
    vis_dir = os.path.join(args.test_root, "visible")
    depth_dir = os.path.join(args.test_root, "depth")
    heat_dir = os.path.join(ROOT, "work_dir/deimv2_heat", args.split)
    rawir_dir = os.path.join(ROOT, "work_dir/deimv2_ir", args.split)
    os.makedirs(args.out, exist_ok=True)

    cfg = YAMLConfig(args.config)
    ck = torch.load(args.ckpt, map_location="cpu")
    sd = ck["ema"]["module"] if "ema" in ck else ck.get("model", ck)
    missing, unexpected = cfg.model.load_state_dict(sd, strict=False)
    print(f"[mm-tta] loaded {args.ckpt}  missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig):
            return self.postprocessor(self.model(images), orig)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    infer = M().to(device).eval()

    imgs = sorted(glob.glob(os.path.join(vis_dir, "*")))
    if args.limit:
        imgs = imgs[: args.limit]
    print(f"[mm-tta] {len(imgs)} images | scales={scales} flips={flips} "
          f"thresh={args.thresh} device={device}", flush=True)

    with torch.no_grad():
        for i, imp in enumerate(imgs):
            name = os.path.basename(imp)
            stem = os.path.splitext(name)[0]
            rgb_pil = Image.open(imp).convert("RGB")
            w, h = rgb_pil.size
            orig = torch.tensor([[w, h]]).to(device)
            x = build_6ch(rgb_pil, heat_dir, rawir_dir, depth_dir, name)

            views = []
            for size in scales:
                t = to_model_input(x, size).to(device)
                for flip in flips:
                    ti = torch.flip(t, dims=[2]) if flip else t
                    labels, boxes, scores = infer(ti.unsqueeze(0), orig)
                    b = boxes[0].clone()
                    if flip:
                        x1, x2 = b[:, 0].clone(), b[:, 2].clone()
                        b[:, 0], b[:, 2] = w - x2, w - x1
                    keep = scores[0] > args.pre_thresh
                    for k in torch.nonzero(keep).flatten():
                        views.append((int(labels[0][k].item()),
                                      [float(b[k, 0]), float(b[k, 1]),
                                       float(b[k, 2]), float(b[k, 3])],
                                      float(scores[0][k].item())))

            fused = wbf(views, args.iou_thr, n_views=len(scales) * len(flips),
                        conf_mode=args.conf_mode) if views else []
            fused = [f for f in fused if f[2] >= args.thresh]
            fused = sorted(fused, key=lambda b: b[2], reverse=True)[:MAX_BOXES]
            lines = []
            for cls, (x1, y1, x2, y2), c in fused:
                cx = min(max((x1 + x2) / 2 / w, 0.0), 1.0)
                cy = min(max((y1 + y2) / 2 / h, 0.0), 1.0)
                bw = min(max((x2 - x1) / w, 0.0), 1.0)
                bh = min(max((y2 - y1) / h, 0.0), 1.0)
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {c:.4f}")
            # the official sample file ends WITH a trailing CRLF on its last line
            # (verified byte-for-byte against docs/预测结果格式范例.zip) -- emit the
            # same, so a strict parser does not drop/choke on the final box.
            payload = (LINESEP.join(lines) + LINESEP) if lines else ""
            with open(os.path.join(args.out, f"{stem}.txt"), "w", newline="") as f:
                f.write(payload)
            if (i + 1) % 50 == 0:
                print(f"[mm-tta] {i + 1}/{len(imgs)}", flush=True)

    print(f"[mm-tta] done -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
