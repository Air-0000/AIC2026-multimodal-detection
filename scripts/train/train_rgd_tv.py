"""Lightweight RGB-D detection ablation with torchvision RetinaNet (no extra deps).

Four single-variable variants (depth channel only changes):
  baseline     -> RGB 3ch (control, no depth)
  inv_eq       -> RGB 3ch + inverse-depth CDF-equalised (the REAL fix)
  naive_clip8  -> RGB 3ch + current broken depth (uint16 clipped to 255)
  zero         -> RGB 3ch + all-black depth (bounds "does depth help at all")

Run all four:
  bash run_rgd_tv.sh            # default epochs=50 imgsz=512 batch=8
Single run:
  python train_rgd_tv.py --variant inv_eq --epochs 50
"""
import argparse
import os
import glob

import numpy as np
import torch
import cv2

from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.models import ResNet50_Weights

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
VIS = os.path.join(ROOT, "dataset/train/visible")
LAB = os.path.join(ROOT, "dataset/train/labels")
DEP_ROOT = os.path.join(ROOT, "work_dir", "depth_variants")
OUT_ROOT = os.path.join(ROOT, "work_dir", "rgd_tv")

# channel 4 (depth) has no ImageNet stats -> normalise to ~[0,1] only
NORM_MEAN = [0.485, 0.456, 0.406, 0.0]
NORM_STD = [0.229, 0.224, 0.225, 1.0]
NUM_CLASSES = 12  # AIC 12 classes, torchvision labels are 1..12


def load_labels(stem):
    p = os.path.join(LAB, stem + ".txt")
    if not os.path.exists(p):
        return None
    img = cv2.imread(os.path.join(VIS, stem + ".jpg"))
    if img is None:
        img = cv2.imread(os.path.join(VIS, stem + ".png"))
    if img is None:
        return None
    H, W = img.shape[:2]
    boxes, labels = [], []
    with open(p) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            cid = int(float(parts[0]))
            cx, cy, w, h = map(float, parts[1:5])
            x1 = max(0.0, (cx - w / 2) * W)
            y1 = max(0.0, (cy - h / 2) * H)
            x2 = min(float(W), (cx + w / 2) * W)
            y2 = min(float(H), (cy + h / 2) * H)
            if x2 - x1 < 1.0 or y2 - y1 < 1.0:  # skip degenerate boxes
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(cid + 1)  # 0 reserved as background
    if not boxes:
        return None
    return {"boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)}


def list_stems(variant):
    stems = []
    for f in sorted(os.listdir(VIS)):
        if not f.endswith((".jpg", ".png")):
            continue
        s = f[:-4]
        if load_labels(s) is None:
            continue
        if variant != "baseline":
            dp = os.path.join(DEP_ROOT, variant, "train", s + ".png")
            if not os.path.exists(dp):
                continue
        stems.append(s)
    return stems


class AICRGDDataset(torch.utils.data.Dataset):
    def __init__(self, stems, variant, imgsz):
        self.stems = stems
        self.variant = variant
        self.imgsz = imgsz

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, i):
        stem = self.stems[i]
        img = cv2.imread(os.path.join(VIS, stem + ".jpg"))
        if img is None:
            img = cv2.imread(os.path.join(VIS, stem + ".png"))
        if img is None:
            return None, None
        H, W = img.shape[:2]
        target = load_labels(stem)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        channels = [img]
        if self.variant != "baseline":
            dp = os.path.join(DEP_ROOT, self.variant, "train", stem + ".png")
            d = cv2.imread(dp, cv2.IMREAD_UNCHANGED)
            if d is None:
                d = np.zeros(img.shape[:2], np.uint8)
            if d.ndim == 3:
                d = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
            channels.append(d[..., None].astype(np.float32))
        arr = np.concatenate(channels, axis=2).astype(np.float32)
        arr = cv2.resize(arr, (self.imgsz, self.imgsz),
                         interpolation=cv2.INTER_NEAREST if arr.shape[2] == 4 else cv2.INTER_LINEAR)
        # GT boxes are in original (H,W) pixels; scale them to the resized
        # (imgsz,imgsz) space so they align with the network's input/anchors.
        if target is not None:
            sx, sy = self.imgsz / W, self.imgsz / H
            b = target["boxes"].float() * torch.tensor([sx, sy, sx, sy])
            target = {"boxes": b, "labels": target["labels"]}
        t = torch.from_numpy(arr / 255.0).permute(2, 0, 1).float()
        mean = torch.tensor(NORM_MEAN[: t.shape[0]])[:, None, None]
        std = torch.tensor(NORM_STD[: t.shape[0]])[:, None, None]
        t = (t - mean) / std
        return t, target


def build_model(variant, device):
    # build with 12 classes + background (random head), then load ImageNet backbone
    m = retinanet_resnet50_fpn(weights=None, num_classes=NUM_CLASSES + 1)
    sd = ResNet50_Weights.DEFAULT.get_state_dict(progress=False)
    m.backbone.body.load_state_dict(sd, strict=False)  # loads conv1/bn/layers (fc skipped)
    if variant != "baseline":
        old = m.backbone.body.conv1
        new = torch.nn.Conv2d(4, old.out_channels, old.kernel_size,
                              old.stride, old.padding, bias=False)
        with torch.no_grad():
            new.weight[:, :3] = old.weight          # 3 RGB channels = ImageNet
            new.weight[:, 3] = old.weight.mean(dim=1)  # 4th (depth) = mean of RGB
        m.backbone.body.conv1 = new
    return m.to(device)


def collate_fn(batch):
    return tuple(zip(*batch))


def box_iou(box, boxes):
    # box: [4], boxes: [N,4]
    boxes = boxes.to(box.device)
    x1 = torch.max(box[0], boxes[:, 0])
    y1 = torch.max(box[1], boxes[:, 1])
    x2 = torch.min(box[2], boxes[:, 2])
    y2 = torch.min(box[3], boxes[:, 3])
    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_box + area_boxes - inter
    return inter / union.clamp(1e-6)


def voc_map(preds_gts, iou_thr=0.5):
    aps = []
    for c in range(1, NUM_CLASSES + 1):
        tpfp, ngt = [], 0
        for gt, out in preds_gts:
            gmask = gt["labels"] == c
            gboxes = gt["boxes"][gmask]
            ngt += len(gboxes)
            pmask = out["labels"] == c
            pboxes = out["boxes"][pmask]
            pscores = out["scores"][pmask]
            if len(pboxes) == 0:
                continue
            order = torch.argsort(-pscores)
            pboxes = pboxes[order]
            matched = [False] * len(gboxes)
            for pb in pboxes:
                if len(gboxes) == 0:
                    tpfp.append(0.0)
                    continue
                ious = box_iou(pb, gboxes)
                best, j = ious.max().item(), int(ious.argmax().item())
                if best >= iou_thr and not matched[j]:
                    matched[j] = True
                    tpfp.append(1.0)
                else:
                    tpfp.append(0.0)
        if ngt == 0 or not tpfp:
            aps.append(0.0)
            continue
        tp = np.cumsum(tpfp)
        fp = np.cumsum([1 - x for x in tpfp])
        rec = tp / ngt
        prec = tp / (tp + fp)
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            ap += (prec[rec >= t].max() if np.any(rec >= t) else 0.0) / 11
        aps.append(ap)
    return float(np.mean(aps)), [float(a) for a in aps]


@torch.no_grad()
def evaluate(model, dataset, device):
    model.eval()
    preds_gts = []
    for i in range(len(dataset)):
        im, target = dataset[i]
        if target is None:
            continue
        out = model([im.to(device)])[0]
        preds_gts.append((target, out))
    return voc_map(preds_gts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="baseline",
                    choices=["baseline", "inv_eq", "naive_clip8", "zero"])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = os.path.join(OUT_ROOT, args.variant)
    os.makedirs(out_dir, exist_ok=True)
    log = open(os.path.join(out_dir, "train.log"), "w")

    def say(*a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        log.write(msg + "\n")

    stems = list_stems(args.variant)
    # fixed 90/10 split, identical across variants (index % 10 == 0 -> val)
    train_stems = [s for i, s in enumerate(stems) if i % 10 != 0]
    val_stems = [s for i, s in enumerate(stems) if i % 10 == 0]
    say(f"[variant={args.variant}] total={len(stems)} train={len(train_stems)} val={len(val_stems)} device={device}")

    # ---- data sanity (fast: labels only, no image read) ----
    bad = 0
    for s in train_stems + val_stems:
        t = load_labels(s)
        if t is None:
            say("BAD label-None", s); bad += 1; continue
        if t["boxes"].shape[0] != t["labels"].shape[0]:
            say("BAD len", s); bad += 1; continue
        if t["labels"].min() < 1 or t["labels"].max() > NUM_CLASSES:
            say("BAD label range", s, int(t["labels"].min()), int(t["labels"].max())); bad += 1; continue
        if not torch.isfinite(t["boxes"]).all():
            say("BAD box nan", s); bad += 1
    say("sanity bad =", bad)

    train_ds = AICRGDDataset(train_stems, args.variant, args.imgsz)
    val_ds = AICRGDDataset(val_stems, args.variant, args.imgsz)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, num_workers=4,
        collate_fn=collate_fn, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)

    model = build_model(args.variant, device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    say(f"params={n_params:.1f}M in_ch={model.backbone.body.conv1.in_channels}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_map, best_state = -1, None

    for ep in range(args.epochs):
        model.train()
        ep_loss = 0.0
        for imgs, targets in train_loader:
            imgs = [im.to(device) for im in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(imgs, targets)
            loss = sum(loss_dict.values())
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        ep_loss /= len(train_loader)
        m, _ = evaluate(model, val_ds, device)
        say(f"ep{ep+1:03d} train_loss={ep_loss:.3f} val_mAP50={m:.4f}")
        if m > best_map:
            best_map = m
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(best_state, os.path.join(out_dir, "best.pth"))
            say(f"  -> new best val_mAP50={best_map:.4f}, saved")

    say(f"DONE variant={args.variant} best_val_mAP50={best_map:.4f}")
    log.close()


if __name__ == "__main__":
    main()
