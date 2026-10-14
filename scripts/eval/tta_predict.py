"""
Test-Time Augmentation (TTA) inference for the RGB RF-DETR baseline.

For each test image we run the model under several augmentations:
  - horizontal flip (on/off)
  - multi-scale resize (SCALES)
Boxes are mapped back to the original image coordinates. The RAW multi-aug
predictions are cached (no pre-fusion), then fused per-class at write time
using either NMS or WBF (Weighted Boxes Fusion). This lets us compare fusion
strategies without re-inferring.

Usage:
    python3 tta_predict.py [--force] [--wbf]
        --force  re-run inference (regenerate cache)
        --wbf    fuse with Weighted Boxes Fusion instead of NMS
Outputs (work_dir/):
    submission_tta_t{0.4,0.5,0.6}.zip            (NMS, default)
    submission_tta_wbf_t{0.4,0.5,0.6}.zip        (WBF)
"""
import os
import sys
import glob
import pickle
import subprocess
import numpy as np
from PIL import Image
from rfdetr import RFDETRBase

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
CKPT = os.path.join(ROOT, "work_dir/rgb_baseline", "checkpoint_best_ema.pth")
TEST_VISIBLE = os.path.join(ROOT, "dataset/test/visible")
CACHE = os.path.join(ROOT, "work_dir", "tta_cache_raw.pkl")

# TTA config
SCALES = [0.83, 1.0, 1.2]
FLIPS = [False, True]
NMS_IOU = 0.5
WBF_IOU = 0.55
MAX_BOXES = 100
# Raw cache is threshold=0.0, so it carries ~1800 boxes/image (mostly sub-0.1
# noise). Drop anything safely below the lowest eval threshold (0.4) before
# fusion: WBF is O(N^2) over the full set, so this cuts fusion ~200x with zero
# effect on the 0.4-0.7 submission outputs.
FUSE_FLOOR = 0.1
THRESHOLDS = [0.4, 0.5, 0.6, 0.65, 0.7]
LINESEP = "\r\n"
USE_WBF = "--wbf" in sys.argv
FORCE = "--force" in sys.argv
FUSE_TAG = "tta_wbf" if USE_WBF else "tta"


def iou(box, boxes):
    x1, y1, x2, y2 = box
    xx1 = np.maximum(x1, boxes[:, 0]); yy1 = np.maximum(y1, boxes[:, 1])
    xx2 = np.minimum(x2, boxes[:, 2]); yy2 = np.minimum(y2, boxes[:, 3])
    w = np.maximum(0.0, xx2 - xx1); h = np.maximum(0.0, yy2 - yy1)
    inter = w * h
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area1 + area2 - inter + 1e-9)


def nms(boxes, scores, iou_thr=NMS_IOU):
    if len(boxes) == 0:
        return np.array([], dtype=int)
    idx = np.argsort(scores)[::-1]
    keep = []
    while len(idx):
        i = idx[0]
        keep.append(i)
        if len(idx) == 1:
            break
        rest = idx[1:]
        ious = iou(boxes[i], boxes[rest])
        idx = rest[ious < iou_thr]
    return np.array(keep, dtype=int)


def fuse_nms(boxes, cats, scores):
    out_b, out_c, out_s = [], [], []
    for cl in np.unique(cats):
        m = cats == cl
        k = nms(boxes[m], scores[m], NMS_IOU)
        out_b.append(boxes[m][k]); out_c.append(cats[m][k]); out_s.append(scores[m][k])
    if out_b:
        return np.concatenate(out_b, 0), np.concatenate(out_c, 0), np.concatenate(out_s, 0)
    return np.zeros((0, 4)), np.array([]), np.array([])


def iou_matrix(boxes):
    """N x N IoU matrix (vectorized)."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    xx1 = np.maximum(x1[:, None], x1[None, :])
    yy1 = np.maximum(y1[:, None], y1[None, :])
    xx2 = np.minimum(x2[:, None], x2[None, :])
    yy2 = np.minimum(y2[:, None], y2[None, :])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h
    union = areas[:, None] + areas[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def fuse_wbf(boxes, cats, scores):
    """Pure-Python Weighted Boxes Fusion (vectorized). For each class, greedily
    cluster boxes whose IoU > WBF_IOU, then output a confidence-weighted average
    box with score = max fused confidence (conf_type='max', fair vs NMS)."""
    if len(boxes) == 0:
        return np.zeros((0, 4)), np.array([]), np.array([])
    # drop non-finite / invalid / zero-area boxes (RF-DETR may emit rare ones)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area = (x2 - x1) * (y2 - y1)
    keep = np.all(np.isfinite(boxes), axis=1) & np.isfinite(scores) & (scores >= 0) & (area > 0)
    boxes, cats, scores = boxes[keep], cats[keep], scores[keep]
    if len(boxes) == 0:
        return np.zeros((0, 4)), np.array([]), np.array([])
    out_b, out_c, out_s = [], [], []
    for cl in np.unique(cats):
        m = cats == cl
        bb, sc = boxes[m], scores[m]
        if len(bb) == 0:
            continue
        iom = iou_matrix(bb)
        used = np.zeros(len(bb), dtype=bool)
        order = np.argsort(-sc)
        for i in order:
            if used[i]:
                continue
            grp = np.where((iom[i] > WBF_IOU) & (~used))[0]
            if len(grp) == 0:
                continue
            used[grp] = True
            w = sc[grp].copy()
            w[w <= 0] = 1e-6  # guard against zero-weight (conf==0) boxes
            out_b.append(np.average(bb[grp], axis=0, weights=w))
            out_c.append(cl)
            out_s.append(float(np.max(sc[grp])))
    if out_b:
        return np.array(out_b), np.array(out_c), np.array(out_s)
    return np.zeros((0, 4)), np.array([]), np.array([])


def main():
    model = RFDETRBase(pretrain_weights=CKPT, resolution=560, num_classes=12)
    imgs = sorted(glob.glob(os.path.join(TEST_VISIBLE, "*")))
    print(f"[tta] {len(imgs)} images, scales={SCALES}, flips={FLIPS}, fuse={'WBF' if USE_WBF else 'NMS'}")

    cache = {}
    if os.path.exists(CACHE) and not FORCE:
        print(f"[tta] loading cached RAW detections from {CACHE}")
        with open(CACHE, "rb") as f:
            cache = pickle.load(f)
    else:
        for n, imp in enumerate(imgs):
            stem = os.path.splitext(os.path.basename(imp))[0]
            im = Image.open(imp).convert("RGB")
            W, H = im.size
            all_xyxy, all_cid, all_conf = [], [], []
            for s in SCALES:
                nw, nh = max(1, int(round(W * s))), max(1, int(round(H * s)))
                im_s = im.resize((nw, nh))
                batch = []
                for flip in FLIPS:
                    batch.append(im_s.transpose(Image.FLIP_LEFT_RIGHT) if flip else im_s)
                dets = model.predict(batch, threshold=0.0)
                for flip, det in zip(FLIPS, dets):
                    xyxy = np.asarray(det.xyxy, dtype=float)
                    cid = np.asarray(det.class_id, dtype=int)
                    conf = np.asarray(det.confidence, dtype=float)
                    if len(xyxy) == 0:
                        continue
                    if flip:
                        x1, y1, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
                        nx1, nx2 = nw - x2, nw - x1
                        xyxy = np.stack([nx1, y1, nx2, y2], axis=1)
                    xyxy = xyxy / s
                    all_xyxy.append(xyxy); all_cid.append(cid); all_conf.append(conf)
            if all_xyxy:
                cache[stem] = (np.concatenate(all_xyxy, 0), np.concatenate(all_cid, 0), np.concatenate(all_conf, 0))
            else:
                cache[stem] = (np.zeros((0, 4)), np.array([]), np.array([]))
            if (n + 1) % 100 == 0:
                print(f"[tta] inferred {n + 1}/{len(imgs)}")
        with open(CACHE, "wb") as f:
            pickle.dump(cache, f)
        print(f"[tta] cached RAW {len(cache)} images -> {CACHE}")

    # fuse once per image (fusion is independent of threshold)
    sizes = {os.path.splitext(os.path.basename(p))[0]: Image.open(p).size for p in imgs}
    fused = {}
    for stem, (b, c, cf) in cache.items():
        if len(cf) == 0:
            fused[stem] = (np.zeros((0, 4)), np.array([]), np.array([]))
        else:
            m = cf >= FUSE_FLOOR
            if USE_WBF:
                fused[stem] = fuse_wbf(b[m], c[m], cf[m])
            else:
                fused[stem] = fuse_nms(b[m], c[m], cf[m])
    for t in THRESHOLDS:
        out = os.path.join(ROOT, f"work_dir/submission_{FUSE_TAG}_t{t}")
        os.makedirs(out, exist_ok=True)
        for stem, (fb, fc, fcf) in fused.items():
            W, H = sizes[stem]
            lines = []
            if len(fcf) > 0:
                m = fcf >= t
                bb, cc, ccf = fb[m], fc[m], fcf[m]
                for k in np.argsort(-ccf)[:MAX_BOXES]:
                    x1, y1, x2, y2 = bb[k]
                    cx = min(max((x1 + x2) / 2 / W, 0.0), 1.0)
                    cy = min(max((y1 + y2) / 2 / H, 0.0), 1.0)
                    w = min(max((x2 - x1) / W, 0.0), 1.0)
                    h = min(max((y2 - y1) / H, 0.0), 1.0)
                    lines.append(f"{int(cc[k])} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {float(ccf[k]):.4f}")
            with open(os.path.join(out, stem + ".txt"), "w", newline="") as f:
                if lines:
                    f.write(LINESEP.join(lines) + LINESEP)
        zipf = os.path.join(ROOT, f"work_dir/submission_{FUSE_TAG}_t{t}.zip")
        if os.path.exists(zipf):
            os.remove(zipf)
        files = glob.glob(os.path.join(out, "*.txt"))
        subprocess.run(["zip", "-j", zipf, *files], check=True)
        print(f"[tta] wrote {len(cache)} files -> {zipf}")

    print("[tta] DONE")


if __name__ == "__main__":
    main()
