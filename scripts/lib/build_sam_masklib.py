"""Build SAM mask library for Copy-Paste augmentation.

Runs SAM (ViT-B) box-prompt over GT boxes of chosen classes in the AIC train
set, saves per-instance cropped RGB + binary mask PNG + an index json.

Usage:
  python build_sam_masklib.py --device cuda            # full run (~10 min on GPU)
  python build_sam_masklib.py --device cpu --limit 30  # smoke subset
"""
import argparse
import json
import os
from collections import defaultdict

import cv2
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry

ROOT = "/home/constant/CodeBuddy/Dev/AIC/work_dir/deimv2_dataset/train"
ANN = os.path.join(ROOT, "annotations/instances_train.json")
CKPT = "/home/constant/CodeBuddy/Dev/AIC/work_dir/sam/sam_vit_b_01ec64.pth"
OUT = "/home/constant/CodeBuddy/Dev/AIC/work_dir/sam/masklib"
TARGET_CLASSES = [
    "tricycle", "ball", "boat", "uav", "garbage can",  # tail
    "seat", "bicycle", "sign",                          # mid (small boost)
]
MIN_AREA = 400  # skip degenerate boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="max images (0=all)")
    ap.add_argument("--min-area", type=int, default=MIN_AREA)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    d = json.load(open(ANN))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    name2id = {v: k for k, v in cats.items()}
    keep_ids = {name2id[n] for n in TARGET_CLASSES if n in name2id}
    imgs = {im["id"]: im for im in d["images"]}

    by_img = defaultdict(list)
    for a in d["annotations"]:
        if a["category_id"] in keep_ids and a.get("area", 0) >= args.min_area:
            by_img[a["image_id"]].append(a)
    img_ids = sorted(by_img)
    if args.limit:
        rng = np.random.default_rng(0)
        rng.shuffle(img_ids)
        img_ids = img_ids[: args.limit]
    n_inst = sum(len(by_img[i]) for i in img_ids)
    print(f"{len(img_ids)} images / {n_inst} instances on {args.device}")

    sam = sam_model_registry["vit_b"](checkpoint=CKPT).to(args.device).eval()
    pred = SamPredictor(sam)

    index = []
    for n, iid in enumerate(img_ids):
        info = imgs[iid]
        path = os.path.join(ROOT, "images", info["file_name"])
        bgr = cv2.imread(path)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pred.set_image(rgb)
        H, W = rgb.shape[:2]
        for a in by_img[iid]:
            x, y, w, h = a["bbox"]
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(W, int(x + w)), min(H, int(y + h))
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            masks, scores, _ = pred.predict(
                box=np.array([x1, y1, x2, y2]),
                multimask_output=False,
            )
            m = masks[0]
            ys, xs = np.where(m)
            if len(xs) == 0:
                continue
            mx1, mx2, my1, my2 = xs.min(), xs.max(), ys.min(), ys.max()
            inst_id = f"{iid}_{a['id']}"
            crop_rgb = rgb[my1:my2 + 1, mx1:mx2 + 1]
            crop_m = (m[my1:my2 + 1, mx1:mx2 + 1] * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(OUT, f"{inst_id}_rgb.png"),
                        cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(OUT, f"{inst_id}_m.png"), crop_m)
            index.append(dict(
                id=inst_id, cat_id=a["category_id"], cat=cats[a["category_id"]],
                file=info["file_name"],
                crop_box=[int(mx1), int(my1), int(mx2), int(my2)],
                gt_wh=[float(w), float(h)], score=float(scores[0]),
                area=int(m.sum()),
            ))
        if (n + 1) % 100 == 0:
            print(f"  {n + 1}/{len(img_ids)}")

    with open(os.path.join(OUT, "index.json"), "w") as f:
        json.dump(index, f)
    from collections import Counter
    cnt = Counter(r["cat"] for r in index)
    print("done:", dict(cnt), "->", os.path.join(OUT, "index.json"))


if __name__ == "__main__":
    main()
