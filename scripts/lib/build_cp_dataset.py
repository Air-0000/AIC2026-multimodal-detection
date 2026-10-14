"""Build OFFLINE Copy-Paste dataset.

对每张训练图离线执行一次 RandomCopyPaste(prob=1.0), 产出:
  train_cpoff/images/          原图(硬链接) + CP_<i>_<orig>.jpg 贴好图
  train_cpoff/annotations/instances_train_cp.json   1800 原图 + 贴好图, 标签同步
贴入规则与在线版逐字一致(同一算子类): class 尺寸/只缩不放/零重叠/sqrt 采样/min_score 0.7

用法: /home/constant/miniconda3/envs/deimv2/bin/python build_cp_dataset.py [--limit N]
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch
from PIL import Image

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
sys.path.insert(0, os.path.join(ROOT, "vendor/DEIMv2"))
sys.path.insert(0, os.path.join(ROOT, "scripts/lib"))

from engine.data.transforms._transforms import RandomCopyPaste  # noqa: E402
from torchvision.tv_tensors import BoundingBoxes as BB  # noqa: E402
from torchvision.tv_tensors import BoundingBoxFormat as BF  # noqa: E402

SRC = os.path.join(ROOT, "work_dir/deimv2_dataset/train")
DST = os.path.join(ROOT, "work_dir/deimv2_dataset/train_cpoff")
ANN_IN = os.path.join(SRC, "annotations/instances_train.json")
IMG_ID_OFF = 100000
ANN_ID_OFF = 1000000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(os.path.join(DST, "images"), exist_ok=True)
    os.makedirs(os.path.join(DST, "annotations"), exist_ok=True)
    A = json.load(open(ANN_IN))
    imgs = {im["id"]: im for im in A["images"]}

    by_img = {}
    for a in A["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)

    # 原图硬链接(不占额外磁盘), 注解原样保留
    link_dir = os.path.join(DST, "images")
    n_linked = 0
    for im in A["images"]:
        dstp = os.path.join(link_dir, im["file_name"])
        if not os.path.exists(dstp):
            os.link(os.path.join(SRC, "images", im["file_name"]), dstp)
        n_linked += 1

    cp = RandomCopyPaste(
        mask_lib=os.path.join(ROOT, "work_dir/sam/masklib/index.json"),
        prob=1.0, max_paste=3, iou_thresh=0.3, sqrt_pow=0.5, min_score=0.7,
        ann_file=ANN_IN, size_mode="class", max_upscale=2.5,
    )

    out_images = list(A["images"])
    out_anns = list(A["annotations"])
    next_img_id = IMG_ID_OFF
    next_ann_id = ANN_ID_OFF
    n_pasted_imgs = 0
    n_pasted_inst = 0
    ids = sorted(imgs)
    if args.limit:
        ids = ids[: args.limit]

    for n, iid in enumerate(ids):
        info = imgs[iid]
        W, H = info["width"], info["height"]
        anns = by_img.get(iid, [])
        boxes = []
        for a in anns:
            x, y, w, h = a["bbox"][:4]
            boxes.append([x, y, x + w, y + h])
        t = {
            "boxes": BB(torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                        format=BF.XYXY, canvas_size=(H, W)),
            "labels": torch.tensor([a["category_id"] for a in anns],
                                   dtype=torch.int64),
        }
        np.random.seed(20000 + n)
        torch.manual_seed(20000 + n)
        img = Image.open(os.path.join(SRC, "images", info["file_name"])).convert("RGB")
        # ⚠ forward 返回 (新图, 新target): 像素改动在新 PIL 对象上,
        #   不接受返回值就会把原图存出去(空框) —— 首版实锤过
        img2, t2 = cp.forward(img, dict(t))
        n_orig = len(anns)
        n_new = int(t2["boxes"].shape[0]) - n_orig
        if n_new <= 0:
            continue
        fname = f"CP_{n:05d}_{info['file_name']}"
        img2.save(os.path.join(link_dir, fname), quality=95)
        b2 = t2["boxes"].numpy()
        l2 = t2["labels"].tolist()
        img_id = next_img_id
        next_img_id += 1
        out_images.append(dict(id=img_id, file_name=fname, width=W, height=H))
        for k in range(n_new):
            x1, y1, x2, y2 = b2[n_orig + k]
            bw, bh = float(x2 - x1), float(y2 - y1)
            out_anns.append(dict(
                id=next_ann_id, image_id=img_id, category_id=int(l2[n_orig + k]),
                bbox=[round(float(x1), 2), round(float(y1), 2), round(bw, 2), round(bh, 2)],
                area=round(bw * bh, 2), iscrowd=0))
            next_ann_id += 1
        n_pasted_imgs += 1
        n_pasted_inst += n_new
        if (n + 1) % 200 == 0:
            print(f"  {n + 1}/{len(ids)} pasted_imgs={n_pasted_imgs} "
                  f"inst={n_pasted_inst}", flush=True)

    out = dict(A)
    out["images"] = out_images
    out["annotations"] = out_anns
    with open(os.path.join(DST, "annotations/instances_train_cp.json"), "w") as f:
        json.dump(out, f)
    print(f"DONE: images={len(out_images)} anns={len(out_anns)} "
          f"cp_imgs={n_pasted_imgs} cp_inst={n_pasted_inst}")


if __name__ == "__main__":
    main()
