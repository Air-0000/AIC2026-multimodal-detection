"""BDD-day + DNDT physical nightify mixed training set (mirrors build_aic_nightcut_dataset.py).

1) nightify every BDD daytime frame -> data_aux/BDD100K/rgb_nightify/<base>.png
2) build a combined COCO json (day + nightify) with shared boxes, symlinks into
   data_aux/BDD100K_nightify/
Usage: python build_bdd_nightify.py
"""
import os, json, sys
import numpy as np
from PIL import Image

AIC = "/home/constant/CodeBuddy/Dev/AIC"
sys.path.insert(0, AIC)
from nightify_aic import nightify, _rng_for

SRC_IMG = os.path.join(AIC, "data_aux/BDD100K/rgb")
SRC_ANN = os.path.join(AIC, "data_aux/BDD100K/instances_bdd_daytime.json")
NIGHT   = os.path.join(AIC, "data_aux/BDD100K/rgb_nightify")
OUT     = os.path.join(AIC, "data_aux/BDD100K_nightify")


def gen_nightify():
    os.makedirs(NIGHT, exist_ok=True)
    fs = sorted(f for f in os.listdir(SRC_IMG) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    done = 0
    for f in fs:
        out = os.path.join(NIGHT, os.path.splitext(f)[0] + ".png")
        if os.path.exists(out):
            continue
        day = np.asarray(Image.open(os.path.join(SRC_IMG, f)).convert("RGB"))
        arr = nightify(day, _rng_for(os.path.splitext(f)[0]))
        Image.fromarray(arr).save(out)
        done += 1
        if done % 500 == 0:
            print(f"nightify {done}/{len(fs)}", flush=True)
    print(f"nightify done: {len(fs)} frames -> {NIGHT}", flush=True)


def build():
    src = json.load(open(SRC_ANN))
    out_img = os.path.join(OUT, "images")
    out_ann = os.path.join(OUT, "annotations")
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_ann, exist_ok=True)
    anns_by_img = {}
    for a in src["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)
    images, anns, aid, miss = [], [], 0, 0
    for im in src["images"]:
        fn = im["file_name"]
        base = os.path.splitext(fn)[0]
        night_name = base + ".png"
        night_src = os.path.join(NIGHT, night_name)
        if not os.path.exists(night_src):
            miss += 1
            continue
        # day copy
        dst = os.path.join(out_img, fn)
        if not os.path.lexists(dst):
            os.symlink(os.path.abspath(os.path.join(SRC_IMG, fn)), dst)
        iid = len(images)
        images.append({"id": iid, "file_name": fn, "width": im["width"], "height": im["height"]})
        for a in anns_by_img.get(im["id"], []):
            b = dict(a); b["id"] = aid; b["image_id"] = iid; anns.append(b); aid += 1
        # nightify copy
        dst = os.path.join(out_img, night_name)
        if not os.path.lexists(dst):
            os.symlink(os.path.abspath(night_src), dst)
        iid = len(images)
        images.append({"id": iid, "file_name": night_name, "width": im["width"], "height": im["height"]})
        for a in anns_by_img.get(im["id"], []):
            b = dict(a); b["id"] = aid; b["image_id"] = iid; anns.append(b); aid += 1
    coco = {"images": images, "annotations": anns,
            "categories": src["categories"], "info": {}, "licenses": []}
    json.dump(coco, open(os.path.join(out_ann, "instances_train.json"), "w"))
    print(f"[bdd-nightify] {len(images)} images, {len(anns)} boxes, missing_night={miss} -> {out_ann}/instances_train.json")


if __name__ == "__main__":
    gen_nightify()
    build()
