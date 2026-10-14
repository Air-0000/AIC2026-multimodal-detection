"""Convert the BDD100K **night** val subset to DEIMv2 COCO for zero-shot testing.

Source: the FiftyOne export `dgural/bdd100k` on HuggingFace (= the official 10k
validation split), fetched through hf-mirror because dl.cv.ethz.ch is blocked:
    data_aux/BDD100K_raw/samples.json          labels + timeofday + bboxes
    data_aux/BDD100K_raw/data/<uuid>.jpg       1280x720 images

Why this benchmark: BDD100K Nighttime is the standard "trained on day, tested at
night" detection protocol, and it publishes per-class AP, so our zero-shot
numbers can be compared class-by-class (see the published bounds quoted in
map_summary section 九).

Class mapping BDD -> AIC (only overlapping classes kept):
    pedestrian    -> 0  (person)
    bicycle       -> 5  (bicycle)
    car           -> 6  (car)
    traffic sign  -> 4  (sign)
    traffic light -> 8  (light)   # AIC "light" = vehicle lamp; approximate
Dropped (no AIC counterpart): rider, motorcycle, bus, truck, train, trailer,
other vehicle, other person.

NOTE: BDD100K is **RGB-only** -- it has no infrared and no depth. The aux folders
built here are null (black), so a 5ch model can only be run in a "missing aux"
configuration; the RGB-only 3ch model is the meaningful comparison.

Output:
    data_aux/BDD100K/rgb/<uuid>.jpg           symlinks to the downloaded images
    data_aux/BDD100K/ir_null/<uuid>.png       all-black (shared source)
    data_aux/BDD100K/depth_null/<uuid>.png    all-black
    data_aux/BDD100K/instances_bdd_night.json COCO, folder=rgb
"""
import os
import json
import glob
import argparse
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
RAW = os.path.join(ROOT, "data_aux/BDD100K_raw")
OUT = os.path.join(ROOT, "data_aux/BDD100K")
REPO = "dgural/bdd100k"

BDD2AIC = {  # maps onto AIC class ids (non-contiguous) -- for zero-shot eval vs AIC model
    "pedestrian": 0,
    "bicycle": 5,
    "car": 6,
    "traffic sign": 4,
    "traffic light": 8,
}
AIC_NAME = {0: "person", 4: "sign", 5: "bicycle", 6: "car", 8: "light"}

# Contiguous 0..4 mapping -- required when training a fresh 5-class head on BDD
# (CocoDetection feeds raw category_id as the label, and the loss does
# one_hot(label, num_classes+1); non-contiguous ids would index out of range).
BDD2C5 = {
    "pedestrian": 0,
    "traffic sign": 1,
    "bicycle": 2,
    "car": 3,
    "traffic light": 4,
}
C5_NAME = {0: "person", 1: "sign", 2: "bicycle", 3: "car", 4: "light"}


def _download(names, workers):
    from huggingface_hub import hf_hub_download
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    dst_dir = os.path.join(RAW, "data")
    os.makedirs(dst_dir, exist_ok=True)

    def one(uuid):
        dst = os.path.join(dst_dir, uuid + ".jpg")
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            return 0
        try:
            hf_hub_download(repo_id=REPO, repo_type="dataset",
                            filename=f"data/{uuid}.jpg", local_dir=RAW)
            return 1
        except Exception as e:
            print("DL FAIL", uuid, e, flush=True)
            return -1

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(one, names)):
            done += (r == 1)
            if (i + 1) % 250 == 0:
                print(f"[dl] {i+1}/{len(names)} (+{done} new)", flush=True)
    print(f"[dl] done, newly downloaded {done}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="night")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--contiguous", action="store_true",
                    help="emit contiguous 0..4 class ids (for training a 5-class head)")
    args = ap.parse_args()

    if args.contiguous:
        BDD2 = BDD2C5
        NAME = C5_NAME
        suffix = "_c5"
    else:
        BDD2 = BDD2AIC
        NAME = AIC_NAME
        suffix = ""

    samples = json.load(open(os.path.join(RAW, "samples.json")))["samples"]
    keep = [s for s in samples
            if (s.get("timeofday") or {}).get("label") == args.split]
    print(f"{args.split} samples: {len(keep)}")

    if not args.no_download:
        uuids = [os.path.splitext(os.path.basename(s["filepath"]))[0] for s in keep]
        _download(uuids, args.workers)

    out_rgb = os.path.join(OUT, "rgb")
    out_ir = os.path.join(OUT, "ir_null")
    out_dep = os.path.join(OUT, "depth_null")
    for d in (out_rgb, out_ir, out_dep):
        os.makedirs(d, exist_ok=True)

    images, anns = [], []
    aid = 0
    dropped = {}
    for s in keep:
        uuid = os.path.splitext(os.path.basename(s["filepath"]))[0]
        src = os.path.join(RAW, "data", uuid + ".jpg")
        if not os.path.exists(src):
            continue
        link = os.path.join(out_rgb, uuid + ".jpg")
        if not os.path.lexists(link):
            os.symlink(os.path.abspath(src), link)
        md = s.get("metadata") or {}
        W = int(md.get("width", 1280))
        H = int(md.get("height", 720))
        iid = len(images)
        images.append({"id": iid, "file_name": uuid + ".jpg", "width": W, "height": H})
        for g in ((s.get("detections") or {}).get("detections") or []):
            lb = g.get("label")
            if lb not in BDD2:
                dropped[lb] = dropped.get(lb, 0) + 1
                continue
            x, y, w, h = g["bounding_box"]          # normalised, top-left origin
            bx, by, bw, bh = x * W, y * H, w * W, h * H
            if bw <= 1 or bh <= 1:
                continue
            anns.append({"id": aid, "image_id": iid, "category_id": BDD2[lb],
                         "bbox": [bx, by, bw, bh],
                         "area": bw * bh, "iscrowd": 0})
            aid += 1

        coco = {
        "images": images,
        "annotations": anns,
        "categories": [{"id": v, "name": NAME[v]} for v in sorted(set(BDD2.values()))],
        "info": {}, "licenses": [],
        }
        jp = os.path.join(OUT, f"instances_bdd_{args.split}{suffix}.json")
        json.dump(coco, open(jp, "w"))
        print(f"[bdd] {len(images)} imgs, {len(anns)} boxes -> {jp}")
        print("[bdd] dropped:", dropped)

    # null aux (BDD100K has no IR/depth)
    black = os.path.join(OUT, "_black.png")
    if not os.path.exists(black):
        Image.fromarray(np.zeros((720, 1280), np.uint8), mode="L").save(black)
    made = 0
    for im in images:
        for d in (out_ir, out_dep):
            dst = os.path.join(d, im["file_name"])
            if not os.path.lexists(dst):
                os.symlink(black, dst)
                made += 1
    print(f"[bdd-null-aux] links made={made}")


if __name__ == "__main__":
    main()
