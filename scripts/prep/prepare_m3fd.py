"""Convert M3FD (multi-modal road-target detection) to DEIMv2 COCO multimodal format.

M3FD ships aligned visible/infrared pairs + COCO JSON (instances_train/val.json).
We use the official VAL split as an OOD test set (never trained). Categories are
mapped onto AIC where they overlap:
    People -> 0 (person)      Car -> 6 (car)      Lamp -> 8 (light, 车灯)
    Bus / Motorcycle / Truck  -> dropped (no AIC counterpart; logged).
This mirrors prepare_flir.py so coco_dataset_mm.py can consume it as 5ch input
(rgb/ + ir/ symlinks, instances_m3fd_val.json). Heat residual + null depth are
also built (work_dir/M3FD_heat, work_dir/M3FD_depth_null) for the 5ch recipes,
exactly like FLIR/LLVIP.
"""
import os
import json
import glob
import argparse
import numpy as np
from PIL import Image
from multiprocessing import Pool

AIC = "/home/constant/CodeBuddy/Dev/AIC"
# AIC category id -> name (for the output json)
AIC_NAME = {0: "person", 6: "car", 8: "light"}
# M3FD COCO name -> AIC id (only these are kept; Bus/Motorcycle/Truck dropped)
NAME2AIC = {"People": 0, "Car": 6, "Lamp": 8}


def find_src(root, file_name, modality_dir):
    base = os.path.basename(file_name)
    cands = [
        os.path.join(root, file_name),
        os.path.join(root, modality_dir, base),
        os.path.join(root, modality_dir, file_name),
        os.path.join(root, modality_dir, os.path.splitext(base)[0] + ".png"),
    ]
    for c in cands:
        if os.path.exists(c):
            return c
    return None


def find_root(m3fd):
    c = glob.glob(os.path.join(m3fd, "**", "COCO", "instances_val.json"), recursive=True)
    if c:
        return os.path.dirname(os.path.dirname(c[0]))
    c = glob.glob(os.path.join(m3fd, "**", "instances_val.json"), recursive=True)
    if c:
        return os.path.dirname(c[0])
    raise SystemExit("cannot find M3FD COCO/instances_val.json under " + m3fd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m3fd", required=True, help="unzipped M3FD root (contains vi/ ir/ COCO/)")
    ap.add_argument("--out", default=os.path.join(AIC, "work_dir/M3FD"))
    ap.add_argument("--split", default="val", choices=["val", "train"])
    args = ap.parse_args()

    root = find_root(args.m3fd)
    coco = json.load(open(os.path.join(root, "COCO", f"instances_{args.split}.json")))

    cats = {c["id"]: c["name"] for c in coco["categories"]}
    keep = {cid: NAME2AIC[n] for cid, n in cats.items() if n in NAME2AIC}
    dropped = {cid: cats[cid] for cid in cats if cid not in keep}
    print("M3FD cats:", cats)
    print("keep(AIC id):", keep, " dropped:", dropped)

    out_rgb = os.path.join(args.out, "rgb")
    out_ir = os.path.join(args.out, "ir")
    os.makedirs(out_rgb, exist_ok=True)
    os.makedirs(out_ir, exist_ok=True)

    old2new = {}
    new_imgs = []
    skipped = 0
    for im in coco["images"]:
        fn = im["file_name"]
        base = os.path.basename(fn)
        vi = find_src(root, fn, "vi")
        ir = find_src(root, fn, "ir")
        if vi is None or ir is None:
            skipped += 1
            continue
        os.symlink(os.path.abspath(vi), os.path.join(out_rgb, base))
        os.symlink(os.path.abspath(ir), os.path.join(out_ir, base))
        nid = len(new_imgs)
        new_imgs.append({"id": nid, "file_name": base,
                         "height": im["height"], "width": im["width"]})
        old2new[im["id"]] = nid

    new_anns = []
    ann_id = 0
    dropped_anns = 0
    for a in coco["annotations"]:
        oid = a["image_id"]
        if oid not in old2new:
            dropped_anns += 1
            continue
        if a["category_id"] not in keep:
            dropped_anns += 1
            continue
        na = dict(a)
        na["image_id"] = old2new[oid]
        na["category_id"] = keep[a["category_id"]]
        na["id"] = ann_id
        new_anns.append(na)
        ann_id += 1

    new_cats = [{"id": v, "name": AIC_NAME[v]} for v in keep.values()]
    out = {
        "images": new_imgs,
        "annotations": new_anns,
        "categories": new_cats,
        "info": {}, "licenses": [],
    }
    out_path = os.path.join(args.out, f"instances_m3fd_{args.split}.json")
    json.dump(out, open(out_path, "w"))
    print(f"[m3fd] {len(new_imgs)} imgs, {len(new_anns)} boxes "
          f"({skipped} imgs skipped, {dropped_anns} anns dropped) -> {out_path}")

    # heat residual + null depth (mirror make_flir_heat_depth.py)
    build_aux(out_rgb, out_ir, args.out)


def build_aux(out_rgb, out_ir, out):
    HEAT = os.path.join(AIC, "work_dir", "M3FD_heat")
    DEPTH = os.path.join(AIC, "work_dir", "M3FD_depth_null")
    os.makedirs(HEAT, exist_ok=True)
    os.makedirs(DEPTH, exist_ok=True)

    def make_heat(name):
        dst = os.path.join(HEAT, name)
        if os.path.exists(dst):
            return 0
        v = np.array(Image.open(os.path.join(out_rgb, name)).convert("RGB")).astype(np.float32).mean(axis=2)
        ir = np.array(Image.open(os.path.join(out_ir, name)).convert("RGB")).astype(np.float32).mean(axis=2)
        if ir.shape != v.shape:
            ir = np.array(Image.fromarray(ir.astype(np.uint8)).resize((v.shape[1], v.shape[0]))).astype(np.float32)
        heat = np.clip(ir - v, 0, 255).astype(np.uint8)
        Image.fromarray(np.stack([heat, heat, heat], axis=-1)).save(dst, quality=95)
        return 1

    names = sorted(os.listdir(out_rgb))
    print(f"[m3fd-heat] {len(names)} images", flush=True)
    done = 0
    with Pool(8) as p:
        for i, r in enumerate(p.imap_unordered(make_heat, names, chunksize=64)):
            done += r
            if (i + 1) % 500 == 0:
                print(f"[m3fd-heat] {i+1}/{len(names)}", flush=True)
    print(f"[m3fd-heat] done: {done} -> {HEAT}", flush=True)

    black = os.path.join(DEPTH, "_black.png")
    if not os.path.exists(black):
        ref = os.path.join(out_rgb, names[0])
        w, h = Image.open(ref).size
        Image.fromarray(np.zeros((h, w), dtype=np.uint8), mode="L").save(black)
    made = 0
    for n in names:
        dst = os.path.join(DEPTH, n)
        if os.path.lexists(dst):
            continue
        os.symlink(black, dst)
        made += 1
    print(f"[m3fd-depth-null] links made={made} total={len(names)}", flush=True)


if __name__ == "__main__":
    main()
