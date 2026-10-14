"""Convert LLVIP (visible/infrared pairs + Pascal-VOC XML) to DEIMv2 COCO multimodal format.

Expected LLVIP layout (after unzip):
    visible/        RGB images, e.g. 000001.jpg
    infrared/       IR images, SAME names as visible
    annotations/    one Pascal-VOC .xml per image (person bboxes)

We build, under --out (default work_dir/LLVIP):
    rgb/                  symlinks -> visible images      (img_folder)
    ir/                   symlinks -> infrared images     (ir_folder)
    instances_llvip.json  COCO json, category person=0     (ann_file)

person -> category_id 0, which matches AIC class index 0 (person), so an
AIC-trained 4ch (rgb+ir) model can be evaluated directly on LLVIP.
The WHOLE LLVIP is used as a TEST set (the model never trains on it), which is
a valid out-of-distribution generalization check.
"""
import os
import glob
import json
import shutil
import argparse
import xml.etree.ElementTree as ET
from PIL import Image


def link(dst, src):
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy(src, dst)


def parse_xml(p):
    root = ET.parse(p).getroot()
    out = []
    for o in root.findall("object"):
        name = o.find("name").text
        b = o.find("bndbox")
        xmin = float(b.find("xmin").text)
        ymin = float(b.find("ymin").text)
        xmax = float(b.find("xmax").text)
        ymax = float(b.find("ymax").text)
        out.append((name, xmin, ymin, xmax, ymax))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llvip", required=True, help="path to unzipped LLVIP root")
    ap.add_argument("--out", default="work_dir/LLVIP")
    ap.add_argument("--split", default="all", choices=["all", "train", "test"],
                    help="which LLVIP split(s) to convert as OOD test set")
    args = ap.parse_args()

    vis = os.path.join(args.llvip, "visible")
    ir = os.path.join(args.llvip, "infrared")
    ann_candidates = [os.path.join(args.llvip, "Annotations"),
                      os.path.join(args.llvip, "annotations")]
    ann = next((p for p in ann_candidates if os.path.isdir(p)), ann_candidates[0])
    out_rgb = os.path.join(args.out, "rgb")
    out_ir = os.path.join(args.out, "ir")
    os.makedirs(out_rgb, exist_ok=True)
    os.makedirs(out_ir, exist_ok=True)

    # LLVIP nests images under visible/{train,test}/; glob recursively.
    img_files = sorted(glob.glob(os.path.join(vis, "**", "*"), recursive=True))
    img_files = [f for f in img_files
                 if os.path.isfile(f) and f.lower().endswith((".jpg", ".jpeg", ".png"))]
    print(f"[llvip] found {len(img_files)} visible images")
    images, annotations = [], []
    ann_id = 0
    skipped = 0
    for i, fp in enumerate(img_files):
        rel = os.path.relpath(fp, vis)            # e.g. train/250423.jpg
        stem, ext = os.path.splitext(rel)
        base = os.path.splitext(os.path.basename(fp))[0]   # 010001 (Annotations is flat)
        if args.split != "all" and not stem.startswith(args.split + "/"):
            continue
        xm = os.path.join(ann, base + ".xml")
        if not os.path.exists(xm):
            skipped += 1
            continue
        objs = parse_xml(xm)
        W, H = Image.open(fp).size
        # namespace filename (train_xxx / test_xxx) to avoid train/test collisions
        ns = stem.replace("/", "_") + ext
        link(os.path.join(out_rgb, ns), fp)
        link(os.path.join(out_ir, ns), os.path.join(ir, rel))
        images.append({"id": i, "file_name": ns, "width": W, "height": H})
        for (name, xmin, ymin, xmax, ymax) in objs:
            if name.lower() != "person":
                continue
            annotations.append({
                "id": ann_id, "image_id": i, "category_id": 0,
                "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                "area": (xmax - xmin) * (ymax - ymin), "iscrowd": 0,
            })
            ann_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 0, "name": "person"}],
        "info": {}, "licenses": [],
    }
    with open(os.path.join(args.out, "instances_llvip.json"), "w") as f:
        json.dump(coco, f)
    print(f"[llvip] converted {len(images)} images, {len(annotations)} person anns "
          f"({skipped} skipped w/o xml) -> {args.out}/instances_llvip.json")


if __name__ == "__main__":
    main()
