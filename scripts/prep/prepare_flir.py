"""Convert FLIR aligned (RGB + thermal + Pascal-VOC XML) to DEIMv2 multimodal COCO.

FLIR "aligned" release layout (after unzip of aligned.zip):
    align/AnnotatedImages/FLIR_XXXXX_RGB.jpg        (visible)
    align/AnnotatedImages/FLIR_XXXXX_PreviewData.jpeg (thermal / IR)
    align/Annotations/FLIR_XXXXX_PreviewData.xml     (boxes, in thermal coords;
                                                      thermal is registered to RGB)
    align/align_train.txt / align_validation.txt     (official split, one base
                                                      name `FLIR_XXXXX_PreviewData` per line)

We mirror prepare_llvip.py: build under --out (default work_dir/FLIR)
    rgb/                     symlinks -> RGB images      (img_folder)
    ir/                      symlinks -> thermal images  (ir_folder)
    instances_flir_train.json / instances_flir_val.json  (ann_file)

Class mapping FLIR -> AIC (so an AIC-trained 5ch model evaluates directly):
    person   -> 0  (person)
    car      -> 6  (car)
    bicycle  -> 5  (bicycle)
    dog      -> 2  (animal)   # AIC has no dog class; collapsed to animal
FLIR only ever annotates these four classes, so the resulting json scores exactly
those categories.
"""
import os
import glob
import json
import shutil
import argparse
import xml.etree.ElementTree as ET
from PIL import Image

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
SRC = os.path.join(ROOT, "work_dir/FLIR_raw/align")

FLIR2AIC = {
    "person": 0,
    "car": 6,
    "bicycle": 5,
    "dog": 2,            # no dog class in AIC -> animal
}
CATS = [
    {"id": 0, "name": "person"},
    {"id": 2, "name": "animal"},
    {"id": 5, "name": "bicycle"},
    {"id": 6, "name": "car"},
]


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


def convert_split(txt, ann_dir, img_dir, out_rgb, out_ir):
    images, annotations = [], []
    ann_id = 0
    n_skip_cls = 0
    with open(txt) as f:
        bases = [ln.strip() for ln in f if ln.strip()]
    for i, base in enumerate(bases):                      # base = FLIR_XXXXX_PreviewData
        rgb_name = base.replace("PreviewData", "RGB") + ".jpg"
        ir_name = base + ".jpeg"
        xml_name = base + ".xml"
        rgb_src = os.path.join(img_dir, rgb_name)
        ir_src = os.path.join(img_dir, ir_name)
        xml_src = os.path.join(ann_dir, xml_name)
        if not (os.path.exists(rgb_src) and os.path.exists(ir_src) and os.path.exists(xml_src)):
            continue
        ns = rgb_name                                     # shared filename for rgb/ + ir/
        link(os.path.join(out_rgb, ns), rgb_src)
        link(os.path.join(out_ir, ns), ir_src)
        W, H = Image.open(rgb_src).size
        images.append({"id": i, "file_name": ns, "width": W, "height": H})
        for (name, xmin, ymin, xmax, ymax) in parse_xml(xml_src):
            cid = FLIR2AIC.get(name.lower())
            if cid is None:
                n_skip_cls += 1
                continue
            annotations.append({
                "id": ann_id, "image_id": i, "category_id": cid,
                "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                "area": (xmax - xmin) * (ymax - ymin), "iscrowd": 0,
            })
            ann_id += 1
    return images, annotations, n_skip_cls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=os.path.join(ROOT, "work_dir/FLIR"))
    args = ap.parse_args()

    ann_dir = os.path.join(args.src, "Annotations")
    img_dir = os.path.join(args.src, "AnnotatedImages")
    out_rgb = os.path.join(args.out, "rgb")
    out_ir = os.path.join(args.out, "ir")
    os.makedirs(out_rgb, exist_ok=True)
    os.makedirs(out_ir, exist_ok=True)

    for split in ("train", "val"):
        txt = os.path.join(args.src, f"align_{'train' if split == 'train' else 'validation'}.txt")
        if not os.path.exists(txt):
            print(f"[flir] {txt} missing, skip {split}")
            continue
        imgs, anns, nskip = convert_split(txt, ann_dir, img_dir, out_rgb, out_ir)
        coco = {
            "images": imgs,
            "annotations": anns,
            "categories": CATS,
            "info": {"source": "FLIR aligned", "split": split},
            "licenses": [],
        }
        out_json = os.path.join(args.out, f"instances_flir_{split}.json")
        with open(out_json, "w") as f:
            json.dump(coco, f)
        print(f"[flir] {split}: {len(imgs)} imgs, {len(anns)} boxes "
              f"({nskip} skipped cls) -> {out_json}")


if __name__ == "__main__":
    main()
