"""
Prepare a YOLO-format dataset for RF-DETR training (RGB-only baseline).

Creates work_dir/rfdetr_dataset with the Roboflow/YOLO layout RF-DETR expects:
    rfdetr_dataset/
        data.yaml
        train/images/*.png|.jpg   (symlink -> dataset/train/visible)
        train/labels/*.txt        (symlink -> dataset/train/labels)
        valid/images/...
        valid/labels/...

Labels are already in YOLO normalized format: [class_id, cx, cy, w, h].
We use symlinks to avoid copying ~2GB of imagery.
"""
import os
import glob
import random
import shutil

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
SRC_VISIBLE = os.path.join(ROOT, "dataset/train/visible")
SRC_LABELS = os.path.join(ROOT, "dataset/train/labels")
OUT = os.path.join(ROOT, "work_dir/rfdetr_dataset")

CLASSES = [
    "person", "boat", "animal", "seat", "sign", "bicycle",
    "car", "ball", "light", "garbage can", "uav", "tricycle",
]


def main():
    os.makedirs(OUT, exist_ok=True)

    imgs = sorted(glob.glob(os.path.join(SRC_VISIBLE, "*")))
    samples = []
    for imp in imgs:
        stem = os.path.splitext(os.path.basename(imp))[0]
        labp = os.path.join(SRC_LABELS, stem + ".txt")
        if os.path.exists(labp):
            samples.append((stem, imp, labp))

    print(f"[prepare] found {len(samples)} image+label pairs")

    random.seed(42)
    random.shuffle(samples)

    n_valid = 200
    valid = samples[:n_valid]
    train = samples[n_valid:]
    print(f"[prepare] train={len(train)} valid={len(valid)}")

    def link_split(split_samples, split):
        img_dir = os.path.join(OUT, split, "images")
        lab_dir = os.path.join(OUT, split, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lab_dir, exist_ok=True)
        for stem, imp, labp in split_samples:
            dst_img = os.path.join(img_dir, os.path.basename(imp))
            dst_lab = os.path.join(lab_dir, stem + ".txt")
            for dst, src in ((dst_img, imp), (dst_lab, labp)):
                if os.path.islink(dst) or os.path.exists(dst):
                    os.remove(dst)
                try:
                    os.symlink(src, dst)
                except OSError:
                    shutil.copy(src, dst)

    link_split(train, "train")
    link_split(valid, "valid")

    with open(os.path.join(OUT, "data.yaml"), "w") as f:
        f.write(f"nc: {len(CLASSES)}\n")
        f.write("names:\n")
        for c in CLASSES:
            f.write(f"  - {c}\n")
        f.write("train: train/images\n")
        f.write("val: valid/images\n")

    print(f"[prepare] dataset written to {OUT}")


if __name__ == "__main__":
    main()
