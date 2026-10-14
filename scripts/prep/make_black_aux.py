"""All-black mirrors of the AIC valid aux channels (IR heat / depth).

Used to ablate each auxiliary modality individually while evaluating the 5ch
checkpoint on AIC val (where REAL depth is available, unlike LLVIP).
"""
import os
import numpy as np
from PIL import Image

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
VALID = os.path.join(ROOT, "work_dir/deimv2_dataset/valid/images")

TARGETS = {
    "data_aux/aic_valid_ir_black": "ir",
    "data_aux/aic_valid_depth_black": "depth",
}

names = sorted(os.listdir(VALID))
print("valid images:", len(names))
w, h = Image.open(os.path.join(VALID, names[0])).size
print("size:", (w, h))

for rel, tag in TARGETS.items():
    out = os.path.join(ROOT, rel)
    os.makedirs(out, exist_ok=True)
    black = os.path.join(out, "_black.png")
    if not os.path.exists(black):
        Image.fromarray(np.zeros((h, w), dtype=np.uint8), mode="L").save(black)
    made = 0
    for n in names:
        dst = os.path.join(out, n)
        if os.path.lexists(dst):
            continue
        os.symlink(black, dst)
        made += 1
    print(f"{tag}: links made={made} total={len(names)} -> {out}")
