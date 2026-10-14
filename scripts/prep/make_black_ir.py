"""Create an all-black IR folder mirroring LLVIP/rgb filenames.

Used for the "no-IR" ablation: same 4ch checkpoint, IR channel fed zeros,
so the only variable is the presence of IR information.
"""
import os
import glob
from PIL import Image
import numpy as np

BASE = "/home/constant/CodeBuddy/Dev/AIC/work_dir/LLVIP"
OUT = "/home/constant/CodeBuddy/Dev/AIC/work_dir/LLVIP_ir_black"
os.makedirs(OUT, exist_ok=True)

ref = sorted(glob.glob(os.path.join(BASE, "rgb", "*.jpg")))[0]
w, h = Image.open(ref).size
black_path = os.path.join(OUT, "_black.jpg")
if not os.path.exists(black_path):
    Image.fromarray(np.zeros((h, w), dtype=np.uint8)).save(black_path)
print("black image:", black_path, (w, h))

names = [os.path.basename(p) for p in ref_list] if False else None
names = sorted(os.listdir(os.path.join(BASE, "rgb")))
made = 0
for n in names:
    dst = os.path.join(OUT, n)
    if os.path.lexists(dst):
        continue
    os.symlink(black_path, dst)
    made += 1
print(f"links: made={made} total={len(names)}")
