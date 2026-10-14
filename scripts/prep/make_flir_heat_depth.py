"""Build the multimodal inputs FLIR needs for eval, mirroring make_llvip_heat_depth.py.

FLIR (aligned) provides paired rgb/ir but NO depth, exactly like LLVIP. So:
  * heat channel = max(IR_gray - RGB_gray, 0)  (same recipe as prep_ir_heat.py)
  * depth channel = all-black (null)  -> LOWER bound for the 5ch recipe, since the
    5ch model saw real depth in 85% of AIC training steps.

Outputs (named to match the COCO `file_name`, e.g. FLIR_XXXXX_RGB.jpg):
    work_dir/FLIR_heat/        3ch png/jpg heat residual (single-channel when read)
    work_dir/FLIR_depth_null/  symlinks -> one black image
These feed the FLIR val_dataloader ir_folder / depth_folder, exactly like LLVIP.
"""
import os
import glob
import numpy as np
from PIL import Image
from multiprocessing import Pool

BASE = "/home/constant/CodeBuddy/Dev/AIC/work_dir/FLIR"
HEAT_OUT = "/home/constant/CodeBuddy/Dev/AIC/work_dir/FLIR_heat"
DEPTH_OUT = "/home/constant/CodeBuddy/Dev/AIC/work_dir/FLIR_depth_null"


def make_heat(name):
    dst = os.path.join(HEAT_OUT, name)
    if os.path.exists(dst):
        return 0
    v = np.array(Image.open(os.path.join(BASE, "rgb", name)).convert("RGB")).astype(np.float32).mean(axis=2)
    ir = np.array(Image.open(os.path.join(BASE, "ir", name)).convert("RGB")).astype(np.float32).mean(axis=2)
    if ir.shape != v.shape:
        ir = np.array(Image.fromarray(ir.astype(np.uint8)).resize((v.shape[1], v.shape[0]))).astype(np.float32)
    heat = np.clip(ir - v, 0, 255).astype(np.uint8)
    Image.fromarray(np.stack([heat, heat, heat], axis=-1)).save(dst, quality=95)
    return 1


def main():
    os.makedirs(HEAT_OUT, exist_ok=True)
    os.makedirs(DEPTH_OUT, exist_ok=True)
    names = sorted(os.listdir(os.path.join(BASE, "rgb")))
    print(f"[flir-heat] {len(names)} images", flush=True)
    done = 0
    with Pool(8) as p:
        for i, r in enumerate(p.imap_unordered(make_heat, names, chunksize=64)):
            done += r
            if (i + 1) % 1000 == 0:
                print(f"[flir-heat] {i+1}/{len(names)}", flush=True)
    print(f"[flir-heat] done: {done} -> {HEAT_OUT}", flush=True)

    black = os.path.join(DEPTH_OUT, "_black.png")
    if not os.path.exists(black):
        ref = os.path.join(BASE, "rgb", names[0])
        w, h = Image.open(ref).size
        Image.fromarray(np.zeros((h, w), dtype=np.uint8), mode="L").save(black)
    made = 0
    for n in names:
        dst = os.path.join(DEPTH_OUT, n)
        if os.path.lexists(dst):
            continue
        os.symlink(black, dst)
        made += 1
    print(f"[flir-depth-null] links made={made} total={len(names)}", flush=True)


if __name__ == "__main__":
    main()
