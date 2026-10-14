"""Build a flat refined-depth dir with ORIGINAL basenames so the detector's
depth loader (looks up <base>.png / <base>.jpg in depth_folder) finds them.

For each file in dataset/train/depth:
  - if IR-guided refined (<base>_refined.png) exists -> use it
  - else (e.g. the few 8-bit .jpg depths depth_refine.py skips) -> copy raw
Output is always <base>.png (16-bit kept, 8-bit kept as 8-bit) so the
detector reads a consistent .png.
"""
import os, cv2
ROOT = "/home/constant/CodeBuddy/Dev/AIC"
SRC_DIR = os.path.join(ROOT, "dataset/train/depth")
REF_DIR = os.path.join(ROOT, "work_dir/depth_refined_irheat/train")
OUT_DIR = os.path.join(ROOT, "work_dir/depth_refined_irheat_png")
os.makedirs(OUT_DIR, exist_ok=True)

n_ref = n_fb = 0
for f in sorted(os.listdir(SRC_DIR)):
    if not f.lower().endswith((".png", ".jpg", ".jpeg")):
        continue
    base = os.path.splitext(f)[0]
    ref = os.path.join(REF_DIR, base + "_refined.png")
    if os.path.exists(ref):
        src, used_ref = ref, True
    else:
        src, used_ref = os.path.join(SRC_DIR, f), False
    im = cv2.imread(src, cv2.IMREAD_UNCHANGED)
    if im is None:
        print("SKIP (unreadable)", f); continue
    cv2.imwrite(os.path.join(OUT_DIR, base + ".png"), im)
    n_ref += used_ref; n_fb += (not used_ref)
print(f"flat done: refined={n_ref} fallback_raw={n_fb} total={n_ref+n_fb}")
