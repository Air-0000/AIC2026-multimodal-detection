"""Side-by-side color comparison: raw | DA | morph (fixed 0..19999 range).

Emits, per chosen frame, a full-resolution 3-panel strip and a 2x zoomed
crop strip, so edges can be inspected for collapse. Output to
work_dir/depth_compare_da_morph/.
"""
import os
import glob
import cv2
import numpy as np

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
VMIN, VMAX = 0.0, 19999.0
OUT = os.path.join(ROOT, "work_dir/depth_compare_da_morph")
os.makedirs(OUT, exist_ok=True)


def colorize(d):
    dn = np.clip((d.astype(np.float32) - VMIN) / (VMAX - VMIN + 1e-6), 0, 1) * 255
    return cv2.applyColorMap(dn.astype(np.uint8), cv2.COLORMAP_TURBO)


def label(img, text):
    c = img.copy()
    cv2.putText(c, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (255, 255, 255), 2, cv2.LINE_AA)
    return c


def load16(p):
    d = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if d is None or d.ndim == 3:
        return None
    return d.astype(np.float32)


def main():
    da_dir = os.path.join(ROOT, "work_dir/depth_refined_da_guided/train")
    mo_dir = os.path.join(ROOT, "work_dir/depth_refined_morph/train")
    raw_dir = os.path.join(ROOT, "dataset/train/depth")
    bases = sorted(os.path.splitext(os.path.basename(x))[0]
                   for x in glob.glob(os.path.join(da_dir, "*.png"))
                   if not x.endswith("_color.png"))
    # pick ~6 evenly spaced frames
    picks = [bases[i] for i in range(0, len(bases), max(1, len(bases) // 6))][:6]
    print("picks:", picks)
    for b in picks:
        raw = load16(os.path.join(raw_dir, b + ".png"))
        da = load16(os.path.join(da_dir, b + ".png"))
        mo = load16(os.path.join(mo_dir, b + ".png"))
        if raw is None or da is None or mo is None:
            print("  ! skip", b)
            continue
        cr, cda, cm = colorize(raw), colorize(da), colorize(mo)
        full = np.hstack([label(cr, "raw"), label(cda, "DA"), label(cm, "morph")])
        cv2.imwrite(os.path.join(OUT, b + "_full.png"), full)
        # 2x zoomed crop: mid-lower scene where objects sit
        H, W = raw.shape
        y0, y1 = int(0.45 * H), int(0.85 * H)
        x0, x1 = int(0.25 * W), int(0.75 * W)
        z = 2
        cz = lambda c: cv2.resize(c[y0:y1, x0:x1], (0, 0), fx=z, fy=z,
                                  interpolation=cv2.INTER_NEAREST)
        crop = np.hstack([label(cz(cr), "raw"), label(cz(cda), "DA"),
                          label(cz(cm), "morph")])
        cv2.imwrite(os.path.join(OUT, b + "_crop.png"), crop)
        print("  wrote", b)
    print("DONE ->", OUT)


if __name__ == "__main__":
    main()
