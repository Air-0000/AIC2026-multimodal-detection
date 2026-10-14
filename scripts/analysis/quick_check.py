from PIL import Image
import numpy as np, os

VALID = "/home/constant/CodeBuddy/Dev/AIC/work_dir/deimv2_dataset/valid/images"
N005 = "/home/constant/CodeBuddy/Dev/AIC/data_aux/aic_valid_rgb_night005"

def mean_bright(d):
    fs = sorted(f for f in os.listdir(d) if f.lower().endswith((".jpg", ".png")))
    vals = []
    for f in fs[:8]:
        a = np.asarray(Image.open(os.path.join(d, f)).convert("RGB")).astype(float) / 255.0
        vals.append(a.mean())
    return np.mean(vals), fs[:3]

o_mean, of = mean_bright(VALID)
n_mean, nf = mean_bright(N005)
print("orig files sample:", of)
print("n005 files sample:", nf)
print(f"orig_mean={o_mean:.3f}  n005_mean={n_mean:.3f}  ratio={n_mean/o_mean:.4f}")
