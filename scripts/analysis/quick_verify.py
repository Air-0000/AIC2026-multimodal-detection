from PIL import Image
import numpy as np, os
base = "/home/constant/CodeBuddy/Dev/AIC/work_dir"
for d in ["aic_valid_rgb_night005", "aic_valid_rgb_night002", "aic_valid_rgb_night001", "aic_valid_rgb_night0005"]:
    fs = sorted(f for f in os.listdir(base + "/" + d) if f.lower().endswith((".jpg", ".png")))
    m = np.mean([np.asarray(Image.open(base + "/" + d + "/" + f).convert("RGB")).astype(float).mean() / 255 for f in fs])
    print(f"{d:28} mean_brightness={m:.4f}")
