import os
from PIL import Image
import numpy as np

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
s = "00000004"
a = np.array(Image.open(f"{ROOT}/dataset/train/infrared_deleak/{s}.png").convert("L")).astype(np.int32)
b = np.array(Image.open(f"{ROOT}/work_dir/deimv2_heat_na/train/images/{s}.png").convert("L")).astype(np.int32)
print("dataset/infrared_deleak shape/std:", a.shape, round(a.std(), 1))
print("deimv2_heat_na      shape/std:", b.shape, round(b.std(), 1))
print("abs diff max:", int((a - b).abs().max()), " mean:", round((a - b).abs().mean(), 2))

na = f"{ROOT}/work_dir/deimv2_heat_na/train/images"
de = f"{ROOT}/dataset/train/infrared_deleak"
na_n = len(os.listdir(na)); de_n = len(os.listdir(de))
print("file count deimv2_heat_na:", na_n, " infrared_deleak:", de_n, " match:", na_n == de_n)
