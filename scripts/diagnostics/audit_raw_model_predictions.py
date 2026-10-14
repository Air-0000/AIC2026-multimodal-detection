import os
import sys
import glob
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T

sys.path.insert(0, "/root/autodl-tmp/AIC/vendor/DEIMv2")
from engine.core import YAMLConfig

cfg = YAMLConfig("/root/autodl-tmp/AIC/vendor/DEIMv2/configs/deimv2/sovereign_v26_model_repair.yml")
model = cfg.model.cuda().eval()
postprocessor = cfg.postprocessor.cuda().eval()

ckpt_path = "/root/autodl-tmp/AIC/vendor/DEIMv2/outputs/sovereign_v26_model_repair_2048/last.pth"
print(f"Loading {ckpt_path}...")
ck = torch.load(ckpt_path, map_location="cpu")
w = ck["ema"]["module"] if ("ema" in ck and "module" in ck["ema"]) else ck.get("model", ck)
model.load_state_dict(w, strict=False)

cache_dir = "/dev/shm/aic_cache"
cases = ["00000432", "000003_018_00000333", "shuming_96_00000120", "002808", "001539"]

CLASS_NAMES = {
    0: "person", 1: "boat", 2: "animal", 3: "seat", 4: "sign", 5: "bicycle",
    6: "car", 7: "ball", 8: "light", 9: "garbage can", 10: "uav", 11: "tricycle"
}

MEAN_6CH = torch.tensor([0.485, 0.456, 0.406, 0.5, 0.5, 0.5]).view(6, 1, 1)
STD_6CH = torch.tensor([0.229, 0.224, 0.225, 0.5, 0.5, 0.5]).view(6, 1, 1)

def get_sample(stem):
    vis_cands = glob.glob(os.path.join(cache_dir, "test_visible", f"{stem}.*"))
    vis_pil = Image.open(vis_cands[0]).convert("RGB")
    orig_w, orig_h = vis_pil.size
    rgb_t = T.functional.pil_to_tensor(vis_pil).float() / 255.0

    ir_cands = glob.glob(os.path.join(cache_dir, "test_aligned_ir", f"{stem}.*"))
    ir_t = T.functional.pil_to_tensor(Image.open(ir_cands[0]).convert("L")).float() / 255.0

    lbg_cands = glob.glob(os.path.join(cache_dir, "test_ir_lbg", f"{stem}.*"))
    lbg_t = T.functional.pil_to_tensor(Image.open(lbg_cands[0]).convert("L")).float() / 255.0 if lbg_cands else ir_t.clone()

    dp_p = glob.glob(os.path.join(cache_dir, "test_depth_p0", f"{stem}.*"))[0]
    dp_arr = torch.from_numpy(np.array(Image.open(dp_p)).astype("float32")).unsqueeze(0) / 255.0

    s = torch.cat([rgb_t, ir_t, lbg_t, dp_arr], dim=0)
    s = F.interpolate(s.unsqueeze(0), size=(1152, 2048), mode="bilinear", align_corners=False).squeeze(0)
    s = (s - MEAN_6CH) / STD_6CH
    return s.cuda(), orig_w, orig_h

for stem in cases:
    s, orig_w, orig_h = get_sample(stem)
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(s.unsqueeze(0))
        res = postprocessor(outputs, torch.tensor([[float(orig_w), float(orig_h)]], device="cuda"), s.unsqueeze(0))[0]
    
    labs = res["labels"].cpu().numpy()
    boxes = res["boxes"].float().cpu().numpy()
    scores = res["scores"].float().cpu().numpy()
    
    print(f"\n=== STEM: {stem} (orig={orig_w}x{orig_h}) ===")
    count = 0
    for l, b, sc in zip(labs, boxes, scores):
        if sc >= 0.20:
            count += 1
            cx = float(((b[0]+b[2])/2)/orig_w)
            cy = float(((b[1]+b[3])/2)/orig_h)
            w = float((b[2]-b[0])/orig_w)
            h = float((b[3]-b[1])/orig_h)
            cname = CLASS_NAMES.get(int(l), str(l))
            print(f"  {cname} ({l}): cx={cx:.4f}, cy={cy:.4f}, w={w:.4f}, h={h:.4f}, sc={sc:.4f}")
    if count == 0:
        print("  [None >= 0.20]")
