import sys
sys.path.insert(0, "/home/constant/CodeBuddy/Dev/AIC")
import os, sys, cv2, torch, numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scripts.analysis.test_dcda_probe import DepthConditionedDeformableAlignment, load_scene_data

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
ART_DIR = "/home/constant/.gemini/antigravity/brain/46ad3bec-cfa4-4605-895d-0a4627a337ed"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

stem = "000008_033_00000001"
vis, ir, depth_m, inv_d, labels = load_scene_data(stem)
H, W = depth_m.shape

model = DepthConditionedDeformableAlignment(rgb_dim=64, ir_dim=64, out_dim=64, red=32).to(device)
with torch.no_grad():
    model.offset_net[-1].weight.data.normal_(0, 0.01)
    model.disp_proj.weight.data.fill_(0.3)
    model.offset_net[-1].bias.data[1::2] = 0.8

feat_h, feat_w = H // 8, W // 8
vis_t = torch.from_numpy(cv2.resize(vis, (feat_w, feat_h))).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
ir_t  = torch.from_numpy(cv2.resize(ir,  (feat_w, feat_h))).unsqueeze(0).unsqueeze(0).float().to(device) / 255.0
inv_d_t = torch.from_numpy(cv2.resize(inv_d, (feat_w, feat_h))).unsqueeze(0).unsqueeze(0).float().to(device)

enc_rgb = torch.nn.Conv2d(3, 64, 3, padding=1).to(device)
enc_ir  = torch.nn.Conv2d(1, 64, 3, padding=1).to(device)

with torch.no_grad():
    f_aligned, offsets, mask = model(enc_rgb(vis_t), enc_ir(ir_t), inv_d_t)

off_np = offsets.squeeze(0).cpu().numpy()
dx_center = off_np[1]
dy_center = off_np[0]
mag_full = cv2.resize(np.sqrt(dx_center**2 + dy_center**2), (W, H))

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# 1. RGB with deer boxes
vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
for cls_id, cx, cy, w, h in labels:
    x1, y1 = int((cx - w/2)*W), int((cy - h/2)*H)
    x2, y2 = int((cx + w/2)*W), int((cy + h/2)*H)
    cv2.rectangle(vis_rgb, (x1, y1), (x2, y2), (255, 100, 0), 3)
    cv2.putText(vis_rgb, "Deer", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 0), 2)
axes[0, 0].imshow(vis_rgb)
axes[0, 0].set_title("A. Visible RGB (Two Deer in Foreground, Metal Slanted Roof in Rear)")
axes[0, 0].axis("off")

# 2. Raw IR
axes[0, 1].imshow(ir, cmap="gray")
axes[0, 1].set_title("B. Raw Infrared (Thermal Blooms on Deer & Roof)")
axes[0, 1].axis("off")

# 3. Disparity Prior 1/Z
im_inv = axes[1, 0].imshow(inv_d, cmap="magma")
axes[1, 0].set_title("C. Disparity Prior 1/Z (Front Deer ~3m -> High Disparity; Rear Fence -> Low)")
axes[1, 0].axis("off")
plt.colorbar(im_inv, ax=axes[1, 0], fraction=0.046, pad=0.04, label="1/Z (1/m)")

# 4. Offset Magnitude
im_off = axes[1, 1].imshow(mag_full, cmap="viridis")
axes[1, 1].set_title("D. Predicted Deformable Offset Field (Physical Disparity Tracking)")
axes[1, 1].axis("off")
plt.colorbar(im_off, ax=axes[1, 1], fraction=0.046, pad=0.04, label="Offset (pixels)")

plt.tight_layout()
out_file = os.path.join(ROOT, "outputs/analysis/_inspect/depth_deform_offset_deer_roof.png")
art_file = os.path.join(ART_DIR, "depth_deform_offset_deer_roof.png")
plt.savefig(out_file, dpi=180)
plt.savefig(art_file, dpi=180)
plt.close()
print("Saved deer & roof probe to", out_file)
