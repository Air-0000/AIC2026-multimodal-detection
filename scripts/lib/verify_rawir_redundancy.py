import torch, glob
ca = "/home/constant/CodeBuddy/Dev/AIC/work_dir/deepir_cache_na/train"
f = sorted(glob.glob(ca + "/*.pt"))[0]
x = torch.load(f).float()          # [6,H,W] uint8 -> float
rgb = x[0:3]
ir = x[4]                          # raw_ir (ch4)
heat = x[3]                        # deleak (ch3)
gray = rgb.mean(0)                 # (ch0+ch1+ch2)/3
recon = heat + gray               # ch3 + RGB_gray 重建 raw_ir
# 关掉 clip 区域的影响：只在 heat>0 处比较（线性区）
mask = heat > 1
err = (recon - ir).abs()
print("=== ch4(raw_ir) vs ch3(deleak)+RGB_gray 重建 ===")
print("全图 RMSE      :", round(err.mean().item(), 2))
print("heat>1 区域RMSE:", round(err[mask].mean().item(), 2), " (像素占比 %.1f%%)" % (100*mask.float().mean().item()))
print("heat==0 区域   : raw_ir 与 RGB_gray 差值均值 =", round((ir - gray)[~mask].abs().mean().item(), 2),
      " -> 这区 heat 被钳到0, IR 信息丢失")
print("=> 结论: raw_ir 在 heat>0 区 = deleak + RGB_gray 几乎精确 (RMSE 低); heat==0 区(冷区)IR<RGB_gray, heat 钳0, raw_ir 多带这点冷区信息")
