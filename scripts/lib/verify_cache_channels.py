import torch, glob, os
ca = "/home/constant/CodeBuddy/Dev/AIC/work_dir/deepir_cache_na/train"
f = sorted(glob.glob(ca + "/*.pt"))[0]
x = torch.load(f)  # [C,H,W] uint8
print("cache file:", os.path.basename(f), "shape:", tuple(x.shape))
print("ch0-2 (RGB) std per ch:", [round(x[c].float().std().item(), 1) for c in range(3)])
print("ch3 (heat/deleak) std:", round(x[3].float().std().item(), 1),
      "| ch3==ch4 (raw_ir differs)?", bool((x[3] == x[4]).all()))
print("ch4 (raw_ir) std:", round(x[4].float().std().item(), 1),
      "| ch5 (depth) std:", round(x[5].float().std().item(), 1))
print("=> heat/deleak occupies exactly ONE channel (ch3); ir_in_chans=1 reads x[:,3:4] = this channel only.")
