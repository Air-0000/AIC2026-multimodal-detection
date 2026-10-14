import re

def curve(path):
    rows = {}
    for line in open(path):
        if "test_coco_eval_bbox" not in line:
            continue
        m = re.search(r'"epoch": (\d+)', line)
        if not m:
            continue
        nums = re.findall(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?", line.split("test_coco_eval_bbox", 1)[1])
        ep = int(m.group(1))
        if ep not in rows:  # resume 会有重复 epoch, 取第一次
            rows[ep] = float(nums[0])
    return rows

def losses(path):
    out = {}
    for line in open(path):
        m = re.search(r'"epoch": (\d+),', line)
        if not m:
            continue
        lm = re.search(r'"train_loss": ([0-9.]+)', line)
        if lm:
            ep = int(m.group(1))
            if ep not in out:
                out[ep] = float(lm.group(1))
    return out

O = "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/outputs/"
base = curve(O + "deim_hgnetv2_x_aic_obj2coco_ft/log.txt")
pki = curve(O + "deim_hgnetv2_x_aic_obj2coco_ft_pki/log.txt")
bl = losses(O + "deim_hgnetv2_x_aic_obj2coco_ft/log.txt")
pl = losses(O + "deim_hgnetv2_x_aic_obj2coco_ft_pki/log.txt")
print("ep | base_val pki_val  diff | base_loss pki_loss")
for ep in range(0, 15):
    b, p = base.get(ep), pki.get(ep)
    bs, ps = bl.get(ep), pl.get(ep)
    d = f"{(p-b)*100:+.1f}pt" if (b is not None and p is not None) else "-"
    print(f"{ep:2d} | {b} {p} {d} | {bs} {ps}")
