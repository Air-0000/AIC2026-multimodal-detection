import torch
import sys
import os

sys.path.insert(0, '/root/autodl-tmp/AIC/vendor/DEIMv2')
os.chdir('/root/autodl-tmp/AIC/vendor/DEIMv2')

from engine.core import YAMLConfig

cfg = YAMLConfig('configs/deimv2/sovereign_v25_model_repair_midboost.yml')
model = cfg.model.cuda()
criterion = cfg.criterion.cuda()

# Test various configurations to find ~88-92 GB VRAM
for H, W in [(1152, 2048), (896, 1600), (768, 1344), (640, 1152)]:
    print(f"\n=================== Testing Resolution {H}x{W} ===================")
    for bs in [4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            x = torch.randn(bs, 6, H, W, device='cuda')
            targets = [{'boxes': torch.rand(10, 4, device='cuda'), 'labels': torch.randint(0, 12, (10,), device='cuda')} for _ in range(bs)]
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                out = model(x, targets)
                loss_dict = criterion(out, targets)
                loss = sum(loss_dict.values())
            loss.backward()
            peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
            reserved_mb = torch.cuda.max_memory_reserved() / (1024**2)
            gb = reserved_mb / 1024.0
            print(f"Res: {H}x{W}, BS: {bs:2d} -> Peak: {peak_mb:7.1f} MB, Reserved: {reserved_mb:7.1f} MB ({gb:5.2f} GB)")
            if gb >= 86.0 and gb <= 94.0:
                print(f"===> [MATCH] PERFECT TARGET at Res {H}x{W}, BS {bs}: {gb:.2f} GB!")
        except torch.cuda.OutOfMemoryError:
            print(f"Res: {H}x{W}, BS: {bs:2d} -> OOM!")
            break
        except Exception as e:
            print(f"Res: {H}x{W}, BS: {bs:2d} -> Error: {e}")
            break
