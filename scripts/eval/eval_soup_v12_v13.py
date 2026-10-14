import torch
import copy
import sys
import os

def create_soup(v12_path, v13_path, out_path, alpha=0.5):
    print(f"Loading V12 from {v12_path}...")
    v12 = torch.load(v12_path, map_location='cpu')
    print(f"Loading V13 from {v13_path}...")
    v13 = torch.load(v13_path, map_location='cpu')

    soup = copy.deepcopy(v13)
    
    # Interpolate model weights
    v12_model = v12['model']
    v13_model = v13['model']
    soup_model = soup['model']
    
    interp_count = 0
    v13_only_count = 0
    for k in v13_model.keys():
        if k in v12_model:
            soup_model[k] = (1.0 - alpha) * v12_model[k] + alpha * v13_model[k]
            interp_count += 1
        else:
            # Scale new module weights by alpha so they act as a gentle boost
            soup_model[k] = v13_model[k] * alpha
            v13_only_count += 1

    # Also interpolate EMA if present
    if 'ema' in v13 and 'ema' in v12 and 'module' in v13['ema'] and 'module' in v12['ema']:
        v12_ema = v12['ema']['module']
        v13_ema = v13['ema']['module']
        soup_ema = soup['ema']['module']
        for k in v13_ema.keys():
            if k in v12_ema:
                soup_ema[k] = (1.0 - alpha) * v12_ema[k] + alpha * v13_ema[k]
            else:
                soup_ema[k] = v13_ema[k] * alpha

    print(f"Interpolated {interp_count} shared tensors, preserved {v13_only_count} new V13 tensors.")
    torch.save(soup, out_path)
    print(f"Saved Model Soup (alpha={alpha}) to {out_path}!")

if __name__ == '__main__':
    v12_path = '/root/autodl-tmp/AIC/vendor/DEIMv2/outputs/deim_hgnetv2_x_aic_sovereign_v12_rtx6000/best_stg1.pth'
    v13_path = '/root/autodl-tmp/AIC/vendor/DEIMv2/outputs/deim_hgnetv2_x_aic_sovereign_v13_rtx6000/best_stg1.pth'
    out_dir = '/root/autodl-tmp/AIC/vendor/DEIMv2/outputs/deim_hgnetv2_x_aic_sovereign_v13_soup'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'soup_alpha05.pth')
    create_soup(v12_path, v13_path, out_path, alpha=0.5)
