import torch
import copy
import os, sys

sys.path.insert(0, '/root/autodl-tmp/AIC/vendor/DEIMv2')
sys.path.insert(0, '/root/autodl-tmp/AIC/vendor/Depth-Anything-V2')
sys.path.insert(0, '/root/autodl-tmp/AIC/scripts/lib')
sys.path.insert(0, '/root/autodl-tmp/AIC')

from engine.core import YAMLConfig
from engine.solver.det_engine import evaluate

def create_soup_in_memory(v12_ckpt, v13_ckpt, alpha=0.5):
    soup_model = copy.deepcopy(v13_ckpt['ema']['module'] if 'ema' in v13_ckpt and 'module' in v13_ckpt['ema'] else v13_ckpt['model'])
    v12_model = v12_ckpt['ema']['module'] if 'ema' in v12_ckpt and 'module' in v12_ckpt['ema'] else v12_ckpt['model']
    
    for k in list(soup_model.keys()):
        if k in v12_model:
            soup_model[k] = (1.0 - alpha) * v12_model[k] + alpha * soup_model[k]
        else:
            soup_model[k] = soup_model[k] * alpha
    return soup_model

def main():
    v12_path = '/root/autodl-tmp/AIC/vendor/DEIMv2/outputs/deim_hgnetv2_x_aic_sovereign_v12_rtx6000/best_stg1.pth'
    v13_path = '/root/autodl-tmp/AIC/vendor/DEIMv2/outputs/deim_hgnetv2_x_aic_sovereign_v13_rtx6000/best_stg1.pth'
    
    print("Loading V12 and V13 checkpoints...")
    v12_ckpt = torch.load(v12_path, map_location='cpu')
    v13_ckpt = torch.load(v13_path, map_location='cpu')
    
    cfg = YAMLConfig('/root/autodl-tmp/AIC/vendor/DEIMv2/configs/deimv2/deim_hgnetv2_x_aic_sovereign_v13_rtx6000.yml')
    device = torch.device('cuda:0')
    model = cfg.model.to(device)
    
    val_dataloader = cfg.val_dataloader
    criterion = cfg.criterion
    postprocessor = cfg.postprocessor
    
    results = []
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    for alpha in alphas:
        print(f"\n--- Testing Model Soup with Alpha = {alpha:.2f} ---")
        soup_weights = create_soup_in_memory(v12_ckpt, v13_ckpt, alpha=alpha)
        model.load_state_dict(soup_weights, strict=False)
        model.eval()
        
        # Reset evaluator
        evaluator = cfg.evaluator
        evaluator.coco_eval = {}
        
        stats, _ = evaluate(model, criterion, postprocessor, val_dataloader, evaluator, device)
        mAP = stats['coco_eval_bbox'][0]
        AP50 = stats['coco_eval_bbox'][1]
        AP75 = stats['coco_eval_bbox'][2]
        AP_s = stats['coco_eval_bbox'][3]
        print(f">>> Result Alpha={alpha:.2f}: mAP={mAP:.4f}, AP50={AP50:.4f}, AP75={AP75:.4f}, AP_s={AP_s:.4f}")
        results.append((alpha, mAP, AP50, AP75, AP_s))

    print("\n" + "="*60)
    print("FULL MODEL SOUP SWEEP SUMMARY:")
    print("="*60)
    for a, m, a50, a75, as_ in results:
        print(f"Alpha: {a:.2f} | mAP: {m*100:.2f}% | AP50: {a50*100:.2f}% | AP75: {a75*100:.2f}% | AP_s: {as_*100:.2f}%")
    best = max(results, key=lambda x: x[1])
    print(f"\nPEAK SOUP: Alpha={best[0]:.2f} -> mAP: {best[1]*100:.2f}% (AP50: {best[2]*100:.2f}%)")
    print("="*60)

if __name__ == '__main__':
    main()
