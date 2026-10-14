import torch
import copy
import os, sys

sys.path.insert(0, '/root/autodl-tmp/AIC/vendor/DEIMv2')
sys.path.insert(0, '/root/autodl-tmp/AIC/vendor/Depth-Anything-V2')
sys.path.insert(0, '/root/autodl-tmp/AIC/scripts/lib')
sys.path.insert(0, '/root/autodl-tmp/AIC')

from engine.core import YAMLConfig
from engine.solver.det_engine import evaluate

def create_soup_in_memory(v12_ckpt, v14_ckpt, alpha=0.5):
    soup_model = copy.deepcopy(v14_ckpt['ema']['module'] if 'ema' in v14_ckpt and 'module' in v14_ckpt['ema'] else v14_ckpt['model'])
    v12_model = v12_ckpt['ema']['module'] if 'ema' in v12_ckpt and 'module' in v12_ckpt['ema'] else v12_ckpt['model']
    
    for k in list(soup_model.keys()):
        if k in v12_model:
            soup_model[k] = (1.0 - alpha) * v12_model[k] + alpha * soup_model[k]
        else:
            soup_model[k] = soup_model[k] * alpha
    return soup_model

def main():
    v12_path = '/root/autodl-tmp/AIC/vendor/DEIMv2/outputs/deim_hgnetv2_x_aic_sovereign_v12_rtx6000/best_stg1.pth'
    v14_path = '/root/autodl-tmp/AIC/vendor/DEIMv2/outputs/deim_hgnetv2_x_aic_sovereign_v14_rtx6000/best_stg1.pth'
    
    print("Loading V12 and V14 checkpoints...")
    v12_ckpt = torch.load(v12_path, map_location='cpu')
    v14_ckpt = torch.load(v14_path, map_location='cpu')
    
    cfg = YAMLConfig('/root/autodl-tmp/AIC/vendor/DEIMv2/configs/deimv2/deim_hgnetv2_x_aic_sovereign_v14_rtx6000.yml')
    device = torch.device('cuda:0')
    model = cfg.model.to(device)
    
    val_dataloader = cfg.val_dataloader
    criterion = cfg.criterion
    postprocessor = cfg.postprocessor
    
    results = []
    alphas = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    for alpha in alphas:
        print(f"\n--- Testing Sovereign V12 + V14 Soup with Alpha = {alpha:.2f} ---")
        soup_weights = create_soup_in_memory(v12_ckpt, v14_ckpt, alpha=alpha)
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
        
    print("\n" + "="*50)
    print("SOVEREIGN V12 + V14 MODEL SOUP SWEEP SUMMARY:")
    for alpha, mAP, AP50, AP75, AP_s in results:
        print(f"Alpha {alpha:.2f}: mAP = {mAP*100:.2f}%, AP50 = {AP50*100:.2f}%, AP75 = {AP75*100:.2f}%, AP_s = {AP_s*100:.2f}%")
    print("="*50)

if __name__ == '__main__':
    main()
