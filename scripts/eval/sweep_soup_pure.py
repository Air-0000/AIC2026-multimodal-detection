import torch
import copy
import os, sys

sys.path.insert(0, '/root/autodl-tmp/AIC/vendor/DEIMv2')
sys.path.insert(0, '/root/autodl-tmp/AIC/vendor/Depth-Anything-V2')
sys.path.insert(0, '/root/autodl-tmp/AIC/scripts/lib')
sys.path.insert(0, '/root/autodl-tmp/AIC')

from engine.core import YAMLConfig
from engine.solver.det_engine import evaluate

def create_soup_in_memory(base_ckpt, pure_ckpt, alpha=0.5):
    """
    Interpolate weights between base checkpoint and pure checkpoint.
    alpha = 1.0 -> purely pure_ckpt
    alpha = 0.0 -> purely base_ckpt
    """
    pure_model = copy.deepcopy(pure_ckpt['ema']['module'] if 'ema' in pure_ckpt and 'module' in pure_ckpt['ema'] else pure_ckpt['model'])
    base_model = base_ckpt['ema']['module'] if 'ema' in base_ckpt and 'module' in base_ckpt['ema'] else base_ckpt['model']
    
    soup_model = {}
    for k in pure_model.keys():
        if k in base_model and base_model[k].shape == pure_model[k].shape:
            soup_model[k] = (1.0 - alpha) * base_model[k].float() + alpha * pure_model[k].float()
        else:
            soup_model[k] = pure_model[k]
    return soup_model

def main():
    base_path = '/root/autodl-tmp/AIC/vendor/DEIMv2/outputs/deim_hgnetv2_x_aic_sovereign_v14_soup/best_stg1.pth'
    pure_path = '/root/autodl-tmp/AIC/vendor/DEIMv2/outputs/deim_hgnetv2_x_aic_sovereign_pure_rtx6000/best_stg1.pth'
    
    print("Loading Base (V14 Soup) and Pure (Purified 8-epoch) checkpoints...")
    base_ckpt = torch.load(base_path, map_location='cpu')
    pure_ckpt = torch.load(pure_path, map_location='cpu')
    
    cfg = YAMLConfig('/root/autodl-tmp/AIC/vendor/DEIMv2/configs/deimv2/deim_hgnetv2_x_aic_sovereign_pure_rtx6000.yml')
    device = torch.device('cuda:0')
    model = cfg.model.to(device)
    
    val_dataloader = cfg.val_dataloader
    criterion = cfg.criterion
    postprocessor = cfg.postprocessor
    
    results = []
    alphas = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    best_map = 0.0
    best_alpha = 1.0
    best_weights = None
    
    for alpha in alphas:
        print(f"\n--- Testing Sovereign Pure Soup with Alpha = {alpha:.2f} ---")
        soup_weights = create_soup_in_memory(base_ckpt, pure_ckpt, alpha=alpha)
        missing, unexpected = model.load_state_dict(soup_weights, strict=False)
        print(f"Loaded soup (missing={len(missing)}, unexpected={len(unexpected)})")
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
        
        if mAP > best_map:
            best_map = mAP
            best_alpha = alpha
            best_weights = soup_weights
            
    print("\n" + "="*50)
    print("SOVEREIGN PURE MODEL SOUP SWEEP SUMMARY:")
    for alpha, mAP, AP50, AP75, AP_s in results:
        print(f"Alpha {alpha:.2f}: mAP = {mAP*100:.2f}%, AP50 = {AP50*100:.2f}%, AP75 = {AP75*100:.2f}%, AP_s = {AP_s*100:.2f}%")
    print(f"BEST ALPHA: {best_alpha:.2f} with mAP = {best_map*100:.2f}%")
    print("="*50)
    
    out_dir = '/root/autodl-tmp/AIC/vendor/DEIMv2/outputs/deim_hgnetv2_x_aic_sovereign_pure_soup'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'best_stg1.pth')
    
    save_ckpt = copy.deepcopy(pure_ckpt)
    if 'ema' in save_ckpt and 'module' in save_ckpt['ema']:
        save_ckpt['ema']['module'] = best_weights
    save_ckpt['model'] = best_weights
    torch.save(save_ckpt, out_path)
    print(f"Saved best champion soup weights to: {out_path}")

if __name__ == '__main__':
    main()
