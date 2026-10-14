import os, sys, argparse, time
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--max-samples', type=int, default=10, help='Max samples to evaluate, -1 for all')
parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
parser.add_argument('--mem-fraction', type=float, default=0.04, help='Max memory fraction cap (0.04 = ~3.9 GB)')
args = parser.parse_args()

# Hard safety cap on GPU 0 memory
torch.cuda.set_per_process_memory_fraction(args.mem_fraction, 0)
total_dev_mem = torch.cuda.get_device_properties(0).total_memory
print(f'[SafetyGuard] Max GPU memory capped at: {total_dev_mem * args.mem_fraction / 1e9:.2f} GB ({args.mem_fraction*100:.1f}%)')

_root = '/root/autodl-tmp/AIC'
sys.path.insert(0, os.path.join(_root, 'vendor/DEIMv2'))
sys.path.insert(0, os.path.join(_root, 'scripts/lib'))
sys.path.insert(0, os.path.join(_root, 'vendor/Depth-Anything-V2'))
os.chdir(os.path.join(_root, 'vendor/DEIMv2'))

import torch.multiprocessing as mp
try:
    mp.set_sharing_strategy('file_system')
except Exception:
    pass

from engine.misc import dist_utils
from engine.core import YAMLConfig
from engine.solver import TASKS
from engine.misc.logger import MetricLogger

def run():
    cfg_path = 'configs/deimv2/sovereign_v23_zero_distort_2048_fast32.yml'
    ckpt_path = 'outputs/sovereign_v23_zero_distort_2048_fast32/best_stg1.pth'
    full_ann = '/root/autodl-tmp/AIC/work_dir/deimv2_dataset/train/annotations/instances_train_quality1998.json'

    cfg = YAMLConfig(cfg_path, resume=ckpt_path)
    cfg.yaml_cfg['val_dataloader']['total_batch_size'] = args.batch_size
    cfg.yaml_cfg['val_dataloader']['num_workers'] = 2
    cfg.yaml_cfg['val_dataloader']['dataset']['ann_file'] = full_ann
    cfg.yaml_cfg['output_dir'] = 'outputs/sovereign_v23_zero_distort_2048_fast32/eval_full2000'

    for bb_name in ['HGNetv2', 'HGNetv2MultiModal', 'HGNetv2ViTDualStream']:
        if bb_name in cfg.yaml_cfg:
            cfg.yaml_cfg[bb_name]['pretrained'] = False

    dist_utils.setup_distributed(0, 'builtin', seed=0)
    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    solver.eval()
    
    module = solver.ema.module if solver.ema else solver.model
    module.eval()
    
    total_imgs = len(solver.val_dataloader.dataset)
    target_count = total_imgs if args.max_samples <= 0 else min(args.max_samples, total_imgs)
    print(f'Starting safe evaluation on {target_count}/{total_imgs} images (batch_size={args.batch_size}, bfloat16 AMP)...')
    
    metric_logger = MetricLogger(delimiter='  ')
    coco_evaluator = solver.evaluator
    coco_evaluator.cleanup()
    iou_types = coco_evaluator.iou_types
    device = solver.device
    
    start_t = time.time()
    processed = 0
    with torch.no_grad():
        for samples, targets in metric_logger.log_every(solver.val_dataloader, 50, 'Eval:'):
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = module(samples)
            
            orig_target_sizes = torch.stack([t['orig_size'] for t in targets], dim=0)
            results = solver.postprocessor(outputs, orig_target_sizes, samples)
            res = {target['image_id'].item(): output for target, output in zip(targets, results)}
            coco_evaluator.update(res)
            
            processed += len(targets)
            if args.max_samples > 0 and processed >= args.max_samples:
                print(f'Reached sample limit: {processed}')
                break
    
    dur = time.time() - start_t
    print(f'Evaluation inference completed in {dur:.2f}s ({processed / max(dur, 0.001):.2f} img/s)')
    print('Synchronizing and accumulating evaluation results...')
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    
    if 'bbox' in coco_evaluator.coco_eval:
        from pathlib import Path
        import json
        stats = coco_evaluator.coco_eval['bbox'].stats.tolist()
        out_dir = Path('outputs/sovereign_v23_zero_distort_2048_fast32/eval_full2000')
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / 'eval_stats.json', 'w') as f:
            json.dump({'bbox_stats': stats}, f, indent=2)
        print(f'Saved stats to {out_dir / "eval_stats.json"}')
    
    peak_mem = torch.cuda.max_memory_allocated() / 1e6
    reserved_mem = torch.cuda.max_memory_reserved() / 1e6
    print(f'[Completed] Peak Allocated GPU Memory: {peak_mem:.1f} MB, Reserved: {reserved_mem:.1f} MB')
    dist_utils.cleanup()

if __name__ == '__main__':
    run()
