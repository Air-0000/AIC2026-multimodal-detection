"""COCO val2017 check of the converted Co-DINO ViT-L checkpoint under the mmdet 3.x port.

Builds the 80-class model of co_dino_vit_l_aic_rgb_1800.py with upstream's test settings (resize to fit 2048 x 1280,
single scale, no TTA, up to 1000 detections per image) and runs mmengine's test loop on val2017. Sense-X reports
65.4 box AP for this checkpoint (65.9 with TTA) through mmdet's CocoMetric, which reads AP at maxDets 1000; the
metric is reported both that way (coco/bbox_mAP, comparable) and at the COCO-standard maxDets 100
(coco_maxdet100/bbox_mAP).

Run from the repo root with codino_env.sh sourced:
    python scripts/train/codino/eval_codino_coco.py --coco /dev/shm/coco \
        --checkpoint weights/codino/co_dino_vit_l_o365tococo_mmdet3.pth --work-dir work_dir/codino/coco_eval
--coco must hold val2017/ and annotations/instances_val2017.json.
"""
import argparse
import os

CONFIG = 'scripts/train/codino/co_dino_vit_l_aic_rgb_1800.py'


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--coco', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--work-dir', required=True)
    p.add_argument('--config', default=CONFIG)
    p.add_argument('--limit', type=int, default=0, help='first N images only (quick check)')
    a = p.parse_args()

    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmdet.utils import register_all_modules
    register_all_modules()
    cfg = Config.fromfile(a.config)
    cfg.model.query_head.num_classes = 80
    cfg.model.roi_head[0].bbox_head.num_classes = 80
    cfg.model.bbox_head[0].num_classes = 80
    cfg.model.test_cfg[0].max_per_img = 1000
    ann = os.path.join(a.coco, 'annotations/instances_val2017.json')
    dataset = dict(type='CocoDataset', data_root=a.coco.rstrip('/') + '/', ann_file='annotations/instances_val2017.json',
                   data_prefix=dict(img='val2017/'), test_mode=True, pipeline=cfg.test_pipeline)
    if a.limit:
        dataset['indices'] = a.limit
    cfg.test_dataloader = dict(batch_size=1, num_workers=8, persistent_workers=False, drop_last=False,
                               sampler=dict(type='DefaultSampler', shuffle=False), dataset=dataset)
    cfg.test_evaluator = [
        dict(type='CocoMetric', ann_file=ann, metric='bbox'),
        dict(type='CocoMetric', ann_file=ann, metric='bbox', proposal_nums=(1, 10, 100), prefix='coco_maxdet100'),
    ]
    cfg.custom_hooks = []  # the converted checkpoint already holds the EMA weights
    cfg.load_from = a.checkpoint
    cfg.work_dir = a.work_dir
    metrics = Runner.from_cfg(cfg).test()
    for k in ('coco/bbox_mAP', 'coco/bbox_mAP_50', 'coco/bbox_mAP_75', 'coco_maxdet100/bbox_mAP'):
        print(f'{k}: {metrics.get(k)}')


if __name__ == '__main__':
    main()
