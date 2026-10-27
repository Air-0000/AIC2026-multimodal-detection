# Co-DINO Swin-L, RGB only, the 12 AIC classes: trains on the 1800-image split and validates on the same 200 images
# and labels as clean baseline v1 (val mAP 55.94), to see how far a much stronger RGB detector gets. A diagnostic: an
# RGB-only model does not meet the three-modality rule and cannot be submitted as is.
# Starts from the official Objects365->COCO checkpoint (COCO val 64.1). Everything not set here is mmdetection
# v3.3.0's projects/CO-DETR/configs/codino/co_dino_5scale_swin_l_16xb1_16e_o365tococo.py (multi-scale resize to at
# most 1536 x 2048 or a 384-600 crop, test resize to fit 2048 x 1280, AdamW with backbone lr x0.1, grad clip 0.1).
# Differences from it:
#   1. 12 classes in the DINO query head and both auxiliary heads (Faster R-CNN, ATSS); their class layers start from
#      scratch (mmengine skips the 80-class tensors with a size-mismatch warning).
#   2. Data: dataset/train/visible with work_dir/deimv2_dataset/{train,valid} labels, paths relative to the repo root
#      (run_codino_aic.sh runs from there); class names in the label files' category order.
#   3. No backbone init_cfg: it would download the ImageNet-22k Swin-L file only for load_from to overwrite it.
#   4. At most 100 detections per image (the competition limit) and COCO mAP read at maxDets 100 (mmdet's
#      CocoMetric otherwise reports bbox_mAP at 1000).
#   5. Batch 1 x 4 accumulated iterations = 4 images per step; lr 2.5e-5 (upstream 1e-4 for 16, scaled linearly),
#      linear warmup over the first 500 iterations.
#   6. 12 epochs, lr x0.1 after epoch 10; validate every epoch and keep the best checkpoint by coco/bbox_mAP.
#   7. TF32 matmul and cuDNN through aic_codino_runtime; weights and activations stay FP32 (no autocast).
_base_ = ['../../../vendor/mmdetection/projects/CO-DETR/configs/codino/'
          'co_dino_5scale_swin_l_16xb1_16e_o365tococo.py']

custom_imports = dict(imports=['projects.CO-DETR.codetr', 'aic_codino_runtime'], allow_failed_imports=False)

load_from = 'weights/codino/co_dino_5scale_swin_large_16e_o365tococo-614254c9.pth'

classes = ('person', 'boat', 'animal', 'seat', 'sign', 'bicycle',
           'car', 'ball', 'light', 'garbage can', 'uav', 'tricycle')
num_classes = len(classes)
metainfo = dict(classes=classes)
data_root = ''
img_prefix = 'dataset/train/visible/'
val_ann = 'work_dir/deimv2_dataset/valid/annotations/instances_val.json'

# Same loss weighting as upstream: the auxiliary losses are scaled by num_dec_layer * loss_lambda.
num_dec_layer = 6
loss_lambda = 2.0

# roi_head, bbox_head and test_cfg are lists, which mmengine replaces instead of merging, so they are written out in
# full; only num_classes and the first max_per_img differ from upstream.
model = dict(
    backbone=dict(init_cfg=None),
    query_head=dict(num_classes=num_classes),
    roi_head=[
        dict(
            type='CoStandardRoIHead',
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32, 64],
                finest_scale=56),
            bbox_head=dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0 * num_dec_layer * loss_lambda),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0 * num_dec_layer * loss_lambda)))
    ],
    bbox_head=[
        dict(
            type='CoATSSHead',
            num_classes=num_classes,
            in_channels=256,
            stacked_convs=1,
            feat_channels=256,
            anchor_generator=dict(
                type='AnchorGenerator',
                ratios=[1.0],
                octave_base_scale=8,
                scales_per_octave=1,
                strides=[4, 8, 16, 32, 64, 128]),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[.0, .0, .0, .0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0 * num_dec_layer * loss_lambda),
            loss_bbox=dict(type='GIoULoss', loss_weight=2.0 * num_dec_layer * loss_lambda),
            loss_centerness=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0 * num_dec_layer * loss_lambda)),
    ],
    test_cfg=[
        dict(max_per_img=100, nms=dict(type='soft_nms', iou_threshold=0.8)),
        dict(
            rpn=dict(nms_pre=1000, max_per_img=1000, nms=dict(type='nms', iou_threshold=0.7), min_bbox_size=0),
            rcnn=dict(score_thr=0.0, nms=dict(type='nms', iou_threshold=0.5), max_per_img=100)),
        dict(nms_pre=1000, min_bbox_size=0, score_thr=0.0, nms=dict(type='nms', iou_threshold=0.6), max_per_img=100),
    ])

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='work_dir/deimv2_dataset/train/annotations/instances_train.json',
        data_prefix=dict(img=img_prefix)))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(data_root=data_root, metainfo=metainfo, ann_file=val_ann, data_prefix=dict(img=img_prefix)))
test_dataloader = val_dataloader

val_evaluator = dict(type='CocoMetric', ann_file=val_ann, metric='bbox', proposal_nums=(1, 10, 100), classwise=True)
test_evaluator = val_evaluator

optim_wrapper = dict(optimizer=dict(lr=2.5e-5), accumulative_counts=4)

max_epochs = 12
train_cfg = dict(max_epochs=max_epochs, val_interval=1)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.01, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end=max_epochs, by_epoch=True, milestones=[10], gamma=0.1),
]

default_hooks = dict(
    logger=dict(interval=50),
    checkpoint=dict(by_epoch=True, interval=1, max_keep_ckpts=2, save_best='coco/bbox_mAP', rule='greater'))

randomness = dict(seed=0)
