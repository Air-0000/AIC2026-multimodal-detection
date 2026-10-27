# Co-DINO ViT-L, RGB only, the 12 AIC classes: trains on the 1800-image split and validates on the same 200 images
# and labels as clean baseline v1, to see how far the strongest released Co-DETR gets. A diagnostic: an RGB-only
# model does not meet the three-modality rule and cannot be submitted as is.
# Starts from Sense-X's Objects365->COCO ViT-L checkpoint (COCO val 65.4 single scale, 65.9 with TTA), converted by
# convert_codino_vit_ckpt.py. Model and recipe follow upstream projects/configs/co_dino_vit/
# co_dino_5scale_vit_large_coco.py (mmdet 2.25.3) on top of mmdetection v3.3.0's Co-DETR project
# (co_dino_5scale_r50_8xb2_1x_coco.py: DETR-style multi-scale training, Co-DINO head, ATSS + Faster R-CNN aux heads):
#   EVA-02 ViT-L/16 (window 24, global attention in blocks 3, 7, ..., 23, drop path 0.4, activation checkpointing)
#   + simple feature pyramid (strides 4-64) from aic_codino_vit.py; 1500 queries, 300 denoising queries, box noise
#   0.4; RoI aux head with 4 shared convs + GN, class-agnostic boxes; train resize short side 480-1536 (step 32),
#   long side <= 2400, or the 384-600 crop branch; test resize to fit 2048 x 1280; inputs padded to 32;
#   AdamW wd 0.01, ViT layer-wise lr decay 0.8, grad clip 0.1, 500-iteration linear warmup, lr x0.1 after epoch 7
#   of 12, exponential-momentum EMA (momentum 1e-4) used for validation and saved checkpoints.
# Differences from upstream:
#   1. 12 classes in the query head and both aux heads (class layers start from scratch).
#   2. Data: dataset/train/visible with work_dir/deimv2_dataset/{train,valid} labels, paths relative to the repo root.
#   3. At most 100 detections per image (competition limit; upstream keeps 1000 for COCO eval) and COCO mAP read
#      at maxDets 100.
#   4. 4 images per optimizer step: 1 per GPU x accumulative_counts (4 on one GPU; run_codino_aic.sh sets it to
#      4 / NGPUS). lr 1.25e-5: upstream's 5e-5 scaled linearly from 16 images per step, the GPU count upstream used
#      is not stated (16 as in its Swin-L runs is an assumption). With accumulation the EMA still updates every
#      iteration, i.e. 4x per optimizer step.
#   5. TF32 matmul / cuDNN through aic_codino_runtime; weights and activations FP32 (no autocast).
_base_ = ['../../../vendor/mmdetection/projects/CO-DETR/configs/codino/co_dino_5scale_r50_8xb2_1x_coco.py']

custom_imports = dict(
    imports=['projects.CO-DETR.codetr', 'aic_codino_runtime', 'aic_codino_vit'], allow_failed_imports=False)

load_from = 'weights/codino/co_dino_vit_l_o365tococo_mmdet3.pth'

classes = ('person', 'boat', 'animal', 'seat', 'sign', 'bicycle',
           'car', 'ball', 'light', 'garbage can', 'uav', 'tricycle')
num_classes = len(classes)
metainfo = dict(classes=classes)
data_root = ''
img_prefix = 'dataset/train/visible/'
val_ann = 'work_dir/deimv2_dataset/valid/annotations/instances_val.json'

num_dec_layer = 6
loss_lambda = 2.0
window_block_indexes = [i for i in range(24) if i % 4 != 3]

# roi_head, bbox_head and test_cfg are lists, which mmengine replaces instead of merging, so they are written out.
model = dict(
    data_preprocessor=dict(pad_size_divisor=32),
    backbone=dict(
        _delete_=True,
        type='CoDETRViT',
        img_size=1536,
        pretrain_img_size=512,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4 * 2 / 3,
        drop_path_rate=0.4,
        window_size=24,
        window_block_indexes=window_block_indexes,
        qkv_bias=True,
        use_act_checkpoint=True,
        init_cfg=None),
    neck=dict(
        _delete_=True, type='CoDETRSFP', in_channels=[1024], out_channels=256, num_outs=5, use_p2=True,
        use_act_checkpoint=False),
    query_head=dict(
        num_query=1500,
        num_classes=num_classes,
        sync_cls_avg_factor=True,
        dn_cfg=dict(box_noise_scale=0.4, group_cfg=dict(num_dn_queries=300)),
        transformer=dict(encoder=dict(with_cp=6))),
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
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='GN', num_groups=32),
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
                type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25,
                loss_weight=1.0 * num_dec_layer * loss_lambda),
            loss_bbox=dict(type='GIoULoss', loss_weight=2.0 * num_dec_layer * loss_lambda),
            loss_centerness=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0 * num_dec_layer * loss_lambda)),
    ],
    test_cfg=[
        dict(max_per_img=100, nms=dict(type='soft_nms', iou_threshold=0.8)),
        dict(
            rpn=dict(nms_pre=8000, max_per_img=2000, nms=dict(type='nms', iou_threshold=0.9), min_bbox_size=0),
            rcnn=dict(score_thr=0.0, nms=dict(type='soft_nms', iou_threshold=0.5), max_per_img=100)),
        dict(nms_pre=1000, min_bbox_size=0, score_thr=0.0, nms=dict(type='soft_nms', iou_threshold=0.6),
             max_per_img=100),
    ])

train_scales = [(short, 2400) for short in range(480, 1537, 32)]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [dict(type='RandomChoiceResize', scales=train_scales, keep_ratio=True)],
            [
                dict(type='RandomChoiceResize', scales=[(400, 4200), (500, 4200), (600, 4200)], keep_ratio=True),
                dict(type='RandomCrop', crop_type='absolute_range', crop_size=(384, 600), allow_negative_crop=True),
                dict(type='RandomChoiceResize', scales=train_scales, keep_ratio=True),
            ],
        ]),
    dict(type='PackDetInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 1280), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
]

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='work_dir/deimv2_dataset/train/annotations/instances_train.json',
        data_prefix=dict(img=img_prefix),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(
        data_root=data_root, metainfo=metainfo, ann_file=val_ann, data_prefix=dict(img=img_prefix),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='CocoMetric', ann_file=val_ann, metric='bbox', proposal_nums=(1, 10, 100), classwise=True)
test_evaluator = val_evaluator

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1.25e-5, weight_decay=0.01),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    constructor='CoDETRViTLayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=24, layer_decay_rate=0.8),
    accumulative_counts=4)

max_epochs = 12
train_cfg = dict(max_epochs=max_epochs, val_interval=1)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.01, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end=max_epochs, by_epoch=True, milestones=[7], gamma=0.1),
]
custom_hooks = [dict(type='EMAHook', ema_type='ExpMomentumEMA', momentum=0.0001, update_buffers=True, priority=49)]

default_hooks = dict(
    logger=dict(interval=50),
    checkpoint=dict(by_epoch=True, interval=1, max_keep_ckpts=2, save_best='coco/bbox_mAP', rule='greater'))

randomness = dict(seed=0)
