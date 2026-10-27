# co_dino_vit_l_aic_rgb_smoke.py shrunk to fit a 16 GB card (the workstation's RTX 5060 Ti): train resize short side
# 480-640 with long side <= 1066 and no crop branch, test resize to fit 1066 x 640. Only for checking that training
# and validation run on Blackwell; its timing and memory say nothing about the full-scale run.
_base_ = ['./co_dino_vit_l_aic_rgb_smoke.py']

train_dataloader = dict(
    num_workers=2,
    dataset=dict(pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='RandomFlip', prob=0.5),
        dict(type='RandomChoiceResize', scales=[(480, 1066), (544, 1066), (608, 1066), (640, 1066)], keep_ratio=True),
        dict(type='PackDetInputs'),
    ]))
val_dataloader = dict(dataset=dict(pipeline=[
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1066, 640), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
]))
test_dataloader = val_dataloader
