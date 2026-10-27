# Smoke run of co_dino_swin_l_aic_rgb_1800.py: 40 training iterations (10 optimizer steps), then validation on the
# first 20 val images, no checkpoint written. Checks the whole train/val path and gives the per-iteration time and
# peak memory that size the real run. The mAP it prints is meaningless (12-class heads after 10 steps; CocoMetric
# still scores against all 200 images).
_base_ = ['./co_dino_swin_l_aic_rgb_1800.py']

train_cfg = dict(_delete_=True, type='IterBasedTrainLoop', max_iters=40, val_interval=40)
param_scheduler = [dict(type='LinearLR', start_factor=0.01, by_epoch=False, begin=0, end=40)]
val_dataloader = dict(dataset=dict(indices=20))
test_dataloader = val_dataloader
default_hooks = dict(
    logger=dict(interval=5),
    checkpoint=dict(_delete_=True, type='CheckpointHook', by_epoch=False, interval=-1, save_last=False))
# The logged time is a moving average over window_size iterations (50 upstream); 5 keeps warm-up out of the late ones.
log_processor = dict(by_epoch=False, window_size=5)
