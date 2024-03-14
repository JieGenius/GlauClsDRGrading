_base_ = [
    '../_base_/models/convnext/convnext-tiny.py',
    '../_base_/datasets/refuge_aptos_amd_pm.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py',
]
checkpoint = "https://download.openmmlab.com/mmclassification/v0/convnext/convnext-tiny_in21k-pre_3rdparty_in1k-384px_20221219-c1182362.pth"

model = dict(
    backbone=dict(init_cfg=dict(type="Pretrained", checkpoint=checkpoint, prefix='backbone.')),
    head=dict(
        type="MultiTaskHead",
        task_heads={
            "Glau": dict(type='LinearClsHead', num_classes=2, in_channels=768,
                         loss=dict(type='LabelSmoothLoss', label_smooth_val=0.1, mode='original',
                                   class_weight=[1.0, 3.0])),
            "DR": dict(type='LinearClsHead', num_classes=5, in_channels=768,
                       loss=dict(type='LabelSmoothLoss', label_smooth_val=0.1, mode='original',
                                 class_weight=[1.0, 1.2, 1.0, 2.5, 3.5]), ),
            "AMD": dict(type='LinearClsHead', num_classes=2, in_channels=768,
                                     loss=dict(type='LabelSmoothLoss', label_smooth_val=0.1, mode='original',
                                               class_weight=[1.0, 3.0])),
            "PM": dict(type='LinearClsHead', num_classes=2, in_channels=768,
                                     loss=dict(type='LabelSmoothLoss', label_smooth_val=0.1, mode='original',
                                               class_weight=[1.0, 3.0])),
        },
        _delete_=True,
    ),
)
# dataset setting
train_dataloader = dict(batch_size=32)

# schedule setting

# runtime setting
custom_hooks = [dict(type='EMAHook', momentum=1e-4, priority='ABOVE_NORMAL')]

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=32)

param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=True,
        end=20,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(type='OneCycleLR', eta_max=0.001, by_epoch=True, begin=20)
]

optim_wrapper = dict(
    optimizer=dict(lr=4e-3),
    clip_grad=None,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'head': dict(lr_mult=1.0),
        }))

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend', init_kwargs=dict(project='GlauClsDRGrading')),
]
visualizer = dict(type='UniversalVisualizer', vis_backends=vis_backends)

train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)