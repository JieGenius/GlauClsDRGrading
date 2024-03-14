# dataset settings
dataset_type = 'MultiTaskDataset'
data_preprocessor = dict(
    num_classes=2,
    # RGB format normalization parameters
    mean=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255],
    std=[0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255],
    # convert image from BGR to RGB
    to_rgb=True,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        scale=448,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackMultiTaskInputs', multi_task_fields=['gt_label']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeEdge',
        scale=448,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=448),
    dict(type='PackMultiTaskInputs', multi_task_fields=['gt_label']),
]

train_dataloader = dict(
    batch_size=16,
    num_workers=16,
    dataset=dict(
        type=dataset_type,
        data_root='data/refuge_aptos_amd_pm',
        ann_file='splits/train.json',
        data_prefix='images',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),

)

val_dataloader = dict(
    batch_size=8,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='data/refuge_aptos_amd_pm',
        ann_file='splits/val.json',
        data_prefix='images',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

test_dataloader = dict(
    batch_size=8,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='data/refuge_aptos_amd_pm',
        ann_file='splits/test.json',
        data_prefix='images',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

val_evaluator = dict(type='MultiTasksMetric', task_metrics={
    task: [
        dict(type='Accuracy', topk=(1,)),
        dict(type="SingleLabelMetric"),
        dict(type="SingleLabelMetric", average=None),
        dict(type="ConfusionMatrix")
    ]
    for task in ["Glau", "DR", "AMD", "PM"]
})

# If you want standard test, please manually configure the test dataset
test_evaluator = val_evaluator
