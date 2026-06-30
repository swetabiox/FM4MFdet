custom_imports = dict(
    imports=[
        'src.custom_mmdet.backbones.uni2h_vit',
        'src.custom_mmdet.necks.simple_feature_pyramid',
        'src.custom_mmdet.transforms.hed_stain_augment',
    ],
    allow_failed_imports=False
)

_base_ = 'mmdet::deformable_detr/deformable-detr_r50_16xb2-50e_coco.py'

img_scale = (1008, 1008)      

metainfo = dict(
    classes=('mitotic figure',),
    palette=[(220, 20, 60)],
)

model = dict(
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],   
        std=[58.395, 57.12, 57.375],     
        bgr_to_rgb=True,
        pad_size_divisor=1,
    ),

    backbone=dict(
        _delete_=True,
        type='UNI2hBackbone',
        frozen=True,
    ),


    neck=dict(
        _delete_=True,
        type='SimpleFeaturePyramid',
        in_channels=1536,                 # UNI2-h embed_dim (ViT-H)
        out_channels=256,
        scale_factors=(2.0, 1.0, 0.5, 0.25),   # 4 levels
        norm='LN',
    ),

    encoder=dict(
        layer_cfg=dict(
            self_attn_cfg=dict(num_levels=4),
        ),
    ),
    decoder=dict(
        layer_cfg=dict(
            cross_attn_cfg=dict(num_levels=4),
        ),
    ),

    num_feature_levels=4,

    bbox_head=dict(
        num_classes=1,
    ),
)


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),

    dict(type='Resize', scale=img_scale, keep_ratio=False, backend='pillow'),
    dict(type='RandomFlip', prob=0.5, direction=['horizontal', 'vertical']),
    dict(
        type='RandomAffine',
        max_rotate_degree=15.0,
        max_translate_ratio=0.05,
        scaling_ratio_range=(0.9, 1.1),
        max_shear_degree=0.0,
        border=(0, 0),
        border_val=(114, 114, 114),
    ),

    dict(
        type='HEDStainAugment',
        sigma=0.05,
        bias=0.02,
        prob=0.5,
    ),

    dict(
        type='PhotoMetricDistortion',
        brightness_delta=16,
        contrast_range=(0.9, 1.1),
        saturation_range=(0.9, 1.1),
        hue_delta=10,
    ),

    dict(type='PackDetInputs'),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=img_scale, keep_ratio=False),
    dict(type='PackDetInputs'),
]
test_pipeline = val_pipeline

data_root = './data/'

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='coco_annotations/patches_1008/midogpp_train.json',
        data_prefix=dict(img='Datensatz/patches_1008/'),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=train_pipeline,
    )
)

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='coco_annotations/patches_1008/midogpp_val.json',
        data_prefix=dict(img='Datensatz/patches_1008/'),
        metainfo=metainfo,
        test_mode=True,
        pipeline=val_pipeline,
    )
)

test_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='coco_annotations/patches_1008/midogpp_test.json',
        data_prefix=dict(img='Datensatz/patches_1008/'),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline,
    )
)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=1e-4, betas=(0.9, 0.999)),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    accumulative_counts=2, 
)

_max_epochs = 100

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False,
         begin=0, end=2000),
    dict(type='MultiStepLR', by_epoch=True, milestones=[80],
         gamma=0.1, begin=0, end=_max_epochs),
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=_max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

val_evaluator = dict(
    type='CocoMetric',
    ann_file='data/coco_annotations/patches_1008/midogpp_val.json',
    metric='bbox',
    format_only=False,
    backend_args=None
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file='data/coco_annotations/patches_1008/midogpp_test.json',
    metric='bbox',
    format_only=False,
    backend_args=None
)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3,
                    save_best='coco/bbox_mAP', rule='greater'),
    logger=dict(type='LoggerHook', interval=10, log_metric_by_epoch=True),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

randomness = dict(seed=42, deterministic=False, diff_rank_seed=True)

resume = False

work_dir = './outputs/work_dirs/deformable_detr_uni2h_1008_100epochs'

custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='coco/bbox_mAP',
        rule='greater',
        patience=20,
        min_delta=0.0005,
    ),
]

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='COMPAYL26',
            name='deformable_detr_uni2h_midogpp',
            group='deformable_detr',
            tags=['UNI2-h', 'deformable_detr', 'midogpp', 'frozen', 'augmented'],
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer',
)
