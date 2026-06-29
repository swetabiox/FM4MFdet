# configs/retinanet_r50_midogpp.py
#
# RetinaNet with a standard ResNet-50 + FPN backbone on MIDOG++.
#
# CONVENTIONAL CNN BASELINE for the COMPAYL26 RetinaNet row: an
# ImageNet-pretrained ResNet-50, FULLY TRAINABLE, with the native Feature
# Pyramid Network. Answers "does a pathology foundation model beat a plain
# supervised ResNet-50 RetinaNet detector?" -- so it is the standard mmdet
# recipe, NOT forced through the ViT-style SimpleFeaturePyramid (which is built
# for a single ViT token map and would handicap a hierarchical CNN that already
# produces a real C3-C5 pyramid).
#
# Deliberate differences from the foundation-model RetinaNet cells (they are
# what a conventional ResNet detector actually is):
#   - neck       : FPN (native, RetinaNet's standard start_level=1), NOT SFP
#   - backbone   : trainable (frozen_stages=1, the standard detection setting)
#   - optimizer  : SGD (canonical RetinaNet), NOT AdamW
#   - strides    : standard ResNet-FPN [8,16,32,64,128]
#   - normalization: ImageNet stats (ResNet was pretrained on ImageNet)
#
# Held CONSTANT with the rest of the matrix where it makes sense:
#   - data        : the same 1008 patches (patches_1008)
#   - augmentation: the same full pipeline incl. HED stain
#   - RetinaNet head internals (focal loss, anchors), early stopping, wandb

_base_ = 'mmdet::retinanet/retinanet_r50_fpn_1x_coco.py'

custom_imports = dict(
    imports=[
        'src.custom_mmdet.transforms.hed_stain_augment',
    ],
    allow_failed_imports=False
)

img_scale = (1008, 1008)

metainfo = dict(
    classes=('mitotic figure',),
    palette=[(220, 20, 60)],
)

model = dict(
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],   # ImageNet RGB mean (ResNet pretrain)
        std=[58.395, 57.12, 57.375],      # ImageNet RGB std
        bgr_to_rgb=True,
        pad_size_divisor=32,              # FPN needs /32-divisible padding
    ),

    # Standard ResNet-50, ImageNet-pretrained, trainable.
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),         # C2,C3,C4,C5
        frozen_stages=1,                  # standard: freeze stem+stage1 only
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,                   # standard detection BN setting
        style='pytorch',
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet50'),
    ),

    # Native FPN. RetinaNet uses start_level=1 with an extra P6/P7 on the
    # top input feature (add_extra_convs='on_input'), giving P3..P7.
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5,
    ),

    bbox_head=dict(
        num_classes=1,
        # RetinaNet head internals (anchors, focal loss) kept at the canonical
        # values inherited from the base config; strides come from the FPN
        # (P3..P7 -> [8,16,32,64,128]), which is correct for ResNet-FPN.
    ),

    test_cfg=dict(
        nms_pre=3000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=300
    )
)

# ---------------------------------------------------------------------------
# AUGMENTED training pipeline -- same full set as the foundation-model cells.
# ---------------------------------------------------------------------------
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
    num_workers=8,
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

# Optimizer: SGD -- canonical RetinaNet for a fully-trainable ResNet.
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=1e-4),
    clip_grad=dict(max_norm=35, norm_type=2),
)

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
work_dir = './outputs/work_dirs/retinanet_r50_1008_100epochs'

_max_epochs = 100

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=_max_epochs,
    val_interval=1,
)

# SGD schedule: linear warmup then step decay at 75%/90% of the budget
# (canonical RetinaNet step-decay style, scaled to _max_epochs).
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False,
         begin=0, end=500),
    dict(type='MultiStepLR', by_epoch=True,
         milestones=[int(_max_epochs * 0.75), int(_max_epochs * 0.9)],
         gamma=0.1, begin=0, end=_max_epochs),
]

custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='coco/bbox_mAP',
        rule='greater',
        patience=10,
        min_delta=0.001,
    ),
]

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='COMPAYL26',
            name='retinanet_r50_midogpp',
            group='retinanet_baseline',
            tags=['resnet50', 'retinanet', 'midogpp', 'trainable',
                  'imagenet', 'augmented'],
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer',
)
