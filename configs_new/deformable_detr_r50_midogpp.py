# configs/deformable_detr_r50_midogpp.py
#
# Deformable DETR with a standard ResNet-50 backbone on MIDOG++.
#
# CONVENTIONAL CNN BASELINE for the COMPAYL26 Deformable DETR row: an
# ImageNet-pretrained ResNet-50, FULLY TRAINABLE, in mmdet's NATIVE Deformable
# DETR setup. Answers "does a pathology foundation model beat a plain
# supervised ResNet-50 Deformable DETR detector?".
#
# Unlike the foundation-model DETR cells (frozen ViT -> single token map ->
# SimpleFeaturePyramid synthesises 4 levels), this baseline uses the detector
# exactly as designed: ResNet's C3,C4,C5 feed a ChannelMapper that produces the
# multi-scale features for deformable attention (num_feature_levels=4, with the
# 4th level an extra conv on C5). This is the standard deformable-detr_r50
# recipe -- NOT SimpleFeaturePyramid, which is a ViT adapter.
#
# Deliberate differences from the FM DETR cells (what a conventional ResNet
# Deformable DETR actually is):
#   - neck       : ChannelMapper (native), NOT SimpleFeaturePyramid
#   - backbone   : trainable (frozen_stages=1 + a lower backbone LR), NOT frozen
#   - input/data : the same 1008 patches (patches_1008), keep_ratio=False
#   - normalization: ImageNet stats (ResNet pretrain)
#
# OPTIMIZER NOTE: Deformable DETR is canonically trained with AdamW (the
# transformer needs it; SGD does not converge well for DETR-family models).
# So this baseline uses AdamW with a 0.1x LR multiplier on the backbone --
# the standard deformable-detr recipe -- even though the ResNet Faster R-CNN /
# RetinaNet baselines use SGD. The optimizer follows the detector head.

_base_ = 'mmdet::deformable_detr/deformable-detr_r50_16xb2-50e_coco.py'

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
        pad_size_divisor=1,
    ),

    # Standard ResNet-50, ImageNet-pretrained, trainable.
    # (Base config already specifies ResNet-50 with out_indices=(1,2,3) ->
    #  C3,C4,C5 and num_feature_levels=4; we only adjust frozen_stages and
    #  ensure the pretrained init + trainable norm.)
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),            # C3,C4,C5 for deformable attention
        frozen_stages=1,                  # standard: freeze stem+stage1 only
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet50'),
    ),

    # Native ChannelMapper neck (inherited from base; stated explicitly).
    # Maps C3,C4,C5 (+1 extra level) to 256-d multi-scale features.
    neck=dict(
        type='ChannelMapper',
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4,
    ),

    bbox_head=dict(num_classes=1),
)

# ---------------------------------------------------------------------------
# AUGMENTED training pipeline -- same full set as the other cells.
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
    batch_size=16,                        # train batch (was 8)
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

# Optimizer: canonical Deformable DETR AdamW with 0.1x backbone LR.
# (For a TRAINABLE ResNet the backbone LR multiplier matters -- the standard
#  DETR recipe trains the backbone slower than the transformer.)
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=1e-4, betas=(0.9, 0.999)),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1),
        }
    ),
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
work_dir = './outputs/work_dirs/deformable_detr_r50_1008_100epochs'

custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='coco/bbox_mAP',
        rule='greater',
        patience=20,           # DETR patience (matches the FM DETR row)
        min_delta=0.0005,
    ),
]

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='COMPAYL26',
            name='deformable_detr_r50_midogpp',
            group='deformable_detr_baseline',
            tags=['resnet50', 'deformable_detr', 'midogpp', 'trainable',
                  'imagenet', 'augmented'],
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer',
)
