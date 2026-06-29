# configs/retinanet_h1_midogpp.py
#
# RetinaNet with a FROZEN H-Optimus-1 (ViT-g/14) backbone + SimpleFeaturePyramid
# neck, for mitotic-figure detection on MIDOG++.
#
# Aligned to the UNI / UNI2-h / Virchow / Virchow2 / H-optimus-0 RetinaNet
# configs so the RetinaNet row differs ONLY in the detector head. Augmentation
# and RetinaNet head hyperparameters are held constant across the row.
#
# BACKBONE (H-optimus-1 model card, bioptimus/H-optimus-1):
#   - ViT-g/14: patch 14, embed_dim 1536, 1.1B params, init_values 1e-5.
#   - H-OPTIMUS-1-SPECIFIC normalization stats (NOT ImageNet) -- see
#     data_preprocessor below.
#
# *** PATCH 14 -> 1008 input, patches_1008, strides [7,14,28,56,112] ***
#   RetinaNet is anchor-based, so the AnchorGenerator strides MUST match the
#   patch-14 token map (same as the other patch-14 backbones).

custom_imports = dict(
    imports=[
        'src.custom_mmdet.backbones.hoptimus1_vit',
        'src.custom_mmdet.necks.simple_feature_pyramid',
        'src.custom_mmdet.transforms.hed_stain_augment',
    ],
    allow_failed_imports=False
)

_base_ = 'mmdet::retinanet/retinanet_r50_fpn_1x_coco.py'

img_scale = (1008, 1008)        # patch-14 divisible (1008/14 = 72)

metainfo = dict(
    classes=('mitotic figure',),
    palette=[(220, 20, 60)],
)

model = dict(
    # Preprocessor INSIDE model so UNI's ImageNet stats are actually applied.
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[180.342, 147.576, 179.422],  # H-optimus-1 pathology stats (x255)
        std=[54.030, 58.680, 45.267],      # H-optimus-1 std (x255)
        bgr_to_rgb=True,
        pad_size_divisor=1,
    ),

    backbone=dict(
        _delete_=True,
        type='H1Backbone',
        frozen=True,
    ),

    neck=dict(
        _delete_=True,
        type='SimpleFeaturePyramid',
        in_channels=1536,                 # H-optimus-1 embed_dim (ViT-g)
        out_channels=256,
        scale_factors=(2.0, 1.0, 0.5, 0.25, 0.125),
        norm='LN',
    ),

    # RetinaNet head -- canonical (Lin et al. 2017); UNCHANGED from the draft.
    bbox_head=dict(
        type='RetinaHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[7, 14, 28, 56, 112],   # patch-14 -> correct for H-optimus-1
            center_offset=0.5,
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]
        ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0
        ),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0),
    ),

    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1
        ),
        allowed_border=-1,
        pos_weight=-1,
        debug=False
    ),

    # Test cfg aligned to Faster R-CNN row + WSI pipeline (low score floor).
    test_cfg=dict(
        nms_pre=3000,
        min_bbox_size=0,
        score_thr=0.05,                     # was 0.15
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=300                     # was 100
    )
)

# ---------------------------------------------------------------------------
# AUGMENTED training pipeline -- IDENTICAL to the Faster R-CNN / DETR configs.
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

# patch-14 backbone -> patches_1008 (1008/14 = 72 clean). Same data as the
# other patch-14 backbones (H0/H1/Virchow/Virchow2). Confirmed on MUSICA.
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

# Optimizer -- matches Faster R-CNN UNI (AdamW, lr 2e-4, frozen backbone).
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=1e-4, betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
    ),
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
work_dir = './outputs/work_dirs/retinanet_h1_1024_100epochs'

# Single source of truth for schedule + early stopping (no duplicate block).
_max_epochs = 100

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=_max_epochs,
    val_interval=1,
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False,
         begin=0, end=1000),
    dict(type='CosineAnnealingLR', by_epoch=True, T_max=_max_epochs,
         eta_min=1e-7, begin=0, end=_max_epochs),
]

custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='coco/bbox_mAP',
        rule='greater',
        patience=10,           # matches Faster R-CNN row (was 8)
        min_delta=0.001,
    ),
]

# Weights & Biases (online).
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='COMPAYL26',
            name='retinanet_h1_midogpp',
            group='retinanet',
            tags=['H-optimus-1', 'retinanet', 'midogpp', 'frozen', 'augmented'],
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer',
)
