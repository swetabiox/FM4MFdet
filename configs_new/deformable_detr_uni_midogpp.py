# configs/deformable_detr_uni_midogpp.py
#
# Deformable DETR with a FROZEN UNI (ViT-L/16) backbone + SimpleFeaturePyramid
# neck, for mitotic-figure detection on MIDOG++.
#
# BACKBONE (UNI model card, MahmoodLab/UNI):
#   - ViT-Large/16: patch 16, embed_dim 1024, init_values 1e-5.
#   - ImageNet normalization (DINOv2 recipe).
#
# *** PATCH 16 -> use 1024 input (1024/16 = 64 clean), NOT 1008 ***
#   UNI is patch-16, so we use the patches_1024 data and a 1024 input. The
#   token map is 64x64 (stride 16). For Deformable DETR this only affects the
#   SFP input resolution; the deformable attention uses normalized reference
#   points, so there are no anchor/stride alignments to recompute (unlike the
#   Faster R-CNN configs). SFP still produces 4 levels at 256 channels.
#
# Deformable DETR head / optimizer / schedule / augmentation are IDENTICAL to
# the H0 / H1 DETR configs (kept constant for a fair backbone comparison);
# only the backbone, embed_dim (1024), normalization, and input/data change.

custom_imports = dict(
    imports=[
        'src.custom_mmdet.backbones.uni_vit',
        'src.custom_mmdet.necks.simple_feature_pyramid',
        'src.custom_mmdet.transforms.hed_stain_augment',
    ],
    allow_failed_imports=False
)

_base_ = 'mmdet::deformable_detr/deformable-detr_r50_16xb2-50e_coco.py'

img_scale = (1024, 1024)        # patch-16 divisible (1024/16 = 64)

metainfo = dict(
    classes=('mitotic figure',),
    palette=[(220, 20, 60)],
)

model = dict(
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],   # ImageNet RGB mean (UNI / DINOv2)
        std=[58.395, 57.12, 57.375],      # ImageNet RGB std
        bgr_to_rgb=True,
        pad_size_divisor=1,
    ),

    backbone=dict(
        _delete_=True,
        type='UNIBackbone',
        img_size=1024,
        frozen=True,
    ),

    # UNI emits one (B, 1024, 64, 64) map. SimpleFeaturePyramid expands it to
    # 4 levels at 256 channels for Deformable DETR's multi-scale deformable
    # attention. No ChannelMapper needed.
    neck=dict(
        _delete_=True,
        type='SimpleFeaturePyramid',
        in_channels=1024,                 # UNI embed_dim (ViT-L)
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

# ---------------------------------------------------------------------------
# AUGMENTED training pipeline -- IDENTICAL to H0 / H1 DETR configs.
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

# patch-16 backbone -> 1024 patch set (matches the UNI Faster R-CNN config).
train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='coco_annotations/patches_1024/midogpp_train.json',
        data_prefix=dict(img='Datensatz/patches_1024/'),
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
        ann_file='coco_annotations/patches_1024/midogpp_val.json',
        data_prefix=dict(img='Datensatz/patches_1024/'),
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
        ann_file='coco_annotations/patches_1024/midogpp_test.json',
        data_prefix=dict(img='Datensatz/patches_1024/'),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline,
    )
)

# ---------------------------------------------------------------------------
# Optimisation: canonical Deformable DETR recipe -- IDENTICAL to H0 / H1.
#   base LR 2e-4, effective batch 32 (16 x accum 2), AdamW, wd 1e-4, clip 0.1.
# ---------------------------------------------------------------------------
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=1e-4, betas=(0.9, 0.999)),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    accumulative_counts=2,   # 16 (physical) x 2 = 32 effective batch
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
    ann_file='data/coco_annotations/patches_1024/midogpp_val.json',
    metric='bbox',
    format_only=False,
    backend_args=None
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file='data/coco_annotations/patches_1024/midogpp_test.json',
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

work_dir = './outputs/work_dirs/deformable_detr_uni_1024_100epochs'

# --- early stopping (identical to H0 / H1) ---
custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='coco/bbox_mAP',
        rule='greater',
        patience=20,
        min_delta=0.0005,
    ),
]

# --- Weights & Biases (online) ---
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='COMPAYL26',
            name='deformable_detr_uni_midogpp',
            group='deformable_detr',
            tags=['UNI', 'deformable_detr', 'midogpp', 'frozen', 'augmented'],
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer',
)
