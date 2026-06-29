# configs/deformable_detr_h0_midogpp.py
#
# Deformable DETR with a FROZEN H-Optimus-0 (ViT-g/14) backbone +
# SimpleFeaturePyramid neck, for mitotic-figure detection on MIDOG++.
#
# BACKBONE (H-optimus-0 model card, bioptimus/H-optimus-0):
#   - ViT-g/14: patch 14, embed_dim 1536, 1.1B params.
#   - H-OPTIMUS-SPECIFIC normalization stats (NOT ImageNet).
#
# PATCH 14 -> 1008 input -> 72x72 token map, same as the other DETR configs.
#
# Deformable DETR head / optimizer / schedule / augmentation are IDENTICAL to
# the H1 / UNI2-h / Virchow / Virchow2 DETR configs (kept constant for a fair
# backbone comparison); only the backbone and normalization change.
#
# *** 4-GPU DDP VERSION (MUSICA full node) ***
# Effective batch is held at 32 to match the single-GPU DETR cells:
#   single-GPU: batch_size=16 x accumulative_counts=2          = 32
#   4-GPU DDP : batch_size=8 (per GPU) x 4 GPUs x accum 1       = 32
# Launch with torchrun --nproc_per_node=4 (see the sbatch). The LR, schedule
# and all else are UNCHANGED, so results stay comparable to the other cells.

custom_imports = dict(
    imports=[
        'src.custom_mmdet.backbones.hoptimus0_vit',
        'src.custom_mmdet.necks.simple_feature_pyramid',
        'src.custom_mmdet.transforms.hed_stain_augment',
    ],
    allow_failed_imports=False
)

_base_ = 'mmdet::deformable_detr/deformable-detr_r50_16xb2-50e_coco.py'

img_scale = (1008, 1008)        # patch-14 divisible (1008/14 = 72)

metainfo = dict(
    classes=('mitotic figure',),
    palette=[(220, 20, 60)],
)

model = dict(
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[180.285, 147.645, 179.520],  # H-optimus-0 pathology stats (x255)
        std=[54.060, 58.650, 45.390],      # H-optimus-0 std (x255)
        bgr_to_rgb=True,
        pad_size_divisor=1,
    ),

    backbone=dict(
        _delete_=True,
        type='H0Backbone',
        frozen=True,
    ),

    # H-optimus-0 emits one (B, 1536, 72, 72) map. SimpleFeaturePyramid expands it
    # to 4 levels at 256 channels for Deformable DETR's multi-scale deformable
    # attention. No ChannelMapper needed.
    neck=dict(
        _delete_=True,
        type='SimpleFeaturePyramid',
        in_channels=1536,                 # H-optimus-0 embed_dim (ViT-g)
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

train_dataloader = dict(
    batch_size=8,    # per-GPU; 8 x 4 GPUs (DDP) = 32 effective (was 16 x accum2 on 1 GPU)
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

# ---------------------------------------------------------------------------
# Optimisation: canonical Deformable DETR recipe -- IDENTICAL to H0 / H1.
# ---------------------------------------------------------------------------
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=1e-4, betas=(0.9, 0.999)),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    accumulative_counts=1,   # DDP: 8 per-GPU x 4 GPUs = 32 effective (no accum needed)
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

work_dir = './outputs/work_dirs/deformable_detr_h0_1008_100epochs'

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
            name='deformable_detr_h0_midogpp',
            group='deformable_detr',
            tags=['H-optimus-0', 'deformable_detr', 'midogpp', 'frozen', 'augmented'],
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer',
)

launcher = 'pytorch'
