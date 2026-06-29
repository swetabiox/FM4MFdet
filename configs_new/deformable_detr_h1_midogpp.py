# configs/deformable_detr_h1_midogpp.py

custom_imports = dict(
    imports=[
        'src.custom_mmdet.backbones.hoptimus1_vit',
        'src.custom_mmdet.necks.simple_feature_pyramid',
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
    # H-optimus-1 normalization stats (model card values x255).
    # mean (0.707223, 0.578729, 0.703617) -> [180.342, 147.576, 179.422]
    # std  (0.211883, 0.230117, 0.177517) -> [54.030, 58.680, 45.267]
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[180.342, 147.576, 179.422],
        std=[54.030, 58.680, 45.267],
        bgr_to_rgb=True,
        pad_size_divisor=1,
    ),

    backbone=dict(
        _delete_=True,
        type='H1Backbone',
        frozen=True,
    ),

    # H1 emits one (B, 1536, 72, 72) map. SimpleFeaturePyramid expands it to
    # 4 levels at 256 channels, which is what Deformable DETR's multi-scale
    # deformable attention expects. No ChannelMapper needed - SFP already
    # outputs the right channel count and number of levels.
    neck=dict(
        _delete_=True,
        type='SimpleFeaturePyramid',
        in_channels=1536,
        out_channels=256,
        scale_factors=(2.0, 1.0, 0.5, 0.25),   # 4 levels: 144,72,36,18
        norm='LN',
    ),

    # Match the number of feature levels SFP produces.
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
    batch_size=8,
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
# Optimisation: canonical Deformable DETR recipe (Option A)
# ---------------------------------------------------------------------------
#
# Reference Deformable DETR (Zhu et al., 2021 / mmdet r50_16xb2-50e):
#   base LR 2e-4, effective batch size 32, AdamW, wd 1e-4, grad clip 0.1.
#
# Our physical batch_size is 8 (single GPU). To reproduce the reference
# effective batch of 32 without more GPU memory, we accumulate gradients
# over 4 steps: 8 * 4 = 32. The backbone is frozen, so the reference's
# separate lr_backbone (2e-5) is not needed - the single 2e-4 LR applies
# only to the SFP neck + transformer encoder/decoder + bbox head.
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=1e-4, betas=(0.9, 0.999)),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    accumulative_counts=1,   # 4-GPU DDP: 8 per-GPU x 4 GPUs x accum1 = 32 effective
)

_max_epochs = 100

# Warmup is iteration-based. With accumulative_counts=4 the optimiser steps
# 4x less often per epoch, so the iteration-counted warmup is lengthened to
# 2000 to keep DETR's warmup-sensitive early phase stable. The LR drop is
# epoch-based and therefore unaffected by accumulation; milestone [80] keeps
# the reference 80%-of-training drop ratio for the 100-epoch upper bound.
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

work_dir = './outputs/work_dirs/deformable_detr_h1_1008_100epochs'

# 4-GPU DDP: MMEngine reads this, runs init_dist (per-rank GPU assignment
# + DDP model wrap + DistributedSampler sharding). Without it, each rank
# would train the full dataset (replication, not speedup).
launcher = 'pytorch'


# ---------------------------------------------------------------------------
# Early stopping  (appended override - evaluated last, so it wins)
# ---------------------------------------------------------------------------
#
# `max_epochs` is only an UPPER BOUND. The EarlyStoppingHook monitors
# coco/bbox_mAP on the patient-disjoint val set; CheckpointHook keeps
# save_best so the best epoch is always retained.
#
# Deformable DETR converges slowly and non-monotonically: bbox_mAP can
# plateau for many epochs and then jump. Patience is therefore set higher
# than the R-CNN heads (20 vs 12) with a smaller min_delta, so a late
# improvement is not cut off prematurely.

custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='coco/bbox_mAP',
        rule='greater',
        patience=20,
        min_delta=0.0005,
    ),
]
