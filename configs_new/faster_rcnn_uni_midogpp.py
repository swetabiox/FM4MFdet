# configs/faster_rcnn_uni_midogpp.py
#
# Faster R-CNN with a FROZEN UNI (ViT-L/16) backbone + SimpleFeaturePyramid
# neck, for mitotic-figure detection on MIDOG++ (1024x1024 patches).
#
# BACKBONE (UNI model card, MahmoodLab/UNI):
#   - ViT-Large/16  (patch size = 16), embed_dim = 1024
#   - img_size=1024 -> token map 64x64, physical stride 16
#
# NECK strides: scale_factors (2.0,1.0,0.5,0.25,0.125) on a stride-16 map give
# physical strides [8, 16, 32, 64, 128] -- these MATCH the head strides below
# because UNI is patch-16 (unlike H0/H1 patch-14). Do not reuse these strides
# for a patch-14 backbone without recomputing.
#
# AUGMENTATION matches the H0/H1 Deformable DETR configs (geometric + stain +
# photometric), so the backbone comparison is not confounded by augmentation.
#
# HYPERPARAMETERS follow standard Faster R-CNN practice (FPN defaults) with
# task-specific adjustments for small, single-class, frozen-backbone detection.
# See inline notes.

custom_imports = dict(
    imports=[
        'src.custom_mmdet.backbones.uni_vit',
        'src.custom_mmdet.necks.simple_feature_pyramid',
        'src.custom_mmdet.transforms.hed_stain_augment',
    ],
    allow_failed_imports=False
)

_base_ = 'mmdet::faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'

img_scale = (1024, 1024)

metainfo = dict(
    classes=('mitotic figure',),
    palette=[(220, 20, 60)],
)

model = dict(
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],   # ImageNet RGB mean (matches UNI)
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

    neck=dict(
        _delete_=True,
        type='SimpleFeaturePyramid',
        in_channels=1024,                 # UNI embed_dim (ViT-L)
        out_channels=256,
        scale_factors=(2.0, 1.0, 0.5, 0.25, 0.125),
        norm='LN',
    ),

    # -----------------------------------------------------------------------
    # RPN head
    # -----------------------------------------------------------------------
    # STANDARD FPN Faster R-CNN uses ONE anchor scale per level (scales=[8])
    # with the FPN strides [4,8,16,32,64] -> base anchor sizes 32..512, i.e.
    # one octave per level. Lin et al. (FPN, 2017) and the mmdet default both
    # use scales=[8], ratios=[0.5,1,2], one scale per level.
    #
    # Here strides are [8,16,32,64,128] and mitoses are small/near-isotropic.
    # We keep the LITERATURE-STANDARD single octave scale (scales=[8]) so each
    # level covers exactly one size band -> base anchors 64,128,...; the finest
    # level (stride 8 x scale 8 = 64 px) already brackets the ~50px mitosis
    # box, and the RPN regresses the rest. (The earlier scales=[4,8] doubled
    # anchors per level off-spec; reverting to the standard single octave.)
    rpn_head=dict(
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128],
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0., 0., 0., 0.],
            target_stds=[1.0, 1.0, 1.0, 1.0],
        ),
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0),
    ),

    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            # Standard FPN RoIExtractor uses the 4 finest levels for RoI
            # pooling (strides 4..32); the coarsest FPN level feeds RPN only.
            # Here we keep the finest 4 of our 5 levels [8,16,32,64], which is
            # the literature-standard arrangement and the right size band for
            # small mitoses.
            featmap_strides=[8, 16, 32, 64],
        ),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2],
            ),
            loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0),
        )
    ),

    # -----------------------------------------------------------------------
    # Train cfg -- standard Faster R-CNN RPN/RCNN assigner+sampler settings
    # (Ren et al. 2015 / mmdet defaults). These are the literature values;
    # kept unchanged because they are well-validated for single-class too.
    # -----------------------------------------------------------------------
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1
            ),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False
            ),
            allowed_border=-1,
            pos_weight=-1,
            debug=False
        ),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0
        ),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1
            ),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True
            ),
            pos_weight=-1,
            debug=False
        )
    ),

    # -----------------------------------------------------------------------
    # Test cfg
    # -----------------------------------------------------------------------
    # Low score_thr (0.05) so FROC/threshold-sweep has the full operating
    # range. max_per_img 300 (>COCO's 100) because a dense 1024 patch can hold
    # many mitoses. These are evaluation-driven, not model-quality changes.
    # -----------------------------------------------------------------------
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0
        ),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=300
        )
    )
)

# ---------------------------------------------------------------------------
# AUGMENTED training pipeline (matches H0/H1: geometric + stain + photometric)
# Order: geometric (operate on boxes too) -> stain -> photometric -> pack.
# All photometric/stain ops MODERATE because the backbone is FROZEN.
# ---------------------------------------------------------------------------
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),

    # --- geometric ---
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

    # --- pathology stain jitter (custom, shared transform) ---
    dict(
        type='HEDStainAugment',
        sigma=0.05,
        bias=0.02,
        prob=0.5,
    ),

    # --- generic photometric (brightness/contrast/saturation/hue) ---
    # Aligned to H0/H1 moderate values for cross-matrix consistency.
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
# Optimizer
# ---------------------------------------------------------------------------
# LITERATURE NOTE: the canonical Faster R-CNN / FPN recipe (Ren 2015, Lin 2017,
# Detectron2) uses SGD (lr 0.02, momentum 0.9, wd 1e-4) at batch 16 with a
# trainable backbone. Here the backbone is FROZEN and the neck uses LayerNorm,
# so AdamW is the appropriate optimizer (SGD+LN/ViT-neck is unstable). We keep
# AdamW lr 2e-4 (linear-scaled from 1e-4 at batch 8), wd 1e-4, and zero decay
# on norm/bias params -- standard for ViT-derived modules.
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
work_dir = './outputs/work_dirs/faster_rcnn_uni_1024_100epochs'


# ---------------------------------------------------------------------------
# Training loop, LR schedule, early stopping
# ---------------------------------------------------------------------------
# LITERATURE NOTE: standard Faster R-CNN uses a STEP schedule (drop x0.1 at
# 8/11 of a 12-epoch 1x run). With a frozen backbone + small task we use a
# longer budget (<=100 epochs) governed by early stopping; cosine decay over
# the budget is a common, well-behaved choice when the stopping point is not
# fixed in advance. Warmup is iteration-based (500 iters at batch 16).
_max_epochs = 100

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=_max_epochs,
    val_interval=1,
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False,
         begin=0, end=500),
    dict(type='CosineAnnealingLR', by_epoch=True, T_max=_max_epochs,
         eta_min=1e-7, begin=0, end=_max_epochs),
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


# ---------------------------------------------------------------------------
# Weights & Biases logging (online; tag 'augmented' to distinguish)
# ---------------------------------------------------------------------------
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='COMPAYL26',
            name='faster_rcnn_uni_midogpp',
            group='faster_rcnn',
            tags=['UNI', 'faster_rcnn', 'midogpp', 'frozen', 'augmented'],
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer',
)
