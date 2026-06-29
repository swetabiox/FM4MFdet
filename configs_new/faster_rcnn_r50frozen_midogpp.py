# configs/faster_rcnn_r50frozen_midogpp.py
#
# FROZEN ResNet-50 baseline for the COMPAYL26 Faster R-CNN row.
#
# This is the CONTROLLED frozen-backbone baseline: an ImageNet-pretrained
# ResNet-50 used STRICTLY AS A FROZEN FEATURE EXTRACTOR (like the FM ViTs),
# with the detector's NATIVE FPN neck + the IDENTICAL Faster R-CNN head,
# optimizer, schedule and augmentation as the frozen-FM cells.
#
# Purpose: isolates "frozen ImageNet CNN features" vs "frozen pathology FM
# features" with everything downstream held constant. Differs from the
# FULLY-FINETUNED ResNet baseline (which adapts the backbone).
#
# FROZEN MECHANICS:
#   - frozen_stages=4         -> freeze stem + all 4 stages (whole backbone)
#   - norm_cfg requires_grad=False + norm_eval=True -> BN affine params frozen
#     AND BN running stats held fixed (no update). Nothing in the backbone
#     trains; only the FPN neck + RPN/RoI heads are learnable.
#
# NECK: native FPN over C2..C5 (ResNet is natively multi-scale, so NO
# SimpleFeaturePyramid -- that is the ViT adapter). Standard FPN strides
# [4,8,16,32,64], so the RPN anchor strides use the mmdet default (NOT the
# patch-14 [7,14,28,56,112] used by the ViT cells).

_base_ = 'mmdet::faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'

custom_imports = dict(
    imports=['src.custom_mmdet.transforms.hed_stain_augment'],
    allow_failed_imports=False,
)

img_scale = (1008, 1008)

metainfo = dict(classes=('mitotic figure',), palette=[(220, 20, 60)])

model = dict(
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],   # ImageNet RGB mean (ResNet pretrain)
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,              # FPN needs /32-divisible padding
    ),

    # FROZEN ResNet-50: whole backbone frozen, BN stats fixed.
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),         # C2,C3,C4,C5 -> FPN
        frozen_stages=4,                  # FREEZE EVERYTHING
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
    ),

    # Native FPN (inherited from base; stated explicitly). 4 ResNet stages
    # -> 5 pyramid levels (P2..P6), 256-d.
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
    ),

    # Faster R-CNN head -- IDENTICAL hyperparameters to the FM cells, but
    # anchor strides are the FPN defaults [4,8,16,32,64] (ResNet-native), NOT
    # the patch-14 strides. Single anchor octave scale 8, ratios .5/1/2.
    rpn_head=dict(
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64],   # FPN-native strides
        ),
    ),
    roi_head=dict(
        bbox_head=dict(num_classes=1),
    ),
    test_cfg=dict(
        rpn=dict(nms_pre=2000, max_per_img=1000,
                 nms=dict(type='nms', iou_threshold=0.7), min_bbox_size=0),
        rcnn=dict(score_thr=0.05, nms=dict(type='nms', iou_threshold=0.5),
                  max_per_img=300),
    ),
)

# ---------------------------------------------------------------------------
# AUGMENTED pipeline / data / evaluators -- IDENTICAL to the FM cells.
# ---------------------------------------------------------------------------
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=img_scale, keep_ratio=False, backend='pillow'),
    dict(type='RandomFlip', prob=0.5, direction=['horizontal', 'vertical']),
    dict(type='RandomAffine', max_rotate_degree=15.0, max_translate_ratio=0.05,
         scaling_ratio_range=(0.9, 1.1), max_shear_degree=0.0,
         border=(0, 0), border_val=(114, 114, 114)),
    dict(type='HEDStainAugment', sigma=0.05, bias=0.02, prob=0.5),
    dict(type='PhotoMetricDistortion', brightness_delta=16,
         contrast_range=(0.9, 1.1), saturation_range=(0.9, 1.1), hue_delta=10),
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
    batch_size=16, num_workers=8, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(type='CocoDataset', data_root=data_root,
        ann_file='coco_annotations/patches_1008/midogpp_train.json',
        data_prefix=dict(img='Datensatz/patches_1008/'),
        metainfo=metainfo, filter_cfg=dict(filter_empty_gt=False),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=4, num_workers=4, persistent_workers=True, drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type='CocoDataset', data_root=data_root,
        ann_file='coco_annotations/patches_1008/midogpp_val.json',
        data_prefix=dict(img='Datensatz/patches_1008/'),
        metainfo=metainfo, test_mode=True, pipeline=val_pipeline))
test_dataloader = dict(
    batch_size=4, num_workers=4, persistent_workers=True, drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type='CocoDataset', data_root=data_root,
        ann_file='coco_annotations/patches_1008/midogpp_test.json',
        data_prefix=dict(img='Datensatz/patches_1008/'),
        metainfo=metainfo, test_mode=True, pipeline=test_pipeline))

val_evaluator = dict(type='CocoMetric',
    ann_file='data/coco_annotations/patches_1008/midogpp_val.json',
    metric='bbox', format_only=False, backend_args=None)
test_evaluator = dict(type='CocoMetric',
    ann_file='data/coco_annotations/patches_1008/midogpp_test.json',
    metric='bbox', format_only=False, backend_args=None)

# Optimizer -- IDENTICAL to FM cells (AdamW; only neck+head train).
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=1e-4, betas=(0.9, 0.999)),
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0),
    clip_grad=dict(max_norm=35, norm_type=2),
)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3,
                    save_best='coco/bbox_mAP', rule='greater'),
    logger=dict(type='LoggerHook', interval=10, log_metric_by_epoch=True),
)
env_cfg = dict(cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
randomness = dict(seed=42, deterministic=False, diff_rank_seed=True)
resume = False
work_dir = './outputs/work_dirs/faster_rcnn_r50frozen_1008_100epochs'

_max_epochs = 100
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=_max_epochs, val_interval=1)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='CosineAnnealingLR', by_epoch=True, T_max=_max_epochs,
         eta_min=1e-7, begin=0, end=_max_epochs),
]
custom_hooks = [
    dict(type='EarlyStoppingHook', monitor='coco/bbox_mAP', rule='greater',
         patience=10, min_delta=0.001),
]
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend', init_kwargs=dict(
        project='COMPAYL26', name='faster_rcnn_r50frozen_midogpp',
        group='faster_rcnn', tags=['ResNet-50', 'frozen', 'faster_rcnn',
        'midogpp', 'baseline', 'augmented'])),
]
visualizer = dict(type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
