# configs/retinanet_r50frozen_midogpp.py
#
# FROZEN ResNet-50 baseline for the COMPAYL26 RetinaNet row.
#
# ImageNet-pretrained ResNet-50 used STRICTLY AS A FROZEN feature extractor
# (like the FM ViTs), with native FPN + the IDENTICAL RetinaNet head,
# optimizer, schedule and augmentation as the frozen-FM cells. Controlled
# "frozen ImageNet CNN" vs "frozen pathology FM" baseline.
#
# FROZEN: frozen_stages=4, BN requires_grad=False + norm_eval=True -> entire
# backbone frozen incl. running stats; only FPN neck + RetinaNet head train.
# NECK: native FPN (P3..P7), strides [8,16,32,64,128]; RetinaNet anchors use
# these ResNet-native strides (NOT the patch-14 values of the ViT cells).

_base_ = 'mmdet::retinanet/retinanet_r50_fpn_1x_coco.py'

custom_imports = dict(
    imports=['src.custom_mmdet.transforms.hed_stain_augment'],
    allow_failed_imports=False,
)

img_scale = (1008, 1008)
metainfo = dict(classes=('mitotic figure',), palette=[(220, 20, 60)])

model = dict(
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
    ),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=4,                  # FREEZE EVERYTHING
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
    ),
    # RetinaNet's standard FPN: in C3,C4,C5 + 2 extra levels on input (P6,P7).
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5,
    ),
    # RetinaNet head -- IDENTICAL hyperparameters to the FM cells (octave base
    # scale 4, 3 scales/octave, ratios .5/1/2, focal gamma 2.0 alpha 0.25),
    # anchor strides are FPN-native [8,16,32,64,128].
    bbox_head=dict(
        num_classes=1,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128],
        ),
    ),
    test_cfg=dict(
        nms_pre=3000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=300,
    ),
)

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
work_dir = './outputs/work_dirs/retinanet_r50frozen_1008_100epochs'

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
        project='COMPAYL26', name='retinanet_r50frozen_midogpp',
        group='retinanet', tags=['ResNet-50', 'frozen', 'retinanet',
        'midogpp', 'baseline', 'augmented'])),
]
visualizer = dict(type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
