# model settings
model = dict(
    type='BimodalMemoryFasterRCNN',
    backbone1=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', 
                      checkpoint='/data1/huyuxuan/mmcv_imagenet_models/resnet50-19c8e357.pth')),
    backbone2=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', 
                      checkpoint='/data1/huyuxuan/mmcv_imagenet_models/resnet50-19c8e357.pth')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='VPAMemoryRoIHead',
        vpa_slot_size=100,
        dropout=0,
        temperature=1/16,
        loss_m1=dict(type='MSELoss', loss_weight=1.0),
        loss_m2=dict(type='MSELoss', loss_weight=1.0),
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=3,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ))
# dataset settings
dataset_type = 'BimodalFLIRDataset'
data_root = '/data1/huyuxuan/FLIR/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadPairedImagesFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PairedImagesResize', img_scale=(640, 512)),
    dict(type='PairedImagesRandomFlip', flip_ratio=0.5),
    dict(type='PairedImagesNormalize', img_norm_cfg1=img_norm_cfg, img_norm_cfg2=img_norm_cfg),
    dict(type='PairedImagesPad', size_divisor=32),
    dict(type='PairedImagesDefaultFormatBundle'),
    dict(type='PairedImagesCollect', keys=['img1', 'img2', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadPairedImagesFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 512),
        flip=False,
        transforms=[
            dict(type='PairedImagesResize'),
            dict(type='PairedImagesNormalize', img_norm_cfg1=img_norm_cfg, img_norm_cfg2=img_norm_cfg),
            dict(type='PairedImagesPad', size_divisor=32),
            dict(type='PairedImagesDefaultFormatBundle'),
            dict(type='PairedImagesCollect', keys=['img1', 'img2'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        img_list="/data1/huyuxuan/FLIR/train.txt",
        ann_file=data_root + 'train/trainlabelrtxt/',
        img_prefix1=data_root + 'train/trainimgr/',
        img_prefix2=data_root + 'train/trainimg/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        img_list="/data1/huyuxuan/FLIR/val.txt",
        ann_file=data_root + 'val/vallabelrtxt/',
        img_prefix1=data_root + 'val/valimgr/',
        img_prefix2=data_root + 'val/valimg/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        img_list="/data1/huyuxuan/FLIR/val.txt",
        ann_file=data_root + 'val/vallabelrtxt/',
        img_prefix1=data_root + 'val/valimgr/',
        img_prefix2=data_root + 'val/valimg/',
        pipeline=test_pipeline))
# evaluation
evaluation = dict(interval=1, metric='mAP')
# optimizer
optimizer = dict(type='SGD', lr=0.004, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=250,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
checkpoint_config = dict(interval=12)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]