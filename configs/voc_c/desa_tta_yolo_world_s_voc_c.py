# DESA-TTA with YOLO-World-S on VOC-C.
#
# One run adapts to ONE corruption stream, starting from the pretrained
# checkpoint. The default stream is `brightness`; pick another one with
#   --cfg-options test_dataloader.dataset.dataset.data_prefix.img=<corruption>/
# or run all 15 with tools/run_voc_c.sh.
_base_ = '../base/yolov8_s_syncbn_fast_8xb16-500e_coco.py'
custom_imports = dict(imports=['yolo_world'], allow_failed_imports=False)

# ----------------------------------------------------------------------------
# DESA-TTA hyper-parameters (paper notation in brackets)
# ----------------------------------------------------------------------------
tta_lr = dict(
    norm=6e-3,           # AdamW learning rate for the BN affine parameters
    score_thr=0.4,       # pseudo-label confidence threshold
    dynamic_ema=True,    # False -> fixed teacher momentum `ema` (=0.9996)
    # Dynamic EMA range: lower bound moves from m_min - delta_min to m_min
    # and upper bound from m_max - delta_max to m_max as the teacher drifts.
    m_min=0.9990,        # [m_min]
    delta_min=0.0060,    # [Delta_min]
    lambda_min=0.036,    # [lambda_min]
    m_max=0.9996,        # [m_max]
    delta_max=0.0016,    # [Delta_max]
    lambda_max=0.021,    # [lambda_max]
    p=2.0,               # [p] exponent of the drift-progress function
    M_max=300,           # [M_max] box count that saturates the density term
    # Source anchoring of the student.
    alpha_max=0.035,     # [alpha_max]
    lambda_alpha=0.07,   # [lambda_alpha]
    eps=1e-6)

# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
data_root = 'data/VOC-C/'
corruption = 'brightness'
class_text_path = 'data/texts/voc_class_texts.json'
metainfo = dict(classes=[
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'])
num_classes = 20
batch_size = 1

# Path to the pretrained YOLO-World-S checkpoint (see README). tools/test.py
# overrides this with its `checkpoint` argument.
load_from = 'pretrained_models/yolo_world_s.pth'

# ----------------------------------------------------------------------------
# Model: YOLO-World-S with frozen VOC text embeddings
# ----------------------------------------------------------------------------
text_channels = 512
neck_embed_channels = [128, 256, _base_.last_stage_out_channels // 2]
neck_num_heads = [4, 8, _base_.last_stage_out_channels // 2 // 32]

model = dict(
    type='DESAYOLOWorldDetector',
    mm_neck=True,
    num_train_classes=num_classes,
    num_test_classes=num_classes,
    embedding_path='embeddings/voc_embed.npy',
    prompt_dim=text_channels,
    data_preprocessor=dict(type='YOLOv5DetDataPreprocessor'),
    backbone=dict(
        _delete_=True,
        type='MultiModalYOLOBackbone',
        image_model={{_base_.model.backbone}},
        frozen_stages=4,
        text_model=None,
        with_text_model=False),
    neck=dict(
        type='YOLOWorldDualPAFPN',
        freeze_all=True,
        guide_channels=text_channels,
        embed_channels=neck_embed_channels,
        num_heads=neck_num_heads,
        block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConv')),
    bbox_head=dict(
        type='YOLOWorldHead',
        head_module=dict(
            type='YOLOWorldHeadModule',
            embed_dims=text_channels,
            freeze_all=True,
            num_classes=num_classes)),
    train_cfg=dict(assigner=dict(num_classes=num_classes)))

# ----------------------------------------------------------------------------
# Test-time pipeline: the teacher sees the weak view, the student the strong
# view. Pseudo-boxes are mapped from the weak to the strong view through
# `scale_factor`, `pad_param` and `homography_matrix`.
# ----------------------------------------------------------------------------
img_scale = (640, 640)
weak_branch = [
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(type='LoadText'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param', 'texts')),
]
strong_branch = [
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='mmdet.PhotoMetricDistortion', _scope_='mmdet'),
    dict(
        type='RandomApply',
        prob=0.7,
        transforms=[
            dict(
                type='mmdet.RandomErasing',
                n_patches=1,
                ratio=(0.141, 0.447),
                squared=False,
                img_border_value=(114, 114, 114),
                _scope_='mmdet')
        ]),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(type='LoadText'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction', 'scale_factor', 'pad_param',
                   'homography_matrix', 'texts'),
        _scope_='mmdet'),
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(
        type='mmdet.MultiBranch',
        branch_field=['weak', 'strong'],
        weak=weak_branch,
        strong=strong_branch),
]

voc_c_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5CocoDataset',
        metainfo=metainfo,
        data_root=data_root,
        ann_file='test.json',
        data_prefix=dict(img=corruption + '/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path=class_text_path,
    pipeline=test_pipeline)

val_dataloader = dict(dataset=voc_c_dataset, batch_size=batch_size)
test_dataloader = val_dataloader

val_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file=data_root + 'test.json',
    metric='bbox')
test_evaluator = val_evaluator

test_cfg = dict(type='TTALoop')
