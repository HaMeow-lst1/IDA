import IDA_config
ida = IDA_config.IDA


_base_ = ['../_base_/models/yolov3.py', '../_base_/datasets/voc0712.py', '../_base_/default_runtime.py']
# model settings
# optimizer
optimizer = dict(type='SGD', lr=1e-5, momentum=0.9, weight_decay=5e-4)
optimizer_config = dict()
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 20])
checkpoint_config = dict(interval=1)
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=24)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (1 GPUs) x (4 samples per GPU)
auto_scale_lr = dict(base_batch_size=4)


load_from = ida.pre_model_path['YOLOv3']
