graph = 'handmp'
modality = ['j']
data_enhance='None'
type = 'STFGCN'
gcn_type = 'unit_gcnatt'
tcn_type = 'mstcn'
model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type=type,
        in_channels=3,
        gcn_adaptive='init',
        gcn_type = gcn_type,
        gcn_with_res=True,
        tcn_type = tcn_type,
        num_stages=6,
        down_stages=[6],
        inflate_stages=[6],
        
        graph_cfg=dict(layout=graph, mode='spatial')),
    train_cfg=dict(data_enhance=data_enhance),
    #更改分类数
    cls_head=dict(type='GCNHead', num_classes=10, in_channels=128))

dataset_type = 'PoseDataset'
#更改路径
ann_file = './data/LD/LD_No_mediapipe_crop.pkl'
clip_len = 32
train_pipeline = [
    dict(type='PreNormalize2D', img_shape=(720, 1280)),
    # dict(type='PreNormalize2D', threshold=0, mode='auto'),
    dict(type='GenSkeFeat', dataset=graph, feats=modality),
    dict(type='UniformSample', clip_len=clip_len),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
val_pipeline = [
    dict(type='PreNormalize2D', img_shape=(720, 1280)),
    # dict(type='PreNormalize2D', threshold=0, mode='auto'),
    dict(type='GenSkeFeat', dataset=graph, feats=modality),
    dict(type='UniformSample', clip_len=clip_len, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    dict(type='PreNormalize2D', img_shape=(720, 1280)),
    # dict(type='PreNormalize2D', threshold=0, mode='auto'),
    dict(type='GenSkeFeat', dataset=graph, feats=modality),
    dict(type='UniformSample', clip_len=clip_len, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
times = 1
batch = 32
#batch16*8GPUs lr=0.1
data = dict(
    #batchsize
    videos_per_gpu=batch,
    workers_per_gpu=4,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=times,
        #更改pkl描述
        dataset=dict(type=dataset_type, ann_file=ann_file, pipeline=train_pipeline, split='train')),
    val=dict(type=dataset_type, ann_file=ann_file, pipeline=val_pipeline, split='test'),
    test=dict(type=dataset_type, ann_file=ann_file, pipeline=test_pipeline, split='test'))

# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)
total_epochs = 100
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'], topk=(1, 5))
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])

# runtime settings
log_level = 'INFO'
work_dir = './work_dirs/model_base/{}/LD_No_{}_{}+{}_msst_j1_lr=0.1_len{}_times{}_batch{}_epochs{}'.\
            format(type,data_enhance,gcn_type,tcn_type,clip_len,times,batch,total_epochs)
