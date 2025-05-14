# #IPN
graph = 'handmp'
modality = ['j']
data_enhance='None'
type = 'STFGCN'
gcn_type = 'unit_gcnatt'
tcn_type = 'mftcn'
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
    cls_head=dict(type='GCNHead', num_classes=13, in_channels=128))

#测试
clip_len = 32
test_pipeline = [
    dict(type='PreNormalize2D', img_shape=(240, 320)),
    # dict(type='PreNormalize2D', threshold=0, mode='auto'),
    dict(type='GenSkeFeat', dataset=graph, feats=modality),
    dict(type='UniformSample', clip_len=clip_len, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]

#配置文件
dataset = 'ipn'
ipn_pth = 'configs/online/best_ipn_No.pth'
annotation_path = 'data/ipn/online/ipnall.json'
video_path = '/home/wjl/Videos/dataset/IPN_Hand/'
result_path = 'work_dirs/online/two_stage_online_results_ipn_pose0.8_size40_801_None_det45_cls4_ac16_final0.9_0pre0.9'
