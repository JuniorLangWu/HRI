#LD
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
    cls_head=dict(type='GCNHead', num_classes=10, in_channels=128))

#测试
clip_len = 32
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

#配置文件
dataset = 'hcigesture'
ipn_pth = 'configs/online/best_LD_No.pth'
annotation_path = 'data/LD/online/hciall.json'
video_path = '/home/wjl/Videos/dataset/LD-ConGR/'
crop = 'data/LD/online/expand_crop_regions.json'
result_path = 'work_dirs/online/two_stage_online_results_hci_pose0.8_size4_81_None_det10_cls1_ac4_final0.9_0pre0.9'