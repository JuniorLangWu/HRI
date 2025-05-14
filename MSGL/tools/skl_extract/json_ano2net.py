#根据中间annos生成hrnet标签
from mmcv import load, dump
from pyskl.smp import *

# ###IPNvideotrack15变为snip级
# # 1.改json路径和视频
# train = load('./data/ipn/ipnall_train.json')
# test = load('./data/ipn/ipnall_test.json')
# lines = train + test
# annotations_videos = load('./data/ipn/ipnall_mediapipe_track15_videos_annos.pkl')
# annotations_med = load('./data/ipn/ipnall_mediapipe_annos.pkl')
# # print(annotations_videos[0])
# # exit(0)
# annotations_hand = []
# annos = dict()
# for iter in range(len(annotations_videos)):
#     print(annotations_videos[iter]['keypoint'].shape)
#     for i, vid_line in enumerate(lines):
#         annotations = '{}:{}_{}'.format(vid_line['vid_name'], vid_line['start_frame'],  vid_line['end_frame'])        
#         if annotations_videos[iter]['frame_dir'] in annotations:
            
#             annos['frame_dir'] = annotations
#             annos['label'] = vid_line['label']
#             annos['img_shape'] = annotations_videos[iter]['img_shape']
#             annos['num_hand_raw'] = 1
#             annos['keypoint'] = annotations_videos[iter]['keypoint'][:, vid_line['start_frame']-1:vid_line['end_frame'], :, :]
#             #他标帧数有问题，不太准确视频
#             annos['total_frames'] = annos['keypoint'].shape[1]
#             annos['keypoint_score'] = annotations_videos[iter]['keypoint_score'][:, vid_line['start_frame']-1:vid_line['end_frame'], :]
#             annotations_hand.append(annos.copy())
#             # print(annotations_videos[iter]['frame_dir'])
#             # print(annotations)
#             print(annotations_hand[i]['frame_dir'])
#             print(annotations_hand[i]['keypoint'].shape)
#             print(annotations_hand[i]['label'])
#             print('-----'*5)
#             print(annotations_med[i]['frame_dir'])
#             print(annotations_med[i]['keypoint'].shape)
#             print(annotations_med[i]['label'])
#             print('-----'*10)
#     # if iter >=1:        
#     #     exit(0)
#     print('#####'*10)
# # 2.改annos路径
# dump(annotations_hand, './data/ipn/ipnall_mediapipe_track15_snip_annos.pkl') 
# ###ipnhandtrack15
# #读取
# ipnhand = load('./data/ipn/ipnall_mediapipe_track15_snip_annos.pkl')
# # print(ipnhand[0])
# #去除无手势0
# ipn = [item for item in ipnhand if item['label'] != 0]
# #替换mediapipe标签
# for ipn_iter in ipn:
#     ipn_iter['label'] = ipn_iter['label'] - 1
# dump(ipn, './data/ipn/ipn_No_mediapipe_track15_snip_annos.pkl')
# #转化为net
# train_No = load('./data/ipn/ipn_No_train.json')
# test_No = load('./data/ipn/ipn_No_test.json')
# annotations_No_snip = load('./data/ipn/ipn_No_mediapipe_track15_snip_annos.pkl')
# split = dict()
# split['train'] = [x['vid_name'] + ':' + str(x["start_frame"]) + '_' + str(x["end_frame"]) for x in train_No]
# split['test'] = [x['vid_name'] + ':' + str(x["start_frame"]) + '_' + str(x["end_frame"]) for x in test_No]
# dump(dict(split=split, annotations=annotations_No_snip), './data/ipn/ipn_No_mediapipe_track15_snip.pkl')
# #转化为ipnallnet
# train = load('./data/ipn/ipnall_train.json')
# test = load('./data/ipn/ipnall_test.json')
# annotations_all_snip = load('./data/ipn/ipnall_mediapipe_track15_snip_annos.pkl')
# split = dict()
# split['train'] = [x['vid_name'] + ':' + str(x["start_frame"]) + '_' + str(x["end_frame"]) for x in train]
# split['test'] = [x['vid_name'] + ':' + str(x["start_frame"]) + '_' + str(x["end_frame"]) for x in test]
# dump(dict(split=split, annotations=annotations_all_snip), './data/ipn/ipnall_mediapipe_track15_snip.pkl')

# ###IPNvideotrackhand2变为snip级
# # 1.改json路径和视频
# train = load('./data/ipn/ipnall_train.json')
# test = load('./data/ipn/ipnall_test.json')
# lines = train + test
# annotations_videos = load('./data/ipn/ipnall_mediapipe_trackhand2_videos_annos.pkl')
# annotations_med = load('./data/ipn/ipnall_mediapipe_annos.pkl')
# # print(annotations_videos[0])
# # exit(0)
# annotations_hand = []
# annos = dict()
# for iter in range(len(annotations_videos)):
#     print(annotations_videos[iter]['keypoint'].shape)
#     for i, vid_line in enumerate(lines):
#         annotations = '{}:{}_{}'.format(vid_line['vid_name'], vid_line['start_frame'],  vid_line['end_frame'])        
#         if annotations_videos[iter]['frame_dir'] in annotations:
            
#             annos['frame_dir'] = annotations
#             annos['label'] = vid_line['label']
#             annos['img_shape'] = annotations_videos[iter]['img_shape']
#             annos['num_hand_raw'] = 1
#             annos['keypoint'] = annotations_videos[iter]['keypoint'][:, vid_line['start_frame']-1:vid_line['end_frame'], :, :]
#             #他标帧数有问题，不太准确视频
#             annos['total_frames'] = annos['keypoint'].shape[1]
#             annos['keypoint_score'] = annotations_videos[iter]['keypoint_score'][:, vid_line['start_frame']-1:vid_line['end_frame'], :]
#             annotations_hand.append(annos.copy())
#             # print(annotations_videos[iter]['frame_dir'])
#             # print(annotations)
#             print(annotations_hand[i]['frame_dir'])
#             print(annotations_hand[i]['keypoint'].shape)
#             print(annotations_hand[i]['label'])
#             print('-----'*5)
#             print(annotations_med[i]['frame_dir'])
#             print(annotations_med[i]['keypoint'].shape)
#             print(annotations_med[i]['label'])
#             print('-----'*10)
#     # if iter >=1:        
#     #     exit(0)
#     print('#####'*10)
# # 2.改annos路径
# dump(annotations_hand, './data/ipn/ipnall_mediapipe_trackhand2_snip_annos.pkl') 

# ###ipnsnip
# #读取
# ipnhand = load('./data/ipn/ipnall_mediapipe_trackhand2_snip_annos.pkl')
# # print(ipnhand[0])
# #去除无手势0
# ipn = [item for item in ipnhand if item['label'] != 0]
# #替换mediapipe标签
# for ipn_iter in ipn:
#     ipn_iter['label'] = ipn_iter['label'] - 1
# dump(ipn, './data/ipn/ipn_No_mediapipe_trackhand2_snip_annos.pkl')
# #转化为net
# train_No = load('./data/ipn/ipn_No_train.json')
# test_No = load('./data/ipn/ipn_No_test.json')
# annotations_No_snip = load('./data/ipn/ipn_No_mediapipe_trackhand2_snip_annos.pkl')
# split = dict()
# split['train'] = [x['vid_name'] + ':' + str(x["start_frame"]) + '_' + str(x["end_frame"]) for x in train_No]
# split['test'] = [x['vid_name'] + ':' + str(x["start_frame"]) + '_' + str(x["end_frame"]) for x in test_No]
# dump(dict(split=split, annotations=annotations_No_snip), './data/ipn/ipn_No_mediapipe_trackhand2_snip.pkl')

# #转化为ipnallnet
# train = load('./data/ipn/ipnall_train.json')
# test = load('./data/ipn/ipnall_test.json')
# annotations_all_snip = load('./data/ipn/ipnall_mediapipe_trackhand2_snip_annos.pkl')
# split = dict()
# split['train'] = [x['vid_name'] + ':' + str(x["start_frame"]) + '_' + str(x["end_frame"]) for x in train]
# split['test'] = [x['vid_name'] + ':' + str(x["start_frame"]) + '_' + str(x["end_frame"]) for x in test]
# dump(dict(split=split, annotations=annotations_all_snip), './data/ipn/ipnall_mediapipe_trackhand2_snip.pkl')


# ###IPNvideotrack变为snip级
# # 1.改json路径和视频
# train = load('./data/ipn/ipnall_train.json')
# test = load('./data/ipn/ipnall_test.json')
# lines = train + test
# annotations_videos = load('./data/ipn/ipnall_mediapipe_track_videos_annos.pkl')
# annotations_med = load('./data/ipn/ipnall_mediapipe_annos.pkl')
# # print(annotations_videos[0])
# # exit(0)
# annotations_hand = []
# annos = dict()
# for iter in range(len(annotations_videos)):
#     print(annotations_videos[iter]['keypoint'].shape)
#     for i, vid_line in enumerate(lines):
#         annotations = '{}:{}_{}'.format(vid_line['vid_name'], vid_line['start_frame'],  vid_line['end_frame'])        
#         if annotations_videos[iter]['frame_dir'] in annotations:
            
#             annos['frame_dir'] = annotations
#             annos['label'] = vid_line['label']
#             annos['img_shape'] = annotations_videos[iter]['img_shape']
#             annos['num_hand_raw'] = 1
#             annos['keypoint'] = annotations_videos[iter]['keypoint'][:, vid_line['start_frame']-1:vid_line['end_frame'], :, :]
#             #他标帧数有问题，不太准确视频
#             annos['total_frames'] = annos['keypoint'].shape[1]
#             annos['keypoint_score'] = annotations_videos[iter]['keypoint_score'][:, vid_line['start_frame']-1:vid_line['end_frame'], :]
#             annotations_hand.append(annos.copy())
#             # print(annotations_videos[iter]['frame_dir'])
#             # print(annotations)
#             print(annotations_hand[i]['frame_dir'])
#             print(annotations_hand[i]['keypoint'].shape)
#             print(annotations_hand[i]['label'])
#             print('-----'*5)
#             print(annotations_med[i]['frame_dir'])
#             print(annotations_med[i]['keypoint'].shape)
#             print(annotations_med[i]['label'])
#             print('-----'*10)
#     # if iter >=1:        
#     #     exit(0)
#     print('#####'*10)
# # 2.改annos路径
# dump(annotations_hand, './data/ipn/ipnall_mediapipe_track_snip_annos.pkl') 

# ###ipnsnip
# #读取
# ipnhand = load('./data/ipn/ipnall_mediapipe_track_snip_annos.pkl')
# # print(ipnhand[0])
# #去除无手势0
# ipn = [item for item in ipnhand if item['label'] != 0]
# #替换mediapipe标签
# for ipn_iter in ipn:
#     ipn_iter['label'] = ipn_iter['label'] - 1
# dump(ipn, './data/ipn/ipn_No_mediapipe_track_snip_annos.pkl')
# #转化为net
# train_No = load('./data/ipn/ipn_No_train.json')
# test_No = load('./data/ipn/ipn_No_test.json')
# annotations_No_snip = load('./data/ipn/ipn_No_mediapipe_track_snip_annos.pkl')
# split = dict()
# split['train'] = [x['vid_name'] + ':' + str(x["start_frame"]) + '_' + str(x["end_frame"]) for x in train_No]
# split['test'] = [x['vid_name'] + ':' + str(x["start_frame"]) + '_' + str(x["end_frame"]) for x in test_No]
# dump(dict(split=split, annotations=annotations_No_snip), './data/ipn/ipn_No_mediapipe_track_snip.pkl')

# #转化为ipnallnet
# train = load('./data/ipn/ipnall_train.json')
# test = load('./data/ipn/ipnall_test.json')
# annotations_all_snip = load('./data/ipn/ipnall_mediapipe_track_snip_annos.pkl')
# split = dict()
# split['train'] = [x['vid_name'] + ':' + str(x["start_frame"]) + '_' + str(x["end_frame"]) for x in train]
# split['test'] = [x['vid_name'] + ':' + str(x["start_frame"]) + '_' + str(x["end_frame"]) for x in test]
# dump(dict(split=split, annotations=annotations_all_snip), './data/ipn/ipnall_mediapipe_track_snip.pkl')

# ###IPNvideohand2变为snip级
# # 1.改json路径和视频
# train = load('./data/ipn/ipnall_train.json')
# test = load('./data/ipn/ipnall_test.json')
# lines = train + test
# annotations_videos = load('./data/ipn/ipnall_mediapipe_hand2_videos_annos.pkl')
# annotations_med = load('./data/ipn/ipnall_mediapipe_annos.pkl')
# # print(annotations_videos[0])
# # exit(0)
# annotations_hand = []
# annos = dict()
# for iter in range(len(annotations_videos)):
#     print(annotations_videos[iter]['keypoint'].shape)
#     for i, vid_line in enumerate(lines):
#         annotations = '{}:{}_{}'.format(vid_line['vid_name'], vid_line['start_frame'],  vid_line['end_frame'])        
#         if annotations_videos[iter]['frame_dir'] in annotations:
            
#             annos['frame_dir'] = annotations
#             annos['label'] = vid_line['label']
#             annos['img_shape'] = annotations_videos[iter]['img_shape']
#             annos['num_hand_raw'] = 1
#             annos['keypoint'] = annotations_videos[iter]['keypoint'][:, vid_line['start_frame']-1:vid_line['end_frame'], :, :]
#             #他标帧数有问题，不太准确视频
#             annos['total_frames'] = annos['keypoint'].shape[1]
#             annos['keypoint_score'] = annotations_videos[iter]['keypoint_score'][:, vid_line['start_frame']-1:vid_line['end_frame'], :]
#             annotations_hand.append(annos.copy())
#             # print(annotations_videos[iter]['frame_dir'])
#             # print(annotations)
#             print(annotations_hand[i]['frame_dir'])
#             print(annotations_hand[i]['keypoint'].shape)
#             print(annotations_hand[i]['label'])
#             print('-----'*5)
#             print(annotations_med[i]['frame_dir'])
#             print(annotations_med[i]['keypoint'].shape)
#             print(annotations_med[i]['label'])
#             print('-----'*10)
#     # if iter >=1:        
#     #     exit(0)
#     print('#####'*10)
# # 2.改annos路径
# dump(annotations_hand, './data/ipn/ipnall_mediapipe_hand2_snip_annos.pkl') 

# ###ipnsnip
# #读取
# ipnhand = load('./data/ipn/ipnall_mediapipe_hand2_snip_annos.pkl')
# # print(ipnhand[0])
# #去除无手势0
# ipn = [item for item in ipnhand if item['label'] != 0]
# #替换mediapipe标签
# for ipn_iter in ipn:
#     ipn_iter['label'] = ipn_iter['label'] - 1
# dump(ipn, './data/ipn/ipn_No_mediapipe_hand2_snip_annos.pkl')
# #转化为net
# train_No = load('./data/ipn/ipn_No_train.json')
# test_No = load('./data/ipn/ipn_No_test.json')
# annotations_No_snip = load('./data/ipn/ipn_No_mediapipe_hand2_snip_annos.pkl')
# split = dict()
# split['train'] = [x['vid_name'] + ':' + str(x["start_frame"]) + '_' + str(x["end_frame"]) for x in train_No]
# split['test'] = [x['vid_name'] + ':' + str(x["start_frame"]) + '_' + str(x["end_frame"]) for x in test_No]
# dump(dict(split=split, annotations=annotations_No_snip), './data/ipn/ipn_No_mediapipe_hand2_snip.pkl')

# #转化为ipnallnet
# train = load('./data/ipn/ipnall_train.json')
# test = load('./data/ipn/ipnall_test.json')
# annotations_all_snip = load('./data/ipn/ipnall_mediapipe_hand2_snip_annos.pkl')
# split = dict()
# split['train'] = [x['vid_name'] + ':' + str(x["start_frame"]) + '_' + str(x["end_frame"]) for x in train]
# split['test'] = [x['vid_name'] + ':' + str(x["start_frame"]) + '_' + str(x["end_frame"]) for x in test]
# dump(dict(split=split, annotations=annotations_all_snip), './data/ipn/ipnall_mediapipe_hand2_snip.pkl')


# annotations_videos = load('./data/ipn/ipn_No_mediapipe_snip.pkl')
# annotations_med = load('./data/ipn/ipn_No_mediapipe.pkl')
# # print(annotations_videos['split'])
# print(annotations_videos['annotations'][4039])
# # print(annotations_med['split'])
# print(annotations_med['annotations'][4039])
# exit(0)

# ###IPNvideo变为snip级
# # 1.改json路径和视频
# train = load('./data/ipn/ipnall_train.json')
# test = load('./data/ipn/ipnall_test.json')
# lines = train + test
# annotations_videos = load('./data/ipn/ipnall_mediapipe_videos_annos.pkl')
# annotations_med = load('./data/ipn/ipnall_mediapipe_annos.pkl')
# # print(annotations_videos[0])
# # exit(0)
# annotations_hand = []
# annos = dict()
# for iter in range(len(annotations_videos)):
#     print(annotations_videos[iter]['keypoint'].shape)
#     for i, vid_line in enumerate(lines):
#         annotations = '{}:{}_{}'.format(vid_line['vid_name'], vid_line['start_frame'],  vid_line['end_frame'])        
#         if annotations_videos[iter]['frame_dir'] in annotations:
            
#             annos['frame_dir'] = annotations
#             annos['label'] = vid_line['label']
#             annos['img_shape'] = annotations_videos[iter]['img_shape']
#             annos['num_hand_raw'] = 1
#             annos['keypoint'] = annotations_videos[iter]['keypoint'][:, vid_line['start_frame']-1:vid_line['end_frame'], :, :]
#             #他标帧数有问题，不太准确视频
#             annos['total_frames'] = annos['keypoint'].shape[1]
#             annos['keypoint_score'] = annotations_videos[iter]['keypoint_score'][:, vid_line['start_frame']-1:vid_line['end_frame'], :]
#             annotations_hand.append(annos.copy())
#             # print(annotations_videos[iter]['frame_dir'])
#             # print(annotations)
#             print(annotations_hand[i]['frame_dir'])
#             print(annotations_hand[i]['keypoint'].shape)
#             print(annotations_hand[i]['label'])
#             print('-----'*5)
#             print(annotations_med[i]['frame_dir'])
#             print(annotations_med[i]['keypoint'].shape)
#             print(annotations_med[i]['label'])
#             print('-----'*10)
#     # if iter >=1:        
#     #     exit(0)
#     print('#####'*10)
# # 2.改annos路径
# dump(annotations_hand, './data/ipn/ipnall_mediapipe_snip_annos.pkl') 

# ###ipnsnip
# #读取
# ipnhand = load('./data/ipn/ipnall_mediapipe_snip_annos.pkl')
# # print(ipnhand[0])
# #去除无手势0
# ipn = [item for item in ipnhand if item['label'] != 0]
# #替换mediapipe标签
# for ipn_iter in ipn:
#     ipn_iter['label'] = ipn_iter['label'] - 1
# dump(ipn, './data/ipn/ipn_No_mediapipe_snip_annos.pkl')
# #转化为net
# train_No = load('./data/ipn/ipn_No_train.json')
# test_No = load('./data/ipn/ipn_No_test.json')
# annotations_No_snip = load('./data/ipn/ipn_No_mediapipe_snip_annos.pkl')
# split = dict()
# split['train'] = [x['vid_name'] + ':' + str(x["start_frame"]) + '_' + str(x["end_frame"]) for x in train_No]
# split['test'] = [x['vid_name'] + ':' + str(x["start_frame"]) + '_' + str(x["end_frame"]) for x in test_No]
# dump(dict(split=split, annotations=annotations_No_snip), './data/ipn/ipn_No_mediapipe_snip.pkl')

# #转化为ipnallnet
# train = load('./data/ipn/ipnall_train.json')
# test = load('./data/ipn/ipnall_test.json')
# annotations_all_snip = load('./data/ipn/ipnall_mediapipe_snip_annos.pkl')
# split = dict()
# split['train'] = [x['vid_name'] + ':' + str(x["start_frame"]) + '_' + str(x["end_frame"]) for x in train]
# split['test'] = [x['vid_name'] + ':' + str(x["start_frame"]) + '_' + str(x["end_frame"]) for x in test]
# dump(dict(split=split, annotations=annotations_all_snip), './data/ipn/ipnall_mediapipe_snip.pkl')

###LDvideostrackcrop变为snip级
# 1.改json路径和视频
train = load('./data/LD/LDall_train.json')
test = load('./data/LD/LDall_test.json')
lines = train + test
annotations_crop = load('./data/LD/LDall_mediapipe_track_videos_crop_annos.pkl')
annotations_med = load('./data/LD/LDall_mediapipe_crop_annos.pkl')
# print(annotations_crop[0])
# exit(0)
annotations_hand = []
annos = dict()
for iter in range(len(annotations_crop)):
    print(annotations_crop[iter]['keypoint'].shape)
    for i, vid_line in enumerate(lines):
        annotations = '{}:{}_{}'.format(vid_line['vid_name'], vid_line['start_frame'],  vid_line['end_frame'])        
        if annotations_crop[iter]['frame_dir'] in annotations:
            
            annos['frame_dir'] = annotations
            annos['label'] = vid_line['label']
            annos['img_shape'] = annotations_crop[iter]['img_shape']
            annos['total_frames'] = vid_line['end_frame'] - vid_line['start_frame'] +1
            annos['num_hand_raw'] = 1
            annos['keypoint'] = annotations_crop[iter]['keypoint'][:, vid_line['start_frame']-1:vid_line['end_frame'], :, :]
            annos['keypoint_score'] = annotations_crop[iter]['keypoint_score'][:, vid_line['start_frame']-1:vid_line['end_frame'], :]
            annotations_hand.append(annos.copy())
            # print(annotations_crop[iter]['frame_dir'])
            # print(annotations)
            print(annotations_hand[i]['keypoint'].shape)
            print(annotations_med[i]['keypoint'].shape)
            print('-----'*10)
    # if iter >=1:        
    #     exit(0)
    print('#####'*10)
# 2.改annos路径
dump(annotations_hand, './data/LD/LDall_mediapipe_track_crop_annos.pkl') 

###LDcrop 
#读取
LDhand = load('./data/LD/LDall_mediapipe_track_crop_annos.pkl')
# print(LDhand[0])
LD = [item for item in LDhand if item['label'] != 10]
# print(LD)
dump(LD, './data/LD/LD_No_mediapipe_track_crop_annos.pkl')
#转化为net
train = load('./data/LD/LD_No_train.json')
test = load('./data/LD/LD_No_test.json')
annotations_No_crop = load('./data/LD/LD_No_mediapipe_track_crop_annos.pkl')
split = dict()
split['train'] = [x['vid_name'] + ':' + str(x["start_frame"]) + '_' + str(x["end_frame"]) for x in train]
split['test'] = [x['vid_name'] + ':' + str(x["start_frame"]) + '_' + str(x["end_frame"]) for x in test]
dump(dict(split=split, annotations=annotations_No_crop), './data/LD/LD_No_mediapipe_track_crop.pkl')

#转化为LDallnet
train = load('./data/LD/LDall_train.json')
test = load('./data/LD/LDall_test.json')
annotations_all_crop = load('./data/LD/LDall_mediapipe_track_crop_annos.pkl')
split = dict()
split['train'] = [x['vid_name'] + ':' + str(x["start_frame"]) + '_' + str(x["end_frame"]) for x in train]
split['test'] = [x['vid_name'] + ':' + str(x["start_frame"]) + '_' + str(x["end_frame"]) for x in test]
dump(dict(split=split, annotations=annotations_all_crop), './data/LD/LDall_mediapipe_track_crop.pkl')

# ###LDvideoscrop变为snip级
# # 1.改json路径和视频
# train = load('./data/LD/LDall_train.json')
# test = load('./data/LD/LDall_test.json')
# lines = train + test
# annotations_crop = load('./data/LD/LDall_mediapipe_videos_crop_annos.pkl')
# annotations_med = load('./data/LD/LDall_mediapipe_annos.pkl')
# # print(annotations_crop[0])
# # exit(0)
# annotations_hand = []
# annos = dict()
# for iter in range(len(annotations_crop)):
#     print(annotations_crop[iter]['keypoint'].shape)
#     for i, vid_line in enumerate(lines):
#         annotations = '{}:{}_{}'.format(vid_line['vid_name'], vid_line['start_frame'],  vid_line['end_frame'])        
#         if annotations_crop[iter]['frame_dir'] in annotations:
            
#             annos['frame_dir'] = annotations
#             annos['label'] = vid_line['label']
#             annos['img_shape'] = annotations_crop[iter]['img_shape']
#             annos['total_frames'] = vid_line['end_frame'] - vid_line['start_frame'] +1
#             annos['num_hand_raw'] = 1
#             annos['keypoint'] = annotations_crop[iter]['keypoint'][:, vid_line['start_frame']-1:vid_line['end_frame'], :, :]
#             annos['keypoint_score'] = annotations_crop[iter]['keypoint_score'][:, vid_line['start_frame']-1:vid_line['end_frame'], :]
#             annotations_hand.append(annos.copy())
#             # print(annotations_crop[iter]['frame_dir'])
#             # print(annotations)
#             print(annotations_hand[i]['keypoint'].shape)
#             print(annotations_med[i]['keypoint'].shape)
#             print('-----'*10)
#     # if iter >=1:        
#     #     exit(0)
#     print('#####'*10)
# # 2.改annos路径
# dump(annotations_hand, './data/LD/LDall_mediapipe_crop_annos.pkl') 

# ###LDcrop 
# #读取
# LDhand = load('./data/LD/LDall_mediapipe_crop_annos.pkl')
# # print(LDhand[0])
# LD = [item for item in LDhand if item['label'] != 10]
# # print(LD)
# dump(LD, './data/LD/LD_No_mediapipe_crop_annos.pkl')
# #转化为net
# train = load('./data/LD/LD_No_train.json')
# test = load('./data/LD/LD_No_test.json')
# annotations_No_crop = load('./data/LD/LD_No_mediapipe_crop_annos.pkl')
# split = dict()
# split['train'] = [x['vid_name'] + ':' + str(x["start_frame"]) + '_' + str(x["end_frame"]) for x in train]
# split['test'] = [x['vid_name'] + ':' + str(x["start_frame"]) + '_' + str(x["end_frame"]) for x in test]
# dump(dict(split=split, annotations=annotations_No_crop), './data/LD/LD_No_mediapipe_crop.pkl')

# #转化为LDallnet
# train = load('./data/LD/LDall_train.json')
# test = load('./data/LD/LDall_test.json')
# annotations_all_crop = load('./data/LD/LDall_mediapipe_crop_annos.pkl')
# split = dict()
# split['train'] = [x['vid_name'] + ':' + str(x["start_frame"]) + '_' + str(x["end_frame"]) for x in train]
# split['test'] = [x['vid_name'] + ':' + str(x["start_frame"]) + '_' + str(x["end_frame"]) for x in test]
# dump(dict(split=split, annotations=annotations_all_crop), './data/LD/LDall_mediapipe_crop.pkl')

###LDvideos变为snip级
# 3.改json路径和视频
# train = load('./data/LD/LDall_train.json')
# test = load('./data/LD/LDall_test.json')
# lines = train + test
# annotations_ps = load('./data/LD/LDall_mediapipe_videos_annos.pkl')
# annotations_med = load('./data/LD/LDall_mediapipe_annos.pkl')
# # print(annotations_ps[0])
# # exit(0)
# annotations_hand = []
# annos = dict()
# for iter in range(len(annotations_ps)):
#     print(annotations_ps[iter]['keypoint'].shape)
#     for i, vid_line in enumerate(lines):
#         annotations = '{}:{}_{}'.format(vid_line['vid_name'], vid_line['start_frame'],  vid_line['end_frame'])        
#         if annotations_ps[iter]['frame_dir'] in annotations:
            
#             annos['frame_dir'] = annotations
#             annos['label'] = vid_line['label']
#             annos['img_shape'] = annotations_ps[iter]['img_shape']
#             annos['total_frames'] = vid_line['end_frame'] - vid_line['start_frame'] +1
#             annos['num_hand_raw'] = 1
#             annos['keypoint'] = annotations_ps[iter]['keypoint'][:, vid_line['start_frame']-1:vid_line['end_frame'], :, :]
#             annos['keypoint_score'] = annotations_ps[iter]['keypoint_score'][:, vid_line['start_frame']-1:vid_line['end_frame'], :]
#             annotations_hand.append(annos.copy())
#             # print(annotations_ps[iter]['frame_dir'])
#             # print(annotations)
#             print(annotations_hand[i]['keypoint'].shape)
#             print(annotations_med[i]['keypoint'].shape)
#             print('-----'*10)
#     # if iter >=1:        
#     #     exit(0)
#     print('#####'*10)
# # 4.改annos路径
# dump(annotations_hand, './data/LD/LDall_mediapipe_snip_annos.pkl') 

# ###LDsnip  
# #读取
# LDhand = load('./data/LD/LDall_mediapipe_snip_annos.pkl')
# # print(LDhand[0])
# LD = [item for item in LDhand if item['label'] != 10]
# # print(LD)
# dump(LD, './data/LD/LD_No_mediapipe_snip_annos.pkl')
# #转化为net
# train = load('./data/LD/LD_No_train.json')
# test = load('./data/LD/LD_No_test.json')
# annotations_ps = load('./data/LD/LD_No_mediapipe_snip_annos.pkl')
# split = dict()
# split['train'] = [x['vid_name'] + ':' + str(x["start_frame"]) + '_' + str(x["end_frame"]) for x in train]
# split['test'] = [x['vid_name'] + ':' + str(x["start_frame"]) + '_' + str(x["end_frame"]) for x in test]
# dump(dict(split=split, annotations=annotations_ps), './data/LD/LD_No_mediapipe_snip.pkl')

# #转化为LDallnet
# train = load('./data/LD/LDall_train.json')
# test = load('./data/LD/LDall_test.json')
# annotations_ps = load('./data/LD/LDall_mediapipe_snip_annos.pkl')
# split = dict()
# split['train'] = [x['vid_name'] + ':' + str(x["start_frame"]) + '_' + str(x["end_frame"]) for x in train]
# split['test'] = [x['vid_name'] + ':' + str(x["start_frame"]) + '_' + str(x["end_frame"]) for x in test]
# dump(dict(split=split, annotations=annotations_ps), './data/LD/LDall_mediapipe_snip.pkl')


# #LDjson修改annos名字
# #3.改json路径和视频
# train = load('./data/LD/LDall_train.json')
# test = load('./data/LD/LDall_test.json')
# # line_train = [(' {}' + ' {}' + ' {}').format(x['vid_name'], x['label'],  x['start_frame'],  x['end_frame']) for x in train]
# # line_test = [(' {}' + ' {}' + ' {}').format(x['vid_name'], x['label'],  x['start_frame'],  x['end_frame']) for vid_line in test]
# lines = train + test
# annotations_ps = load('./data/LD/LDall_hand_annos.pkl')
# for i, vid_line in enumerate(lines):
#     annotations = '{}:{}_{}'.format(vid_line['vid_name'], vid_line['start_frame'],  vid_line['end_frame'])
#     if annotations_ps[i]['frame_dir'] in annotations:
#         annotations_ps[i]['frame_dir'] = annotations
#         print(annotations_ps[i]['frame_dir'])
#     else:
#         # 报错
#         raise ValueError(f"路径 '{annotations_ps[i]['frame_dir']}' 不存在于 json 中。")

# #4.改annos路径
# dump(annotations_ps, './data/LD/LDall_mediapipe_annos.pkl')   

# #读取json和pkl数据
# # #LD
# train = load('./data/LD/LDall_train.json')
# test = load('./data/LD/LDall_test.json')
# #LDhand
# annotations_ps = load('./data/LD/LDall_mediapipe_annos.pkl')
# # annotations_ipn = load('./data/ipn/ipnall_mediapipe_annos.pkl')
# # print(annotations_ps[0])
# # print(annotations_ipn[0])
# # exit(0)
# split = dict()
# split['train'] = [x['vid_name'] + ':' + str(x["start_frame"]) + '_' + str(x["end_frame"]) for x in train]
# split['test'] = [x['vid_name'] + ':' + str(x["start_frame"]) + '_' + str(x["end_frame"]) for x in test]
# dump(dict(split=split, annotations=annotations_ps), './data/LD/LDall_mediapipe.pkl')

# #读取json和pkl数据
# # #ipn
# # train = load('./data/ipn/ipn_train.json')
# # test = load('./data/ipn/ipn_test.json')
# # annotations = load('./data/ipn/ipnhand_annos.pkl')
# train = load('./data/ipn/ipnall_train.json')
# test = load('./data/ipn/ipnall_test.json')
# #ipnpose
# # annotations_pose = load('./data/ipn/ipnall_pose_annos.pkl')
# #ipnposehand
# annotations_ps = load('./data/ipn/ipnall_pose_hand_annos.pkl')
# split = dict()
# split['train'] = [x['vid_name'] + ':' + str(x["start_frame"]) + '_' + str(x["end_frame"]) for x in train]
# split['test'] = [x['vid_name'] + ':' + str(x["start_frame"]) + '_' + str(x["end_frame"]) for x in test]
# # dump(dict(split=split, annotations=annotations_pose), './data/ipn/ipnall_pose.pkl')
# dump(dict(split=split, annotations=annotations_ps), './data/ipn/ipnall_pose_hand.pkl')

# # paxis_hand
# train = load('./data/paxis/dataset/paxis_train.json')
# test = load('./data/paxis/dataset/paxis_test.json')
# annotations = load('./data/paxishand/paxishand_annos.pkl')
# #paxis
# train = load('./data/paxis/dataset/paxis_train.json')
# test = load('./data/paxis/dataset/paxis_test.json')
# annotations = load('./data/paxis/dataset/paxis_annos.pkl')

# split = dict()
# split['train'] = [x['vid_name'] + ':' + str(x["start_frame"]) + '_' + str(x["end_frame"]) for x in train]
# split['test'] = [x['vid_name'] + ':' + str(x["start_frame"]) + '_' + str(x["end_frame"]) for x in test]
# print(annotations[100]['keypoint'].shape)
# print(annotations)
# exit(0)
#ipn
#2.存放
##ipn
# dump(dict(split=split, annotations=annotations), './data/ipn/ipn_mediapipe.pkl')
#ipnpose
# dump(dict(split=split, annotations=annotations), './data/ipn/ipn_pose.pkl')
#3.展示
# ipn = load('./data/ipn/ipn_mediapipe.pkl')
# print(ipn)