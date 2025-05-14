from mmcv import load, dump


#读取
#显示表面x标签1开始
ipnhand15 = load('./data/ipn/ipn_No_mediapipe_track15_snip.pkl')
ipnhand = load('./data/ipn/ipn_No_mediapipe_track15_s0k0_snip.pkl')
for i in range(len(ipnhand15['annotations'])):
    print(ipnhand15['annotations'][i])
    print(ipnhand['annotations'][i])
    print('==='*20)

# #读取
# #显示表面x标签1开始
# ipnhand15 = load('./data/ipn/ipn_No_mediapipe_track15_snip.pkl')
# # ipnhand = load('./data/ipn/ipn_No_mediapipe_track_snip.pkl')
# for i in range(len(ipnhand15['annotations'])):

#     # print(ipnhand15['annotations'][i])
#     ipnhand15['annotations'][i]['keypoint_score'][ipnhand15['annotations'][i]['keypoint_score'] == -1] = 0
#     ipnhand15['annotations'][i]['keypoint'][ipnhand15['annotations'][i]['keypoint'] == -1] = 0
#     # print(ipnhand15['annotations'][i])
#     # print(ipnhand['annotations'][i])
#     # print('==='*20)

# dump(ipnhand15, './data/ipn/ipn_No_mediapipe_track15_s0k0_snip.pkl')

# print(ipnhandx['annotations'][0])
#替换mediapipe标签
# ipn = load('./data/ipn/ipnall_mediapipe.pkl')
# for ipn_iter in ipn['annotations']:
#     ipn_iter['label'] = ipn_iter['label'] - 1
    # print(ipn_iter['label'])
# print(ipn)
# dump(ipn, './data/ipn/ipnallx_mediapipe.pkl')

#看pose、hand格式是否相同
# ipnhand = load('./data/ipn/ipnhand_annos.pkl')
# ipnpose = load('./data/ipn/ipn_pose_annos.pkl')
# ipnposehand = load('./data/ipn/ipn_pose_hand_annos.pkl')
# print(ipnhand[0])
# print(ipnpose[0])
# print(ipnposehand[0])
#看上你paxis格式
# paxishand = load('./data/paxishand/paxis_mediapipe.pkl')
# print(paxishand)

#posec3dRGB格式会出问题，替换但为成功
# ntu = load('/home/user/github/pyskl/data/nturgbd/ntu60_hrnet.pkl')
# for i, ntu_iter in enumerate(ntu['annotations']):
#     # ntu_iter['frame_dir'] = ntu_iter['frame_dir'] + '_rgb'
#     if  ntu_iter['frame_dir'] == 'S017C001P007R001A030':
#         print(ntu_iter['frame_dir']+ ':' + '1')
#         del ntu['annotations'][i]
# print(ntu['split'].keys())
# exit(0)
# for ntu_train in ntu['split']['xsub_train']:
#     if  ntu_train == 'S017C001P007R001A030':
#         print(ntu_train+ ':' + 'train')
# for ntu_test in ntu['split']['xsub_val']:
#     if  ntu_test == 'S017C001P007R001A030':
#         print(ntu_test+ ':' + 'test')
# dump(ntu, './data/nturgbd/ntu60_d_hrnet.pkl')    
# ntud = load('/home/user/github/pyskl/data/nturgbd/ntu60_d_hrnet.pkl')
# # print(a['annotations'][0])
# for ntud_iter in ntud['annotations']:
#     # ntu_iter['frame_dir'] = ntu_iter['frame_dir'] + '_rgb'
#     if  ntud_iter['frame_dir'] == 'S017C001P007R001A030':
#         print(ntud_iter['frame_dir']+ ':' + '2')
       