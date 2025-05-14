#生成.json标签
import os
import pandas as pd
import json

#test:json2list
from mmcv import load, dump
from pyskl.smp import *
#3.改json路径和视频
test = load('/home/user/github/mypyskl/data/ipn/ipnall_test.json')
test_tmpl = '/home/user/Videos/dataset/IPN_Hand/videos/{}'
line_test = [(test_tmpl + ' {}' + ' {}' + ' {}').format(x['vid_name'], x['label'],  x['start_frame'],  x['end_frame']) for x in test]
lines = line_test
#4.改list路径
mwlines(lines, '/home/user/github/mypyskl/data/ipn/ipn_all.txt')
#3.改json路径和视频
test = load('/home/user/github/mypyskl/data/LD/LDall_test.json')
test_tmpl = '/home/user/Videos/dataset/LD-ConGR/videos/{}'
line_test = [(test_tmpl + ' {}' + ' {}' + ' {}').format(x['vid_name'], x['label'],  x['start_frame'],  x['end_frame']) for x in test]
lines = line_test
#4.改list路径
mwlines(lines, '/home/user/github/mypyskl/data/LD/LD_all.txt')

###IPN
# #1.改标签路径
# # #标签文件路径
# # test_dir = r'/home/user/Videos/dataset/IPN_Hand/annotations/Annot_TestList.txt'
# # train_dir = r'/home/user/Videos/dataset/IPN_Hand/annotations/Annot_TrainList.txt'
# #测试数据集效果
# test_dir = r'/home/user/Videos/dataset/IPN_Hand/annotations/TestList.txt'
# train_dir = r'/home/user/Videos/dataset/IPN_Hand/annotations/TrainList.txt'
# # label = pd.read_csv(test_dir, header=None)
# # vid_name = label.iloc[:, 0]
# # print(vid_name)
# # exit(0)
# # classIdx_dir = r'/home/user/Videos/dataset/IPN_Hand/annotations/classIdx.txt'
# # classIdx = pd.read_csv(classIdx_dir)
# # 将 id 和标签作为字典输出
# # id_to_class = dict(zip(classIdx['id'], classIdx['label']))
# # class_to_id = dict(zip(classIdx['label'], classIdx['id']))
# # 打印结果
# # print('标签转id：', class_to_id)
# def writeJson(labelpath, jsonpath):
#     label = pd.read_csv(labelpath, header=None)

#     outpot_list = []    
#     for index, row in label.iterrows():
#         video,label,id,t_start,t_end,frames= row
#         labeldict = {}
#         labeldict['vid_name'] = video
#         labeldict['label'] = id-1
#         labeldict['start_frame'] = t_start
#         labeldict['end_frame'] = t_end
#         outpot_list.append(labeldict.copy())
#     # 将列表转换为JSON字符串，并在每对大括号之间插入换行符
#     json_str = '[\n' + ',\n'.join(json.dumps(d, indent=4) for d in outpot_list) + '\n]'
#     with open(jsonpath, 'w') as outfile:
#         outfile.write(json_str)
# #2.改json路径
# # data_dir = r'/home/user/Videos/dataset/IPN_Hand/videos'
# train_json_dir = r'./data/ipn/ipn_train.json'
# os.makedirs(os.path.dirname(train_json_dir), exist_ok=True)
# test_json_dir = r'./data/ipn/ipn_test.json'
# os.makedirs(os.path.dirname(test_json_dir), exist_ok=True)

# writeJson(test_dir, test_json_dir )
# writeJson(train_dir, train_json_dir )

# #json2list
# from mmcv import load, dump
# from pyskl.smp import *
# #3.改json路径和视频
# train = load('/home/user/github/pyskl/data/ipn/ipn_train.json')
# test = load('/home/user/github/pyskl/data/ipn/ipn_test.json')
# train_tmpl = '/home/user/Videos/dataset/IPN_Hand/videos/{}'
# test_tmpl = '/home/user/Videos/dataset/IPN_Hand/videos/{}'
# line_train = [(train_tmpl + ' {}' + ' {}' + ' {}').format(x['vid_name'], x['label'],  x['start_frame'],  x['end_frame']) for x in train]
# line_test = [(test_tmpl + ' {}' + ' {}' + ' {}').format(x['vid_name'], x['label'],  x['start_frame'],  x['end_frame']) for x in test]
# lines = line_train + line_test
# #4.改list路径
# mwlines(lines, '/home/user/github/pyskl/data/ipn/ipn.list')

###LD
# #1.改标签路径
# #标签文件路径
# test_dir = r'/home/user/Videos/dataset/LD-ConGR/vallistall.txt'
# train_dir = r'/home/user/Videos/dataset/LD-ConGR/trainlistall.txt'
# def writeJson(labelpath, jsonpath):
#     label = pd.read_csv(labelpath, header=None)

#     outpot_list = []    
#     for index, row in label.iterrows():
#         split_row = row[0].split()
#         # print(split_row[0])
#         # exit(0)
#         video, id, t_start, t_end = split_row
#         video = video.replace('./frames/', '')
#         video = video.replace('_all', '')
#         labeldict = {}
#         labeldict['vid_name'] = video
#         labeldict['label'] = int(id)-1
#         labeldict['start_frame'] = int(t_start)
#         labeldict['end_frame'] = int(t_end)
#         outpot_list.append(labeldict.copy())
#     # 将列表转换为JSON字符串，并在每对大括号之间插入换行符
#     json_str = '[\n' + ',\n'.join(json.dumps(d, indent=4) for d in outpot_list) + '\n]'
#     with open(jsonpath, 'w') as outfile:
#         outfile.write(json_str)
# #2.改json路径
# train_json_dir = r'./data/LD/LDall_train.json'
# os.makedirs(os.path.dirname(train_json_dir), exist_ok=True)
# test_json_dir = r'./data/LD/LDall_test.json'
# os.makedirs(os.path.dirname(test_json_dir), exist_ok=True)

# writeJson(test_dir, test_json_dir )
# writeJson(train_dir, train_json_dir )

# #json2list
# from mmcv import load, dump
# from pyskl.smp import *
# #3.改json路径和视频
# train = load('./data/LD/LDall_train.json')
# test = load('./data/LD/LDall_test.json')
# train_tmpl = '/home/user/Videos/dataset/LD-ConGR/videos/{}'
# test_tmpl = '/home/user/Videos/dataset/LD-ConGR/videos/{}'
# line_train = [(train_tmpl + ' {}' + ' {}' + ' {}').format(x['vid_name'], x['label'],  x['start_frame'],  x['end_frame']) for x in train]
# line_test = [(test_tmpl + ' {}' + ' {}' + ' {}').format(x['vid_name'], x['label'],  x['start_frame'],  x['end_frame']) for x in test]
# lines = line_train + line_test
# #4.改list路径
# mwlines(lines, './data/LD/LDall.list')
