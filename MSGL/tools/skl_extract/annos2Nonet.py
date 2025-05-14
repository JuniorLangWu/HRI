import json
from mmcv import load, dump

###ipn
# # 读取.json 文件
# with open('./data/ipn/ipnall_train.json', 'r') as file:
#     ipndata = json.load(file)
# # 去除 "label": 0 的字典
# filtered_data = [item for item in ipndata if item['label'] != 0]
# # 将过滤后的数据写回 .json 文件
# with open('./data/ipn/ipn_No_train.json', 'w') as file:
#     json.dump(filtered_data, file, indent=4)
# # 读取.json 文件
# with open('./data/ipn/ipnall_test.json', 'r') as file:
#     ipndata = json.load(file)
# # 去除 "label": 0 的字典
# filtered_data = [item for item in ipndata if item['label'] != 0]
# # 将过滤后的数据写回 .json 文件
# with open('./data/ipn/ipn_No_test.json', 'w') as file:
#     json.dump(filtered_data, file, indent=4)

# #读取
# ipnhand = load('./data/ipn/ipnall_mediapipe_annos.pkl')
# # print(ipnhand[0])
# ipn = [item for item in ipnhand if item['label'] != 0]
# #替换mediapipe标签
# for ipn_iter in ipn:
#     ipn_iter['label'] = ipn_iter['label'] - 1
#     # print(ipn_iter['label'])
# # print(ipn)
# dump(ipn, './data/ipn/ipn_No_mediapipe_annos.pkl')

# #转化为net
# train = load('./data/ipn/ipn_No_train.json')
# test = load('./data/ipn/ipn_No_test.json')
# annotations_ps = load('./data/ipn/ipn_No_mediapipe_annos.pkl')
# split = dict()
# split['train'] = [x['vid_name'] + ':' + str(x["start_frame"]) + '_' + str(x["end_frame"]) for x in train]
# split['test'] = [x['vid_name'] + ':' + str(x["start_frame"]) + '_' + str(x["end_frame"]) for x in test]
# dump(dict(split=split, annotations=annotations_ps), './data/ipn/ipn_No_mediapipe.pkl')

# ###LD
# # 读取.json 文件
# with open('./data/LD/LDall_train.json', 'r') as file:
#     LDdata = json.load(file)
# # 去除 "label": 10 的字典
# filtered_data = [item for item in LDdata if item['label'] != 10]
# # 将过滤后的数据写回 .json 文件
# with open('./data/LD/LD_No_train.json', 'w') as file:
#     json.dump(filtered_data, file, indent=4)
# # 读取.json 文件
# with open('./data/LD/LDall_test.json', 'r') as file:
#     LDdata = json.load(file)
# # 去除 "label": 10 的字典
# filtered_data = [item for item in LDdata if item['label'] != 10]
# # 将过滤后的数据写回 .json 文件
# with open('./data/LD/LD_No_test.json', 'w') as file:
#     json.dump(filtered_data, file, indent=4)
    
# #读取
# LDhand = load('./data/LD/LDall_mediapipe_annos.pkl')
# # print(LDhand[0])
# LD = [item for item in LDhand if item['label'] != 10]
# # print(LD)
# dump(LD, './data/LD/LD_No_mediapipe_annos.pkl')

# #转化为net
# train = load('./data/LD/LD_No_train.json')
# test = load('./data/LD/LD_No_test.json')
# annotations_ps = load('./data/LD/LD_No_mediapipe_annos.pkl')
# split = dict()
# split['train'] = [x['vid_name'] + ':' + str(x["start_frame"]) + '_' + str(x["end_frame"]) for x in train]
# split['test'] = [x['vid_name'] + ':' + str(x["start_frame"]) + '_' + str(x["end_frame"]) for x in test]
# dump(dict(split=split, annotations=annotations_ps), './data/LD/LD_No_mediapipe.pkl')

###LDsnip  
#读取
LDhand = load('./data/LD/LDall_mediapipe_snip_annos.pkl')
# print(LDhand[0])
LD = [item for item in LDhand if item['label'] != 10]
# print(LD)
dump(LD, './data/LD/LD_No_mediapipe_snip_annos.pkl')

#转化为net
train = load('./data/LD/LD_No_train.json')
test = load('./data/LD/LD_No_test.json')
annotations_ps = load('./data/LD/LD_No_mediapipe_snip_annos.pkl')
split = dict()
split['train'] = [x['vid_name'] + ':' + str(x["start_frame"]) + '_' + str(x["end_frame"]) for x in train]
split['test'] = [x['vid_name'] + ':' + str(x["start_frame"]) + '_' + str(x["end_frame"]) for x in test]
dump(dict(split=split, annotations=annotations_ps), './data/LD/LD_No_mediapipe_snip.pkl')

LDhand = load('./data/LD/LD_No_mediapipe.pkl')
LDsnip = load('./data/LD/LD_No_mediapipe_snip.pkl')
print(len(LDhand['split']))
print(len(LDhand['split']))