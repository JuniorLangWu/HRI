import re
import os
import sys
dataset = 'hcigesture'

def process_line(line):
    # 解析每行的数据
    key, data = line.split(',', 1)
    pairs = re.findall(r'(\d+) (\d+) (\d+)', data)
    
    # 创建嵌套字典结构
    nested_dict = {}
    for pair in pairs:
        sub_key, sub_value1, sub_value2 = map(int, pair)
        if sub_key not in nested_dict:
            nested_dict[sub_key] = []
        nested_dict[sub_key].append((sub_value1, sub_value2))
    # if dataset == 'ipn':
    #     dir_iter = list(nested_dict.keys())
    #     for iter in range(len(dir_iter)-1):
    #         # print(nested_dict[dir_iter[iter]][-1])
    #         if (nested_dict[dir_iter[iter+1]][-1][1]==0):
    #             nested_dict[dir_iter[iter]][-1] = (dir_iter[iter+1]-1,nested_dict[dir_iter[iter]][-1][1])
    #     for iter in range(len(dir_iter)-1):
    #         iter = iter+1       
    #         if (nested_dict[dir_iter[iter-1]][-1][1]==0):
    #             dir_iter_ori = dir_iter[iter]
    #             dir_iter[iter] = nested_dict[dir_iter[iter-1]][-1][0]+1-17
    #             # print(dir_iter[iter])
    #             # 确保旧的键存在于字典中
    #             if dir_iter_ori in nested_dict:
    #                 # 将值赋给新的键
    #                 nested_dict[dir_iter[iter]] = nested_dict.pop(dir_iter_ori)
    #             #字典重新排序
    # nested_dict = dict(sorted(nested_dict.items()))
    if dataset == 'hcigesture':
        pass
    # dir_iter = list(nested_dict.keys())
    # for sub_key in dir_iter:
    #     if dataset == 'ipn':
    #         if nested_dict[sub_key][-1][1] == 0:
    #             iter_last=dir_iter.index(sub_key)       
    #             nested_dict[dir_iter[iter_last-1]][-1] = (nested_dict[dir_iter[iter_last-1]][-1][0]+26,nested_dict[dir_iter[iter_last-1]][-1][1])
    #字典重新排序
    nested_dict = dict(sorted(nested_dict.items()))
    # 检查并去除最后一个值为0的元素
    for sub_key in list(nested_dict.keys()):
        if dataset == 'ipn':
            if nested_dict[sub_key][-1][1] == 0:
                nested_dict[sub_key].pop()
        if dataset == 'hcigesture':
            if nested_dict[sub_key][-1][1] == 10:
                nested_dict[sub_key].pop()
        # 如果该子键的值为空，删除该子键
        if not nested_dict[sub_key]:
            del nested_dict[sub_key]
    return key, nested_dict
# 测试文件路径
if dataset == 'ipn':
    file_set = os.path.join('/home/user/Videos/dataset/IPN_Hand/', 'Video_TestList.txt')
    test_paths = []
    buf = 0
    with open(file_set,'rb') as f:
        for line in f:
            vid_name = line.decode().split('\t')[0]
            test_paths.append(os.path.join('/home/user/Videos/dataset/IPN_Hand/', 'frames', vid_name))
    file_path = '/home/user/github/mypyskl/work_dirs/online/two_stage_online_results_ipn_pose0.8_size34_68_None_det45_cls4_ac16_final0.9_0pre0.9/ipn_all_results.txt'
    results_file = open(os.path.join('/home/user/github/mypyskl/work_dirs/online/two_stage_online_results_ipn_pose0.8_size34_68_None_det45_cls4_ac16_final0.9_0pre0.9/', 'ipn_all_None_results'+'.txt'), "w")
elif dataset == 'hcigesture':
    test_paths = []
    with open(os.path.join('/home/user/Videos/dataset/LD-ConGR/', 'test_labels.txt')) as test_file:
        for line in test_file:
            video_name = line.split(',')[0]
            #去除回车符号
            video_name = video_name.replace('\n', '') 
            test_paths.append(os.path.join('/home/user/Videos/dataset/LD-ConGR/', 'frames', video_name))
    file_path = '/home/user/github/mypyskl/work_dirs/online/0_None_online/hci/two_stage_online_results_hci_pose0.8_size7_14_None_det10_cls1_ac4_final0.9_0pre0.9/hci_all_results.txt'
    results_file = open(os.path.join('/home/user/github/mypyskl/work_dirs/online/0_None_online/hci/two_stage_online_results_hci_pose0.8_size7_14_None_det10_cls1_ac4_final0.9_0pre0.9/', 'hci_all_None_results'+'.txt'), "w")

result_dict = {}
with open(file_path, 'r') as f:
    for line in f:
        key, nested_dict = process_line(line.strip())
        result_dict[key] = nested_dict
        # print(f"Key: {key}, Processed Data: {nested_dict}")
        # exit(0)
for path in test_paths:
    video_name = path.split('frames'+'/')[1]
    result_line = video_name
    i = 0
    for s in result_dict[video_name]:

        e, label  = result_dict[video_name][s][0]
        if dataset == 'ipn':
            s = s+14
            e = e-24
            if i >= 1:
                if int(end) > int(s):
                    end = s-1
                result_line += ','+' '.join([str(start),str(end),str(label_pre)])

            start = s
            end = e
            label_pre = label
            i += 1
            if i >= len(result_dict[video_name]):
                result_line += ','+' '.join([str(start),str(end),str(label_pre)])
        if dataset == 'hcigesture':
            s = s+2
            e = e-2
            if i >= 1:
                if int(end) > int(s):
                    end = s-1
                result_line += ','+' '.join([str(start),str(end),str(label_pre)])

            start = s
            end = e
            label_pre = label
            i += 1
            if i >= len(result_dict[video_name]):
                result_line += ','+' '.join([str(start),str(end),str(label_pre)])
    results_file.write(result_line+'\n')
    sys.stdout.flush()
results_file.close()