import json
import os
import sys
def convert_json_to_string(json_file_path):
    # 读取JSON文件
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    # 创建一个空字典来存储转换后的数据
    converted_dict = {}
    # 遍历每个记录
    for record in data:
        vid_name = record['vid_name']
        label = record['label']
        start_frame = record['start_frame']
        end_frame = record['end_frame']        
        
        # 如果字典中还没有这个视频名称，初始化一个空字符串
        if vid_name not in converted_dict:
            converted_dict[vid_name] = []
        
        # 将标签和帧差添加到相应的字符串中
        converted_dict[vid_name].append(f"{start_frame} {end_frame} {label}")
    
    # 将字典转换为所需的字符串格式
    result_strings = []
    for vid_name, frames in converted_dict.items():
        result_strings.append(f"{vid_name},{','.join(frames)}")
    # print(result_strings)
    # exit(0)
    # 返回或打印转换结果
    return '\n'.join(result_strings)

# 假设你的JSON文件路径为 "input.json"
json_file_path = '/home/user/github/mypyskl/data/ipn/ipn_No_test.json'

# 转换并输出结果
output_string = convert_json_to_string(json_file_path)
# print(output_string)
results_file = open(os.path.join('/home/user/github/mypyskl/data/ipn/online/', 'ipn_test_labels'+'.txt'), "w")
results_file.write(output_string)
sys.stdout.flush()
results_file.close()