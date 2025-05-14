import matplotlib.pyplot as plt
import glob
import os
from mmcv import load
from pyskl.smp import *

# work_dir = './work_dirs/rgbposec3d/pose_only_ipnall_mediapipe_times5_batch16_epochs50'
# work_dir = './work_dirs/stgcn++_hand/stgcn++_ipnall_mediapipe_j_times20_batch64_epochs50'
# work_dir = './work_dirs/stgcn++_hand/stgcn++_ipnall_rtmpose_mediapipe_j_times20_batch64_epochs50'
# work_dir = './work_dirs/stgcn++/stgcn++_ipnall_rtmpose_j_times20_batch64_epochs50'
# work_dir = './work_dirs/rgbposec3d/pose_only_ipnall_mediapipe_times10_batch32_epochs120'
# work_dir = './work_dirs/stgcn++_hand/stgcn++_ipnall_mediapipe_j_dropout=0.5_times20_batch64_epochs120'
# work_dir = './work_dirs/stgcn++_hand/stgcn++_ipnall_rtmpose_mediapipe_j_dropout=0.5_times20_batch64_epochs120'
work_dir = './work_dirs/stgcn++_hand/stgcn++_ipnall_mediapipe_j_clsdropout_times20_batch64_epochs120'
json_files = glob.glob(os.path.join(work_dir, '*.log.json'))
train_top1_acc = []
train_top5_acc = []
train_loss = []
val_top1_acc = []
val_top5_acc =[]
val_mean_class_accuracy = []
val_mean_class_accuracy = []
with open(json_files[0], 'r') as f:
    try:
        for line in f:
            line = line.strip()  # 去除首尾的空白字符
            if line:  # 忽略空行
                log_entry = json.loads(line)
                # 提取 mode 为 "train" 的条目的 top1_acc 值
                if log_entry.get('mode') == 'train':
                    train_top1_acc.append(log_entry['top1_acc'])
                    train_top5_acc.append(log_entry['top5_acc'])
                    train_loss.append(log_entry['loss'])
                if log_entry.get('mode') == 'val':
                    val_top1_acc.append(log_entry['top1_acc'])
                    val_top5_acc.append(log_entry['top5_acc'])
                    val_mean_class_accuracy.append(log_entry['mean_class_accuracy'])
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from file {json_files}: {e}")


# 创建第一个图形对象
work_dir = './' + work_dir.split('/')[1] + '/figure/' + work_dir.split('/')[3]
plt.figure()
plt.plot(range(len(train_top1_acc)), train_top1_acc, label='train_top1_acc')
plt.legend()
plt.savefig('{}_{}.png'.format(work_dir, 'train_top1_acc'))
plt.figure()
plt.plot(range(len(train_loss)), train_loss, label='train_loss')
plt.legend()
plt.savefig('{}_{}.png'.format(work_dir, 'train_loss'))
# 创建第二个图形对象
plt.figure()
plt.plot(range(len(val_top1_acc)), val_top1_acc, label='val_top1_acc')
plt.plot(range(len(val_mean_class_accuracy)), val_mean_class_accuracy, label='val_mean_class_accuracy')
plt.legend()
plt.savefig('{}_{}.png'.format(work_dir, 'val'))
# 显示图像
# plt.show()