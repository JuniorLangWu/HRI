import cv2
import os
import sys
import pandas as pd
import csv
import torch

from tools.online_tools.utils import Logger, AverageMeter, LevenshteinDistance, Queue

import numpy as np

import argparse
import time
from mmcv import Config

from pyskl.apis import init_recognizer
from pyskl.datasets.pipelines import Compose
from online_tools.hand_pose import posehand_inference_online
import json

def parse_args():
    parser = argparse.ArgumentParser(
        description='pyskl test (and eval) a model')
    parser.add_argument('--config', help='test config file path')
    parser.set_defaults(config='{}/online_tools/online_configs/stfgcn_ipn_online_two.py'.format(os.path.dirname(os.path.abspath(__file__))))
    parser.add_argument('--n_classes_clf', default=13, type=int, help='Number of classes (ipn: 13, LD: 10, )')
    parser.add_argument('--clf_queue_size', default=32, type=int, help='Classifier queue size')
    parser.add_argument('--sample_duration_clf', default=64, type=int, help='Temporal duration of inputs')
    parser.add_argument('--modality', default='RGB', type=str, help='Modality of generated model. RGB,')
    parser.add_argument('--stride_len', default=1, type=int, help='Stride Lenght of video loader window')    
    parser.add_argument('--batch_size', default=1, type=int, help='Batch Size')
    parser.add_argument('--n_threads', default=4, type=int, help='Number of threads for multi-thread loading')
    parser.add_argument('--clf_strategy', default='ma', type=str, help='Classifier filter (raw | median | ma | ewma)')
    parser.add_argument('--clf_threshold_pre', default=0.9, type=float, help='Cumulative sum threshold to predict at the end')
    parser.add_argument('--clf_threshold_final', default=0.9, type=float, help='Cumulative sum threshold to predict at the end')
    parser.add_argument('--det_counter', default=64, type=float, help='Number of consequtive detection')
    parser.add_argument('--cls_counter', default=4, type=float, help='Number of consequtive detection')
    parser.add_argument('--active_threshold', default=16, type=int, help='confirm a gesture if the total number of active frames is greater than the threshold')
    parser.add_argument('--set_work', action='store_true', help='If true, ')
    parser.set_defaults(set_work=False)
    args ,unknown = parser.parse_known_args()    
    return args

def create_fake_anno(keypoint):

    # keypoint = np.array(results[::-1], dtype=np.float32)[None]
    total_frames = keypoint.shape[1]
    return dict(
        keypoint=keypoint,
        total_frames=total_frames,
        frame_dir='NA',
        label=0,
        start_index=0,
        modality='Pose',
        test_mode=True)

def create_fake_anno_empty():
    return dict(
        keypoint=np.zeros([1, opt.sample_duration_clf , 21, 3], dtype=np.float32),
        total_frames=opt.sample_duration_clf,
        frame_dir='NA',
        label=0,
        start_index=0,
        modality='Pose',
        test_mode=True)


file_path = os.path.dirname(os.path.abspath(__file__)) + '/'
#开始
args = parse_args()
opt = args
#参数.py加载
cfg = Config.fromfile(args.config)
#设置online相机图片大小
cfg.test_pipeline[0].img_shape = (480,640)

if cfg.get('annotation_path', None) is not None:
    opt.annotation_path = file_path + cfg.annotation_path

label_csv_path = os.path.join(opt.annotation_path.rsplit(os.sep, 1)[0], 'classIndAll.txt')
labels_data = pd.read_csv(label_csv_path, delimiter=' ', header=None)

if cfg.get('ipn_pth', None) is not None:
    opt.ipn_pth = file_path + cfg.ipn_pth

#初始化模型
recognizer = init_recognizer(args.config, opt.ipn_pth, device='cuda:0')
recognizer.eval()

#初始化
device = next(recognizer.parameters()).device
test_pipeline = Compose(cfg.test_pipeline)
fake_anno = create_fake_anno_empty()
sample = test_pipeline(fake_anno)['keypoint'][None].to(device)
prediction = recognizer(sample, return_loss=False)[0]

if cfg.get('dataset', None) is not None:
    opt.dataset = cfg.dataset
if cfg.get('video_path', None) is not None:
    opt.video_path = cfg.video_path
if cfg.get('crop', None) is not None:
    opt.crop = cfg.crop
if opt.dataset == 'ipn':
    opt.n_classes_clf = 13
    opt.clf_queue_size = 38
    opt.sample_duration_clf = 76
    opt.clf_threshold_pre = 0.9
    opt.clf_threshold_final = 0.9

    opt.det_counter = 45
    opt.cls_counter = 4
    opt.active_pre = 0
    opt.active_threshold = 16
elif opt.dataset == 'hcigesture':
    opt.n_classes_clf = 10
    # opt.clf_queue_size = 4
    # opt.sample_duration_clf = 8
    opt.clf_threshold_pre = 0.9
    opt.clf_threshold_final = 0.9

    opt.det_counter = 10
    opt.cls_counter = 1
    opt.active_pre = 0
    opt.active_threshold = 4


#开始
print('Start Evaluation')
# exit(0)
if cfg.get('result_path', None) is not None:
    opt.result_path = file_path + cfg.result_path
if opt.set_work:
    find_j = opt.result_path.split('_')
    find_j[7] = 'size' + str(opt.clf_queue_size)
    find_j[8] = str(opt.sample_duration_clf)
    opt.result_path = '_'.join(find_j)
if not os.path.exists(opt.result_path):
    os.makedirs(opt.result_path)
#建立两个文件
if opt.dataset == 'ipn':
    # results_file = open(os.path.join(opt.result_path, 'ipn_all_results'+'.txt'), "w")
    # results_No_file = open(os.path.join(opt.result_path, 'ipn_None_results'+'.txt'), "w")
    ori_prediction_file = open(os.path.join(opt.result_path, 'ipn_all_ori_prediction'+'.txt'), "w")
    # levenshtein_file = open(os.path.join(opt.result_path, 'ipn_all_levenshtein'+'.txt'), "w")
elif opt.dataset == 'hcigesture':
    results_file = open(os.path.join(opt.result_path, 'hci_all_results'+'.txt'), "w")
    results_No_file = open(os.path.join(opt.result_path, 'hci_None_results'+'.txt'), "w")
    ori_prediction_file = open(os.path.join(opt.result_path, 'hci_all_ori_prediction'+'.txt'), "w")
    levenshtein_file = open(os.path.join(opt.result_path, 'hci_all_levenshtein'+'.txt'), "w")

levenshtein_accuracies = AverageMeter()
results = {}
# results_No = {}

iter=0
pred_classes = []
active = False
active_index = 0
passive_count = 0
activate_count =0
last_activate_frame = 0
prev_gesture = -1  # None
start_frame_index = 0
prediction_clf = 0
clf_selected_queue = np.zeros(opt.n_classes_clf, )
outputs_clf = np.zeros(opt.n_classes_clf ,)
myqueue_clf = Queue(opt.clf_queue_size, n_classes=opt.n_classes_clf)
keypoint = np.zeros((1, opt.sample_duration_clf, 21, 3), dtype=np.float32)

if cfg.get('output_file_end', None) is not None:
    output_file_end = cfg.output_file_end
# 定义输出视频的名称、编码格式、帧率和分辨率
output_filei = '/home/wjl/Videos/robotonline/output_videoi{}.avi'.format(output_file_end)
output_files = '/home/wjl/Videos/robotonline/output_videos{}.avi'.format(output_file_end)
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 视频编码格式（XVID是常用格式之一）
fps = 30.0  # 帧率，表示每秒显示多少帧
frame_size = (640, 480)  # 视频的分辨率（宽 x 高）
# 创建 VideoWriter 对象
outi = cv2.VideoWriter(output_filei, fourcc, fps, frame_size)
outs = cv2.VideoWriter(output_files, fourcc, fps, frame_size)
def gesture(inputs, results_No):
    global levenshtein_accuracies
    global results
    # global results_No
    video_name = 'online'
    predict_gesture = 0
    global iter
    results[video_name] = {} # start: (end, label)
    # results_No[video_name] = {} # start: (end, label)
    global pred_classes
    global active
    global active_index
    global passive_count
    global activate_count
    global last_activate_frame
    global prev_gesture  # None
    global start_frame_index
    global prediction_clf
    global clf_selected_queue
    global outputs_clf
    global myqueue_clf
    global keypoint
    try:
        window_tail_frame = iter * opt.stride_len 
        start_all = time.time()
        outi.write(inputs)
        ori_frame = inputs
        HEIGHT, WIDTH, CHANNEL = inputs.shape
        with torch.no_grad():
            #必须是整数
            np_imgs = inputs
            s =time.time()
            if opt.dataset == 'ipn':
                keypoint_end = posehand_inference_online(np_imgs)
            elif opt.dataset == 'hcigesture':
                keypoint_end = posehand_inference_online(np_imgs)                          
            keypoint[0,0:-1,:,:] = keypoint[0,1:,:,:]
            keypoint[0,-1,:,:] = keypoint_end
            hand_score = keypoint[0, :, 0, 2]
            hand_det = keypoint[0, -1, 0, -1]
            e =time.time()
            # print("运行时间:%.2f毫秒"%((e-s)*1000))
            # print('---'*5)
            # print(keypoint.shape)
            # print(keypoint[0,-1,:,0:2])

            #cv2写可视化
            #绘制hands骨骼点
            for point in keypoint[0,-1,:,0:2]:
                x, y = map(int, point)
                # 定义点的颜色，这里使用绿色 (0, 225, 0)
                color = (0, 225, 0)
                # 绘制点，圆心坐标为 (x, y)，半径为 2，颜色为 color，线宽为 -1（表示实心圆）
                cv2.circle(ori_frame, (x, y), 2, color, -1)
            # ori_frame = cv2.cvtColor(ori_frame, cv2.COLOR_BGR2RGB)
            cv2.imshow('Pose_and_hands', ori_frame)
            outs.write(inputs)
            cv2.waitKey(1)
   

            # continue
            #分类器开始
            # print(hand_det)
            iter=iter+1
            if hand_det >= 0.8:
                #骨骼点转换
                start = time.time()
                sample = create_fake_anno(keypoint)
                sample = test_pipeline(sample)['keypoint'][None].to(device)
                #动作识别
                prediction = recognizer(sample, return_loss=False)[0]
                outputs_clf = prediction
                end = time.time()
                # print("运行时间:%.2f毫秒"%((end-start)*1000))
                # print('---'*10)

                myqueue_clf.enqueue(outputs_clf.tolist())
                passive_count = 0
                if opt.clf_strategy == 'raw':
                    clf_selected_queue = outputs_clf
                elif opt.clf_strategy == 'median':
                    clf_selected_queue = myqueue_clf.median
                elif opt.clf_strategy == 'ma':
                    clf_selected_queue = myqueue_clf.ma
                elif opt.clf_strategy == 'ewma':
                    clf_selected_queue = myqueue_clf.ewma

                prediction_clf = np.argmax(clf_selected_queue)
                if opt.dataset == 'ipn':
                    predict_gesture = prediction_clf+1#labels_data[1][prediction_clf]
                elif opt.dataset == 'hcigesture':
                    predict_gesture = prediction_clf

                ori_prediction_file.write('%d %d %f\n' % (window_tail_frame, predict_gesture, clf_selected_queue[prediction_clf]))
            else:
                # Push the probabilities to queue
                myqueue_clf.enqueue(np.zeros(opt.n_classes_clf ,).tolist())
                if opt.clf_strategy == 'raw':
                    clf_selected_queue = outputs_clf
                elif opt.clf_strategy == 'median':
                    clf_selected_queue = myqueue_clf.median
                elif opt.clf_strategy == 'ma':
                    clf_selected_queue = myqueue_clf.ma
                elif opt.clf_strategy == 'ewma':
                    clf_selected_queue = myqueue_clf.ewma
                passive_count += 1
                if opt.dataset == 'ipn':
                    ori_prediction_file.write('%d %d %f\n' % (window_tail_frame, 0, 1))
                elif opt.dataset == 'hcigesture':
                    ori_prediction_file.write('%d %d %f\n' % (window_tail_frame, 10, 1))
            #检测为无手势
            if passive_count >= opt.det_counter:
                active = False
                if opt.dataset == 'ipn':
                    predict_gesture = 0
                elif opt.dataset == 'hcigesture':
                    predict_gesture = 10
            else:
                active = True  
            if active:
                best2, best1 = tuple(clf_selected_queue.argsort()[-2:][::1])
                if outputs_clf[prediction_clf] > opt.clf_threshold_final:
                    if predict_gesture != prev_gesture: # and not (prev_gesture=='double-click' and predict_gesture=='click')
                        if activate_count >= opt.cls_counter:  # activate a new gesture
                            active_index = 1
                            start_frame_index = window_tail_frame-opt.sample_duration_clf+14 # window_head_frame
                            activate_count = 0
                            isAdded = False
                            prev_gesture = predict_gesture
                            # last_activate_frame = window_tail_frame#num_frame
                        else:
                            activate_count += opt.stride_len

                    elif float(clf_selected_queue[best1]- clf_selected_queue[best2]) > opt.clf_threshold_pre:
                        # active_index += opt.stride_len
                        last_activate_frame = window_tail_frame-24#num_frame
                        if active_index > opt.active_pre:
                            # if int(last_activate_frame) > int(start_frame_index):
                            #     last_activate_frame = start_frame_index-1
                            # results[video_name][start_frame_index] = (last_activate_frame, prev_gesture)
                            results_No[start_frame_index] = (last_activate_frame, prev_gesture)
                            print('Early detect',start_frame_index,last_activate_frame, prev_gesture,clf_selected_queue[best1],clf_selected_queue[best2])
                            return prev_gesture
                    else: 
                        active_index += opt.stride_len
                        last_activate_frame = window_tail_frame-24#num_frame
                        if active_index > opt.active_threshold and (float(clf_selected_queue[best1]- clf_selected_queue[best2]) > 0.5): # and not isAdded:
                            # if int(last_activate_frame) > int(start_frame_index):
                            #     last_activate_frame = start_frame_index-1
                            # results[video_name][start_frame_index] = (last_activate_frame, prev_gesture)
                            results_No[start_frame_index] = (last_activate_frame, prev_gesture)
                            print('later detect',start_frame_index,last_activate_frame, prev_gesture,clf_selected_queue[best1],clf_selected_queue[best2])
                            return prev_gesture

        #     else:
        #         if predict_gesture != prev_gesture:
        #             start_frame_index = window_tail_frame-opt.det_counter
        #         if opt.dataset == 'ipn':
        #             prev_gesture = 0
        #         elif opt.dataset == 'hcigesture':
        #             prev_gesture = 10
        #         last_activate_frame = window_tail_frame#num_frame
        #         results[video_name][start_frame_index] = (last_activate_frame, prev_gesture)
        # end_all = time.time()
        # print("运行时间:%.2f毫秒"%((end_all-start_all)*1000))
        # print('***'*10)
    except KeyboardInterrupt:
        outi.release()
        outs.release()
        pass
        # result_line = video_name
        # for s in results[video_name]:
        #     e, label  = results[video_name][s]
        #     result_line += ','+' '.join([str(s),str(e),str(label)])
        #     pred_classes.append(label)
        # results_file.write(result_line+'\n')
        # result_line = video_name
        # for s in results_No[video_name]:
        #     e, label  = results_No[video_name][s]
        #     result_line += ','+' '.join([str(s),str(e),str(label)])
        # results_No_file.write(result_line+'\n')
        # sys.stdout.flush()

        # if opt.dataset == 'ipn':
        #     true_classes = []
        #     true_frames = []
        #     true_starts = []
        #     with open('data/ipn/online/vallistall.txt') as csvfile:
        #         readCSV = csv.reader(csvfile, delimiter=' ')
        #         for row in readCSV:
        #             if row[0][2:] == opt.whole_path:
        #                 if row[1] != '1' :
        #                     true_classes.append(int(row[1])-1)
        #                     true_starts.append(int(row[2]))
        #                     true_frames.append(int(row[3]))
        # elif opt.dataset == 'hcigesture':
        #     true_classes = []
        #     with open(os.path.join('data/LD/online/vallistall.txt')) as csvfile:
        #         readCSV = csv.reader(csvfile, delimiter=' ')
        #         for row in readCSV:
        #             if row[0].replace('./', '').replace('_color_all', '') == opt.whole_path:
        #                 if row[1] != '11':#str(opt.n_classes+1):
        #                     true_classes.append(int(row[1])-1)


        # #距离计算,生成动作序列？
        # # pred_frames = np.array(pred_frames)
        # true_classes = np.array(true_classes)
        # predicted = np.array(pred_classes)
        # print(predicted)
        # levenshtein_file.write('{}\npred_classes:{}\n'.format(video_name,predicted))
        # predicted =  np.array([key for key, _ in groupby(predicted)])
        # # print(predicted)
        # if opt.dataset == 'ipn':
        #     predicted = predicted[predicted != 0]
        # elif opt.dataset == 'hcigesture':
        #     predicted = predicted[predicted != 10]
        # print(predicted)
        # print(true_classes)
        # levenshtein_distance = LevenshteinDistance(true_classes, predicted)
        # print(levenshtein_distance)
        # levenshtein_file.write('pred_classes:{}\ntrue_classes:{}\nlevenshtein_distance:{}\n'.format(predicted,true_classes,levenshtein_distance))
        # # pdb.set_trace()
        # levenshtein_accuracy = 1-(levenshtein_distance/len(true_classes))

        # if levenshtein_distance <0: # Distance cannot be less than 0
        #     levenshtein_accuracies.update(0, len(true_classes))
        #     # pass
        # else:
        #     levenshtein_accuracies.update(levenshtein_accuracy, len(true_classes))
        # print('Levenshtein Accuracy = {} ({}) frame detections: {}/{}'.format(levenshtein_accuracies.val, levenshtein_accuracies.avg, 0, 0))
        # levenshtein_file.write('Levenshtein Accuracy = {} ({}) frame detections: {}/{}'.format(levenshtein_accuracies.val, levenshtein_accuracies.avg, 0, 0)+'\n')
        # # exit(0)
        # print('Average Levenshtein Accuracy= {}'.format(levenshtein_accuracies.avg))

        # print('-----Evaluation is finished------')
        # results_file.close()
        # ori_prediction_file.close()
        # levenshtein_file.close()
    finally:
        pass