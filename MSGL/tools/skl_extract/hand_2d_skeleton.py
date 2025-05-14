# Copyright (c) OpenMMLab. All rights reserved.
#视频检测转化为输入annos
import argparse
import copy as cp
import decord
import cv2
import mmcv
import numpy as np
import os
import os.path as osp
import torch.distributed as dist
from mmcv.runner import get_dist_info, init_dist
from tqdm import tqdm
from PIL import Image

import pyskl  # noqa: F401
from pyskl.smp import mrlines

import mediapipe as mp

pyskl_root = osp.dirname(pyskl.__path__[0])

HEIGHT = 480
WIDTH = 640
def extract_frame(video_path):
    #文件夹帧操作
    if os.path.isdir(video_path):
        # print('文件夹')
        vid = []
        for i, img in enumerate(os.listdir(video_path)):
            img = os.path.join(video_path, img)
            pil_img = Image.open(img)
            pil_img = pil_img.convert("RGB")  # 建议使用，可将所有（包括灰色）转化为RGB格式
            pil_img = pil_img.resize((640, 480))
            # pil_img = np.array(pil_img)
            pil_img = np.array(pil_img)[:, :, ::-1]
            vid.append(pil_img)
        return vid
    #视频帧操作
    else:
        if video_path.split(':'):
            [video_path, start2end] = video_path.split(':')
            [start_frame, end_frame] = start2end.split('_')
        # print('视频文件')
        # print(video_path)
        # print(start_frame)
        # print(end_frame)
        # vid = decord.VideoReader('/home/user/Videos/dataset/IPN_Hand/videos/1CM1_1_R_#218.avi')
        # 使用OpenCV打开视频
        cap = cv2.VideoCapture(video_path + '.avi')
        
        # 创建一个空列表来存储帧
        frames = []
        
        # 读取视频的帧，直到达到结束帧或没有帧可读
        current_frame = 0
        while current_frame < int(end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            
            # 如果当前帧处于所需范围内，则将其保存
            if current_frame >= (int(start_frame)-1):
                # 将帧转换为RGB格式（OpenCV默认的是BGR格式）
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            
            current_frame += 1
        # 关闭视频文件
        cap.release()
        # print(current_frame)
        # print(len(frames))
        # exit(0)
        return frames
    
#手部模型mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8)

#手部姿势估计
def hand_inference(anno_in, handmodel, frames, compress=False):
    anno = cp.deepcopy(anno_in)
    total_frames = len(frames)
    [HEIGHT, WIDTH] = frames[0].shape[:2]
    # print(WIDTH)
    # exit(0)
    anno['total_frames'] = total_frames
    num_hand = 1
    anno['num_hand_raw'] = num_hand
    if compress:
        pass
    else:
        kp = np.zeros((num_hand, total_frames, 21, 3), dtype=np.float32)
        for i, f in enumerate(frames):
            # mediapip估计
            handpose = handmodel.process(f)
            if handpose.multi_hand_landmarks:
                for j, (landmarks, hand_label) in enumerate(zip(handpose.multi_hand_landmarks,handpose.multi_handedness)):
                    #防止出现三只手/两只手
                    if j >= num_hand:
                        break  
                    landmarks_list = landmarks.landmark  
                    # 将landmarks坐标转换为numpy数组            
                    kp[j, i]  = np.array([[landmark.x * WIDTH, landmark.y * HEIGHT, hand_label.classification[0].score] for landmark in landmarks_list])                 
        anno['keypoint'] = kp[..., :2].astype(np.float16)
        anno['keypoint_score'] = kp[..., 2].astype(np.float16)
        # print(anno)
    return anno
#参数
def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate 2D pose annotations for a custom video dataset')
    # * Accepted formats for each line in video_list are:
    # * 1. "xxx.mp4" ('label' is missing, the dataset can be used for inference, but not training)
    # * 2. "xxx.mp4 label" ('label' is an integer (cat.split("_")egory index),
    # * the result can be used for both training & testing)
    # * All lines should take the same format.
    parser.add_argument('--video-list', type=str, help='the list of source videos')
    # * out should ends with '.pkl'
    parser.add_argument('--out', type=str, help='output pickle name')
    parser.add_argument('--tmpdir', type=str, default='tmp')
    parser.add_argument('--local_rank', type=int, default=0)
    # * When non-dist is set, will only use 1 GPU
    parser.add_argument('--non-dist', action='store_true', help='whether to use distributed skeleton extraction')
    parser.add_argument('--compress', action='store_true', help='whether to do K400-style compression')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    args = parser.parse_args()
    return args

#主函数
def main():
    #获取参数，并判断参数
    args = parse_args()
    assert args.out.endswith('.pkl')

    lines = mrlines(args.video_list)
    lines = [x.split() for x in lines]

    # * We set 'frame_dir' as the base name (w/o. suffix) of each video
    assert len(lines[0]) in [1, 2, 4]
    #新建annos，并赋予文件基础信息
    if len(lines[0]) == 1:
        annos = [dict(frame_dir=osp.basename(x[0]), filename=x[0]) for x in lines]
    elif len(lines[0]) == 2:
        annos = [dict(frame_dir=osp.basename(x[0]), filename=x[0], label=int(x[1])) for x in lines]
    else:
        annos = [dict(frame_dir=osp.basename(x[0]) + ':' + x[2] + '_' + x[3], filename=x[0] + ':' + x[2] + '_' + x[3], label=int(x[1])) for x in lines]
    if args.non_dist:
        my_part = annos
        os.makedirs(args.tmpdir, exist_ok=True)
    else:
        init_dist('pytorch', backend='nccl')
        rank, world_size = get_dist_info()
        if rank == 0:
            os.makedirs(args.tmpdir, exist_ok=True)
        dist.barrier()
        my_part = annos[rank::world_size]

    #训练获取手部骨架
    results = [] 
    for  anno in tqdm(my_part):
        frames = extract_frame(anno['filename'])
        shape = frames[0].shape[:2]
        anno['img_shape'] = shape
        #姿势估计
        anno = hand_inference(anno, hands, frames, compress=args.compress)    
        anno.pop('filename')
        results.append(anno)
    
    #
    if args.non_dist:
        mmcv.dump(results, args.out)
    else:
        mmcv.dump(results, osp.join(args.tmpdir, f'part_{rank}.pkl'))
        dist.barrier()

        if rank == 0:
            parts = [mmcv.load(osp.join(args.tmpdir, f'part_{i}.pkl')) for i in range(world_size)]
            rem = len(annos) % world_size
            if rem:
                for i in range(rem, world_size):
                    parts[i].append(None)

            ordered_results = []
            for res in zip(*parts):
                ordered_results.extend(list(res))
            ordered_results = ordered_results[:len(annos)]
            mmcv.dump(ordered_results, args.out)


if __name__ == '__main__':
    main()
