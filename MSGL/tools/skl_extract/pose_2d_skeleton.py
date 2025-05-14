# Copyright (c) OpenMMLab. All rights reserved.
#视频检测转化为输入annos
import argparse
import copy as cp
import decord
import mmcv
import numpy as np
import os
import os.path as osp
import torch.distributed as dist
from mmcv.runner import get_dist_info, init_dist
from tqdm import tqdm
from PIL import Image
import cv2
import time
import pyskl  # noqa: F401
from pyskl.smp import mrlines

try:
    import mmdet  # noqa: F401
    from mmdet.apis import inference_detector, init_detector
except (ImportError, ModuleNotFoundError):
    raise ImportError('Failed to import `inference_detector` and '
                      '`init_detector` form `mmdet.apis`. These apis are '
                      'required in this script! ')

try:
    import mmpose  # noqa: F401
    from mmpose.apis import inference_top_down_pose_model, init_pose_model
except (ImportError, ModuleNotFoundError):
    raise ImportError('Failed to import `inference_top_down_pose_model` and '
                      '`init_pose_model` form `mmpose.apis`. These apis are '
                      'required in this script! ')

pyskl_root = osp.dirname(pyskl.__path__[0])
default_det_config = f'{pyskl_root}/configs/faster_rcnn_r50_fpn_1x_coco-person.py'
default_det_ckpt = f'{pyskl_root}/pth/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth'
default_pose_config = f'{pyskl_root}/configs/hrnet_w32_coco_256x192.py'
default_pose_ckpt = f'{pyskl_root}/pth/hrnet_w32_coco_256x192-c78dce93_20200708.pth'


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

def detection_inference(model, frames):
    results = []
    for frame in frames:
        start = time.time()
        result = inference_detector(model, frame)
        results.append(result)
        end = time.time()
        print('单次检测需要时间{}'.format((end-start)*1000))
        print('***'*10)
    return results


def pose_inference(anno_in, model, frames, det_results, compress=False):
    anno = cp.deepcopy(anno_in)
    assert len(frames) == len(det_results)
    total_frames = len(frames)
    num_person = max([len(x) for x in det_results])
    anno['total_frames'] = total_frames
    anno['num_person_raw'] = num_person

    if compress:
        kp, frame_inds = [], []
        for i, (f, d) in enumerate(zip(frames, det_results)):
            # Align input format
            d = [dict(bbox=x) for x in list(d)]
            pose = inference_top_down_pose_model(model, f, d, format='xyxy')[0]
            for j, item in enumerate(pose):
                kp.append(item['keypoints'])
                frame_inds.append(i)
        anno['keypoint'] = np.stack(kp).astype(np.float16)
        anno['frame_inds'] = np.array(frame_inds, dtype=np.int16)
    else:
        kp = np.zeros((num_person, total_frames, 17, 3), dtype=np.float32)
        for i, (f, d) in enumerate(zip(frames, det_results)):
            # Align input format
            start = time.time()
            d = [dict(bbox=x) for x in list(d)]
            pose = inference_top_down_pose_model(model, f, d, format='xyxy')[0]
            for j, item in enumerate(pose):
                kp[j, i] = item['keypoints']
            end = time.time()
            print('单次估计需要时间{}'.format((end-start)*1000))
            print('---'*10)
        anno['keypoint'] = kp[..., :2].astype(np.float16)
        anno['keypoint_score'] = kp[..., 2].astype(np.float16)
        print(anno['keypoint'].shape)
        print(anno['keypoint_score'].shape)
        exit(0)
        #绘制pose骨骼点
        for i in anno['keypoint'][0]:
            for point in i[0:9, :]:
                x, y = map(int, point)
                # 定义点的颜色，这里使用红色 (0, 0, 255)
                color = (0, 0, 255)
                # 绘制点，圆心坐标为 (x, y)，半径为 5，颜色为 color，线宽为 -1（表示实心圆）
                cv2.circle(f, (x, y), 5, color, -1)
                #绘画出
                # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) 
            
            cv2.imshow('Pose_and_hands', f)
            # cv2.imshow('frame', frame)
            if cv2.waitKey(10) & 0xFF == 27:
                break
            # print("绘图运行时间:%.2f毫秒"%((end-start)*1000))
    return anno


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate 2D pose annotations for a custom video dataset')
    # * Both mmdet and mmpose should be installed from source
    # parser.add_argument('--mmdet-root', type=str, default=default_mmdet_root)
    # parser.add_argument('--mmpose-root', type=str, default=default_mmpose_root)
    parser.add_argument('--det-config', type=str, default=default_det_config)
    parser.add_argument('--det-ckpt', type=str, default=default_det_ckpt)
    parser.add_argument('--pose-config', type=str, default=default_pose_config)
    parser.add_argument('--pose-ckpt', type=str, default=default_pose_ckpt)
    # * Only det boxes with score larger than det_score_thr will be kept
    parser.add_argument('--det-score-thr', type=float, default=0.7)
    # * Only det boxes with large enough sizes will be kept,
    parser.add_argument('--det-area-thr', type=float, default=1600)
    # * Accepted formats for each line in video_list are:
    # * 1. "xxx.mp4" ('label' is missing, the dataset can be used for inference, but not training)
    # * 2. "xxx.mp4 label" ('label' is an integer (category index),
    # * the result can be used for both training & testing)
    # * All lines should take the same format.
    parser.add_argument('--video-list', type=str, help='the list of source videos',
                         default='/home/user/github/pyskl/data/ipn/ipnall.list')
    # * out should ends with '.pkl'
    parser.add_argument('--out', type=str, help='output pickle name',
                        default='/home/user/github/pyskl/data/ipn/ipnh_pose_hand_annos.pkl')
    # parser.add_argument('--out', type=str, help='output pickle name', default='/home/user/github/pyskl/data/paxis/dataset/paxis_annos.pkl')
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


def main():
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
    det_model = init_detector(args.det_config, args.det_ckpt, 'cuda')
    assert det_model.CLASSES[0] == 'person', 'A detector trained on COCO is required'
    pose_model = init_pose_model(args.pose_config, args.pose_ckpt, 'cuda')

    results = []
    for anno in tqdm(my_part):
        start = time.time()
        frames = extract_frame(anno['filename'])
        # print(np.array(frames).shape)
        # from matplotlib import pyplot as plt
        # plt.imshow(frames[0])
        # plt.axis('off')
        # plt.savefig("333.jpg", bbox_inches='tight')
        # plt.show()
        # exit(0)
        det_results = detection_inference(det_model, frames)
        # * Get detection results for human
        det_results = [x[0] for x in det_results]
        end = time.time()
        print('检测需要时间{}'.format((end-start)*1000))
        start = time.time()
        for i, res in enumerate(det_results):
            # * filter boxes with small scores
            res = res[res[:, 4] >= args.det_score_thr]
            # * filter boxes with small areas
            box_areas = (res[:, 3] - res[:, 1]) * (res[:, 2] - res[:, 0])
            assert np.all(box_areas >= 0)
            res = res[box_areas >= args.det_area_thr]
            det_results[i] = res
        shape = frames[0].shape[:2]
        anno['img_shape'] = shape
        anno = pose_inference(anno, pose_model, frames, det_results, compress=args.compress)
        end = time.time()
        print('估计需要时间{}'.format((end-start)*1000))
        anno.pop('filename')
        results.append(anno)

    # if args.non_dist:
    #     mmcv.dump(results, args.out)
    # else:
    #     mmcv.dump(results, osp.join(args.tmpdir, f'part_{rank}.pkl'))
    #     dist.barrier()

    #     if rank == 0:
    #         parts = [mmcv.load(osp.join(args.tmpdir, f'part_{i}.pkl')) for i in range(world_size)]
    #         rem = len(annos) % world_size
    #         if rem:
    #             for i in range(rem, world_size):
    #                 parts[i].append(None)

    #         ordered_results = []
    #         for res in zip(*parts):
    #             ordered_results.extend(list(res))
    #         ordered_results = ordered_results[:len(annos)]
    #         mmcv.dump(ordered_results, args.out)


if __name__ == '__main__':
    main()
