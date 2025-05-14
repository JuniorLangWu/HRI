# STEP 1: Import the necessary modules.
import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import numpy as np
import mmcv
#全局变量
num_hand = 1
#mediapipe手部姿势估计器
base_options = python.BaseOptions(model_asset_path='tools/online_tools/hand_landmarker.task', delegate=python.BaseOptions.Delegate.GPU)
running_mode = vision.RunningMode.VIDEO
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       running_mode =running_mode,
                                       num_hands = num_hand,
                                       min_hand_detection_confidence = 0.8,
                                       min_hand_presence_confidence = 0.8,
                                       min_tracking_confidence = 0.8
                                         )
hands = vision.HandLandmarker.create_from_options(options)

#ipn姿势估计
def posehand_inference(frames):
    total_frames = len(frames)
    [HEIGHT_ori, WIDTH_ori] = frames[0].shape[:2]
    kp = np.zeros((1, total_frames, 21, 3), dtype=np.float32)

    for ret, frame in enumerate(frames):
        #测量代码运行时间
        start =time.time()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        #realsense等相机也不是自拍相机
        #默认检测照片为自拍,如果不是自拍,加上翻转
        frame = frame.astype(np.uint8)
        # frame = cv2.flip(frame, 1)
        HEIGHT_R, WIDTH_R, CHANNEL_R = frame.shape
        #mediapipe手部姿势估计
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ori_frame = frame  
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        results_hand = hands.detect_for_video(mp_image, int(time.time()*1000))

        
        #每次生成变量
        hkp = np.zeros((num_hand*21, 3), dtype=np.float32)
        hands_keypoints = np.zeros((num_hand*21, 2), dtype=np.float32)

        #获取手部骨骼点
        for i, left_right_landmarks in enumerate(results_hand.hand_landmarks):
            #防止出现三只手/两只手
            if i >= num_hand:
                #捉bug
                print('手数不对'*10)
                break  
            # 查看输出手标签
            # print(left_right_landmarks)
            # exit(0)
            hand_label = results_hand.handedness[i][0].category_name
            #检测到不对手
            # if hand_label == 'Right':
            landmarks_list = left_right_landmarks
            # 将landmarks坐标转换为numpy数组,归一化消除,图片翻转坐标变化,将坐标映射到大图中
            hkp  = np.array(
                [[landmark.x * WIDTH_R, landmark.y * HEIGHT_R, results_hand.handedness[i][0].score] for landmark in landmarks_list]) 
            #获取值
            # print(hkp)
            # exit(0)
        hands_keypoints = hkp[..., :2].astype(np.float16)
        kp[0, ret, :, 0:2] = hands_keypoints
        hands_score = hkp[..., 2].astype(np.float16)
        kp[0, ret, :, 2] = hands_score
        end =time.time()
        # print('手部姿势估计:{}'.format((end-start)*1000))
        # print('---'*5)
    return kp   

def posehand_inference_once(frame):

    kp = np.zeros((1, 1, 21, 3), dtype=np.float32)


    #测量代码运行时间
    start =time.time()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    #realsense等相机也不是自拍相机
    #默认检测照片为自拍,如果不是自拍,加上翻转
    frame = frame.astype(np.uint8)
    # frame = cv2.flip(frame, 1)
    HEIGHT_R, WIDTH_R, CHANNEL_R = frame.shape
    #mediapipe手部姿势估计
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    ori_frame = frame  
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    results_hand = hands.detect_for_video(mp_image, int(time.time()*1000))

    
    #每次生成变量
    hkp = np.zeros((num_hand*21, 3), dtype=np.float32)
    hands_keypoints = np.zeros((num_hand*21, 2), dtype=np.float32)

    #获取手部骨骼点
    for i, left_right_landmarks in enumerate(results_hand.hand_landmarks):
        #防止出现三只手/两只手
        if i >= num_hand:
            #捉bug
            print('手数不对'*10)
            break  
        # 查看输出手标签
        # print(left_right_landmarks)
        # exit(0)
        hand_label = results_hand.handedness[i][0].category_name
        #检测到不对手
        # if hand_label == 'Right':
        landmarks_list = left_right_landmarks
        # 将landmarks坐标转换为numpy数组,归一化消除,图片翻转坐标变化,将坐标映射到大图中
        hkp  = np.array(
            [[landmark.x * WIDTH_R, landmark.y * HEIGHT_R, results_hand.handedness[i][0].score] for landmark in landmarks_list]) 
        #获取值
        # print(hkp)
        # exit(0)
    hands_keypoints = hkp[..., :2].astype(np.float16)
    kp[..., 0:2] = hands_keypoints
    hands_score = hkp[..., 2].astype(np.float16)
    kp[..., 2] = hands_score
    end =time.time()
    # print('手部姿势估计:{}'.format((end-start)*1000))
    # print('---'*5)
    return kp   

#检测不到手缩小检测框
def hand_crop(bboxes):
    xlen = (bboxes[2]-bboxes[0])*0.1
    ylen = (bboxes[3]-bboxes[1])*0.1
    bboxes[0] = bboxes[0]+3*xlen
    bboxes[1] = bboxes[1]+3*ylen
    bboxes[2] = bboxes[2]-3*xlen
    bboxes[3] = bboxes[3]-2*ylen
    # print('---')
    return bboxes

#依然检测不到
#检测不到手缩小检测框
def hand_crop_hand(bboxes):
    xlen = (bboxes[2]-bboxes[0])*0.1
    ylen = (bboxes[3]-bboxes[1])*0.1
    bboxes[0] = bboxes[0]+4*xlen
    bboxes[1] = bboxes[1]+4*ylen
    bboxes[2] = bboxes[2]-4*xlen
    bboxes[3] = bboxes[3]-3*ylen
    # print('---')
    return bboxes

#LD姿势估计
def posehand_inference_crop(frames, crop):
    total_frames = len(frames)
    [HEIGHT_ori, WIDTH_ori] = frames[0].shape[:2]
    kp = np.zeros((1, total_frames, 21, 3), dtype=np.float32)

    BBOXE = np.array(crop[next(iter(crop))])
    CROP = BBOXE.copy()
    BBOXE = hand_crop(BBOXE)
    start_run = True
    point_run = 0

    for ret, frame in enumerate(frames):
        #测量代码运行时间
        start =time.time()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        #realsense等相机也不是自拍相机
        #默认检测照片为自拍,如果不是自拍,加上翻转
        frame = frame.astype(np.uint8)
        # frame = cv2.flip(frame, 1)
        #mediapipe手部姿势估计
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ori_frame = frame
        if str(ret+1) in crop:
            frame = mmcv.imcrop(frame, BBOXE)         
        else:
            frame = mmcv.imcrop(frame, BBOXE)
        # print(BBOXE)  
        #获取检测框大小
        X, Y ,X1, Y1= BBOXE
        HEIGHT_R, WIDTH_R, CHANNEL_R = frame.shape
        #realsense等相机也不是自拍相机
        #默认检测照片为自拍,如果不是自拍,加上翻转
        frame = cv2.flip(frame, 1)  
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        results_hand = hands.detect_for_video(mp_image, int(time.time()*1000))

        
        #每次生成变量
        hkp = np.zeros((num_hand*21, 3), dtype=np.float32)
        hands_keypoints = np.zeros((num_hand*21, 2), dtype=np.float32)

        #获取手部骨骼点
        for i, left_right_landmarks in enumerate(results_hand.hand_landmarks):
            #防止出现三只手/两只手
            if i >= num_hand:
                #捉bug
                print('手数不对'*10)
                break  
            # 查看输出手标签
            # print(left_right_landmarks)
            # exit(0)
            hand_label = results_hand.handedness[i][0].category_name
            #检测到不对手
            # if hand_label == 'Right':
            landmarks_list = left_right_landmarks
            # 将landmarks坐标转换为numpy数组,归一化消除,图片翻转坐标变化,将坐标映射到大图中
            hkp  = np.array(
                [[WIDTH_R - landmark.x * WIDTH_R + X, landmark.y * HEIGHT_R + Y, results_hand.handedness[i][0].score] for landmark in landmarks_list]) 
            #获取值
            # print(hkp)
            # exit(0)
        hands_keypoints = hkp[..., :2].astype(np.float16)
        kp[0, ret, :, 0:2] = hands_keypoints
        hands_score = hkp[..., 2].astype(np.float16)
        kp[0, ret, :, 2] = hands_score
        end =time.time()
        #检测不到手则检测人，下一帧使用检测结果，并且记录检测结果直到视频结束
        if not len(results_hand.handedness):
            point_run = point_run + 1
            if start_run and point_run>=10:
                # print(BBOXE)
                BBOXE = hand_crop_hand(CROP.copy())
                # print(BBOXE)
                # print('-----')
                start_run = False
            #手部姿势捕捉不成功调用前面
            # if 0 <= point_run < 2:
            #     # kp[0, ret, :, 0:2] = kp[0, ret-point_run, :, 0:2]
            # else:
            #     point_run = 0
        if len(results_hand.handedness):
            hand_x = hands_keypoints[:, 0]
            hand_y = hands_keypoints[:, 1]
            xlen = (BBOXE[2]-BBOXE[0])*0.1
            ylen = (BBOXE[3]-BBOXE[1])*0.1
            
            BBOXE[0] = min(BBOXE[0], hand_x.min()-2*xlen)
            BBOXE[1] = min(BBOXE[1],hand_y.min()-2*ylen)
            BBOXE[2] = max(BBOXE[2], hand_x.max()+2*xlen)
            BBOXE[3] = max(BBOXE[3], hand_y.max()+2*ylen)

            BBOXE[0] = max(BBOXE[0], 0)
            BBOXE[1] = max(BBOXE[1], 0)
            BBOXE[2] = min(BBOXE[2], WIDTH_ori)
            BBOXE[3] = min(BBOXE[3], HEIGHT_ori)

    return kp, BBOXE

def posehand_inference_once_crop(frame, BBOXE):

    kp = np.zeros((1, 1, 21, 3), dtype=np.float32)


    #测量代码运行时间
    start =time.time()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    #realsense等相机也不是自拍相机
    #默认检测照片为自拍,如果不是自拍,加上翻转
    frame = frame.astype(np.uint8)
    # frame = cv2.flip(frame, 1)
    #mediapipe手部姿势估计
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    ori_frame = frame 
    frame = mmcv.imcrop(frame, BBOXE)
    #获取检测框大小
    X, Y ,X1, Y1= BBOXE
    HEIGHT_R, WIDTH_R, CHANNEL_R = frame.shape
    #realsense等相机也不是自拍相机
    #默认检测照片为自拍,如果不是自拍,加上翻转
    frame = cv2.flip(frame, 1)   
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    results_hand = hands.detect_for_video(mp_image, int(time.time()*1000))

    
    #每次生成变量
    hkp = np.zeros((num_hand*21, 3), dtype=np.float32)
    hands_keypoints = np.zeros((num_hand*21, 2), dtype=np.float32)

    #获取手部骨骼点
    for i, left_right_landmarks in enumerate(results_hand.hand_landmarks):
        #防止出现三只手/两只手
        if i >= num_hand:
            #捉bug
            print('手数不对'*10)
            break  
        # 查看输出手标签
        # print(left_right_landmarks)
        # exit(0)
        hand_label = results_hand.handedness[i][0].category_name
        #检测到不对手
        # if hand_label == 'Right':
        landmarks_list = left_right_landmarks
            # 将landmarks坐标转换为numpy数组,归一化消除,图片翻转坐标变化,将坐标映射到大图中
        hkp  = np.array(
                [[WIDTH_R - landmark.x * WIDTH_R + X, landmark.y * HEIGHT_R + Y, results_hand.handedness[i][0].score] for landmark in landmarks_list]) 
        #获取值
        # print(hkp)
        # exit(0)
    hands_keypoints = hkp[..., :2].astype(np.float16)
    kp[..., 0:2] = hands_keypoints
    hands_score = hkp[..., 2].astype(np.float16)
    kp[..., 2] = hands_score
    end =time.time()
    # print('手部姿势估计:{}'.format((end-start)*1000))
    # print('---'*5)
    return kp 


def posehand_inference_online(frame):

    kp = np.zeros((1, 1, 21, 3), dtype=np.float32)


    #测量代码运行时间
    start =time.time()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    #realsense等相机也不是自拍相机
    #默认检测照片为自拍,如果不是自拍,加上翻转
    frame = frame.astype(np.uint8)
    # frame = cv2.flip(frame, 1)
    HEIGHT_R, WIDTH_R, CHANNEL_R = frame.shape
    #mediapipe手部姿势估计
    ori_frame = frame  
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    results_hand = hands.detect_for_video(mp_image, int(time.time()*1000))

    
    #每次生成变量
    hkp = np.zeros((num_hand*21, 3), dtype=np.float32)
    hands_keypoints = np.zeros((num_hand*21, 2), dtype=np.float32)

    #获取手部骨骼点
    for i, left_right_landmarks in enumerate(results_hand.hand_landmarks):
        #防止出现三只手/两只手
        if i >= num_hand:
            #捉bug
            print('手数不对'*10)
            break  
        # 查看输出手标签
        # print(left_right_landmarks)
        # exit(0)
        hand_label = results_hand.handedness[i][0].category_name
        #检测到不对手
        # if hand_label == 'Right':
        landmarks_list = left_right_landmarks
        # 将landmarks坐标转换为numpy数组,归一化消除,图片翻转坐标变化,将坐标映射到大图中
        hkp  = np.array(
            [[landmark.x * WIDTH_R, landmark.y * HEIGHT_R, results_hand.handedness[i][0].score] for landmark in landmarks_list]) 
        #获取值
        # print(hkp)
        # exit(0)
    hands_keypoints = hkp[..., :2].astype(np.float16)
    kp[..., 0:2] = hands_keypoints
    hands_score = hkp[..., 2].astype(np.float16)
    kp[..., 2] = hands_score
    end =time.time()
    # print('手部姿势估计:{}'.format((end-start)*1000))
    # print('---'*5)
    return kp   