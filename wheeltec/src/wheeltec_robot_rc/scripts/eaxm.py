#!/usr/bin/env python
# coding=utf-8
import sys, os, select, termios, tty
import cv2
from online import gesture
import time
#获取键值函数
def getKey():
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.001)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key
try:
    settings = termios.tcgetattr(sys.stdin) #获取键值初始化，读取终端相关属性
    #realsense等相机也不是自拍相机
    cap = cv2.VideoCapture(0)
    while(1):
        start = time.time()
        ret, inputs = cap.read()
        ges = 0 
        if ret:
            ges = gesture(inputs)
            if ges:
                print(ges)
                break
        key = getKey() #获取键值
        print(key)
        if key =='q':
            break
        end = time.time()
        print("运行时间:%.2f毫秒"%((end-start)*1000))
        print('###'*10)
#运行出现问题则程序终止并打印相关错误信息
except Exception as e:
    print(e)

#程序结束前发布速度为0的速度话题
finally:
    pass

#程序结束前设置终端相关属性
termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)