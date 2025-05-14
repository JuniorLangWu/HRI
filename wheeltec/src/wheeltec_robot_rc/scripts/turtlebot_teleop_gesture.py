#!/usr/bin/env python
# coding=utf-8
# Copyright (c) 2011, Willow Garage, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#    * Neither the name of the Willow Garage, Inc. nor the names of its
#      contributors may be used to endorse or promote products derived from
#       this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import rospy

from geometry_msgs.msg import Twist

import sys, select, termios, tty

import cv2
from online import gesture
import time
import os
from mmcv import Config
cfg = Config.fromfile('{}/online_tools/online_configs/stfgcn_ipn_online_two.py'.format(os.path.dirname(os.path.abspath(__file__))))
if cfg.get('result_path', None) is not None:
    result_path = cfg.result_path
result_path = file_path = os.path.dirname(os.path.abspath(__file__)) + '/'+result_path + '/'
video_name = 'online'

msg = """
Control Your Turtlebot!
---------------------------
Moving around:
   u    i    o
   j    k    l
   m    ,    .

q/z : increase/decrease max speeds by 1%
w/x : increase/decrease only linear speed by 10%
e/c : increase/decrease only angular speed by 10%
space key, k : force stop
anything else : stop smoothly
b : switch to OmniMode/CommonMode
CTRL-C to quit
"""
Omni = 0 #全向移动模式

#键值对应移动/转向方向
moveBindings = {
        'i':( 1, 0),
        'o':( 1,-1),
        'j':( 0, 1),
        'l':( 0,-1),
        'u':( 1, 1),
        ',':(-1, 0),
        '.':(-1, 1),
        'm':(-1,-1),
           }

#键值对应速度增量
speedBindings={
        'q':(1.01,1.01),
        'z':(0.99,0.99),
        'w':(1.1,1),
        'x':(0.9,1),
        'e':(1,  1.1),
        'c':(1,  0.9),
          }

#获取键值函数
def getKey():
    tty.setraw(sys.stdin.fileno())
    #1ms读取键盘值
    rlist, _, _ = select.select([sys.stdin], [], [], 0.001)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


speed = 0.2 #默认移动速度 m/s
turn  = 0.5   #默认转向速度 rad/s
#以字符串格式返回当前速度
def vels(speed,turn):
    return "currently:\tspeed %s\tturn %s " % (speed,turn)

results_No = {}
#主函数
if __name__=="__main__":
    settings = termios.tcgetattr(sys.stdin) #获取键值初始化，读取终端相关属性
    #realsense等相机也不是自拍相机
    cap = cv2.VideoCapture(0)

    rospy.init_node('turtlebot_teleop') #创建ROS节点
    pub = rospy.Publisher('~cmd_vel', Twist, queue_size=5) #创建速度话题发布者，'~cmd_vel'='节点名/cmd_vel'

    x      = 0   #前进后退方向
    th     = 0   #转向/横向移动方向
    count  = 0   #键值不再范围计数
    target_speed = 0 #前进后退目标速度
    target_turn  = 0 #转向目标速度
    target_HorizonMove = 0 #横向移动目标速度
    control_speed = 0 #前进后退实际控制速度
    control_turn  = 0 #转向实际控制速度
    control_HorizonMove = 0 #横向移动实际控制速度
    # 初始化标志位
    mode_activated = False  # 记录是否已经切换了移动模式
    try:
        print(msg) #打印控制说明
        print(vels(speed,turn)) #打印当前速度
        while(1):
            start = time.time()
            ret, inputs = cap.read()
            ges = 0 
            # Moving around:
            #     u    i    o
            #     j    k    l
            #     m    ,    .
            if ret:

                ges = gesture(inputs, results_No)

                if ges == 0:
                    key = 'k'  
                key = getKey() #获取键值
                if ges:
                    print(ges)
                    #基本运动
                    if ges == 5:
                        key = 'i'
                        print('front')
                    if ges == 6:
                        key = ','
                        print('back')
                    if ges == 7:
                        key = 'j'
                        print('left')
                    if ges == 8:
                        key = 'l'
                        print('right')

                    #复合运动
                    if ges == 1:
                        key = 'u'
                        print('front+left')
                    if ges == 2:
                        key = 'm'
                        print('back+left')
                    if ges == 3 or ges == 10:
                        key = 'o'
                        print('front+right')
                    if ges == 4 or ges == 11:
                        key = '.'
                        print('back+right')
                    #调整速度
                    if ges == 12:
                        key = 'q'
                        print('increase 1%')
                    if ges == 13:
                        key = 'z'
                        print('decrease 1%')
                    if ges != 9:
                        mode_activated = True
                    if ges == 9:
                        key = 'b'
                        # if Omni: 
                        #     print("Switch to OmniMode")
                        # else:
                        #     print("Switch to CommonMode")                  
                        # print('switch Mode')
     
                    # if ges == 3 or ges == 4 or ges == 10 or ges == 11:
                    #     key = 'o'
                    # if ges == 1 or ges == 2:
                    #     key = 'u'
                    # if ges == 9 or ges == 12 or ges == 13:
                    #     key = 'i'

                    # if ges == 5 or ges ==6:
                    #     key = ','
                    # if ges == 7:
                    #     key = 'j'   #left
                    # if ges == 8:
                    #     key = 'l'
                    # print(key)
                    # break

            end = time.time()
            # print("运行时间:%.2f毫秒"%((end-start)*1000))
            # print('###'*10)

            #切换是否为全向移动模式，全向轮/麦轮小车可以加入全向移动模式
            if key=='b' and mode_activated:               
                Omni=~Omni
                if Omni: 
                    print("Switch to OmniMode")
                    moveBindings['.']=[-1,-1]
                    moveBindings['m']=[-1, 1]
                else:
                    print("Switch to CommonMode")
                    moveBindings['.']=[-1, 1]
                    moveBindings['m']=[-1,-1]
                mode_activated = False
            
            #判断键值是否在移动/转向方向键值内
            if key in moveBindings.keys():
                x  = moveBindings[key][0]
                th = moveBindings[key][1]
                count = 0

            #判断键值是否在速度增量键值内
            elif key in speedBindings.keys():
                speed = speed * speedBindings[key][0]
                turn  = turn  * speedBindings[key][1]
                count = 0
                print(vels(speed,turn)) #速度发生变化，打印出来

            #空键值/'k',相关变量置0
            elif key == ' ' or key == 'k' :
                x  = 0
                th = 0
                control_speed = 0
                control_turn  = 0
                HorizonMove   = 0

            #长期识别到不明键值，相关变量置0
            else:
                count = count + 1
                if count > 4:
                    x  = 0
                    th = 0
                if (key == '\x03'):
                    break

            #根据速度与方向计算目标速度
            target_speed = speed * x
            target_turn  = turn * th
            target_HorizonMove = speed*th

            #平滑控制，计算前进后退实际控制速度
            if target_speed > control_speed:
                control_speed = min( target_speed, control_speed + 0.1 )
            elif target_speed < control_speed:
                control_speed = max( target_speed, control_speed - 0.1 )
            else:
                control_speed = target_speed

            #平滑控制，计算转向实际控制速度
            if target_turn > control_turn:
                control_turn = min( target_turn, control_turn + 0.5 )
            elif target_turn < control_turn:
                control_turn = max( target_turn, control_turn - 0.5 )
            else:
                control_turn = target_turn

            #平滑控制，计算横向移动实际控制速度
            if target_HorizonMove > control_HorizonMove:
                control_HorizonMove = min( target_HorizonMove, control_HorizonMove + 0.1 )
            elif target_HorizonMove < control_HorizonMove:
                control_HorizonMove = max( target_HorizonMove, control_HorizonMove - 0.1 )
            else:
                control_HorizonMove = target_HorizonMove
         
            twist = Twist() #创建ROS速度话题变量
            #根据是否全向移动模式，给速度话题变量赋值
            if Omni==0:
                twist.linear.x  = control_speed; twist.linear.y = 0;  twist.linear.z = 0
                twist.angular.x = 0;             twist.angular.y = 0; twist.angular.z = control_turn
            else:
                twist.linear.x  = control_speed; twist.linear.y = control_HorizonMove; twist.linear.z = 0
                twist.angular.x = 0;             twist.angular.y = 0;                  twist.angular.z = 0

            pub.publish(twist) #ROS发布速度话题

            results_No_file = open(os.path.join(result_path, 'ipn_None_results'+'.txt'), "w")
            result_line = video_name
            for s in results_No:
                e, label  = results_No[s]
                result_line += ','+' '.join([str(s),str(e),str(label)])
                # print(result_line)
            results_No_file.write(result_line+'\n')
            sys.stdout.flush()

    #运行出现问题则程序终止并打印相关错误信息
    except Exception as e:
        print(e)

    #程序结束前发布速度为0的速度话题
    finally:
        cap.release()
        # 关闭所有 OpenCV 窗口
        cv2.destroyAllWindows()
        # results_No_file.close()
        twist = Twist()
        twist.linear.x = 0;  twist.linear.y = 0;  twist.linear.z = 0
        twist.angular.x = 0; twist.angular.y = 0; twist.angular.z = 0
        pub.publish(twist)

    #程序结束前设置终端相关属性
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)

