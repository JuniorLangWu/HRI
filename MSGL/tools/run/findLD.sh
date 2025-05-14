#!/bin/bash

echo "$(date '+%Y-%m-%d %H:%M:%S')>>>msstgcn找最好执行开始" >> find.out

 bash tools/dist_train.sh configs/stgcn++_hand/stfgcnat_LD_No_mediapipe_crop_j.py  1  --validate  --test-best  --deterministic  --rank 1

 # 检查上一个命令是否成功执行
if [ $? -ne 0 ]; then
    echo "msstgcn1 执行失败" >> find.out
    exit 1
fi
echo "msstgcn1 执行成功" >> find.out


 bash tools/dist_train.sh configs/stgcn++_hand/stfgcnat_LD_No_mediapipe_crop_j.py  1  --validate  --test-best  --deterministic  --rank 2
 
  # 检查上一个命令是否成功执行
if [ $? -ne 0 ]; then
    echo "msstgcn2 执行失败" >> find.out
    exit 1
fi
echo "msstgcn2 执行成功" >> find.out

 bash tools/dist_train.sh configs/stgcn++_hand/stfgcnat_LD_No_mediapipe_crop_j.py  1  --validate  --test-best  --deterministic  --rank 3
 
  # 检查上一个命令是否成功执行
if [ $? -ne 0 ]; then
    echo "msstgcn3 执行失败" >> find.out
    exit 1
fi
echo "msstgcn3 执行成功" >> find.out