#!/bin/bash

echo "$(date '+%Y-%m-%d %H:%M:%S')>>>找寻AA和CTR的LD随机执行开始" >> find.out
 bash tools/dist_train.sh configs/aagcn_LD_No_mediapipe_crop_j.py  1  --validate  --test-best  --deterministic  --rank 1

 # 检查上一个命令是否成功执行
if [ $? -ne 0 ]; then
    echo "AALD1 执行失败" >> find.out
    exit 1
fi
echo "AALD1 执行成功" >> find.out


 bash tools/dist_train.sh configs/ctrgcn_LD_No_mediapipe_crop_j.py  1  --validate  --test-best  --deterministic  --rank 1
 
  # 检查上一个命令是否成功执行
if [ $? -ne 0 ]; then
    echo "CTRLD1 执行失败" >> find.out
    exit 1
fi
echo "CTRLD1 执行成功" >> find.out


