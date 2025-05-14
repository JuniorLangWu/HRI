#!/bin/bash

echo "$(date '+%Y-%m-%d %H:%M:%S')>>>ipnmsst外面执行开始" >> find.out
 bash tools/dist_train.sh configs/stgcn++_hand/stfgcn_ipn_No_mediapipe_snip_j.py  1  --validate  --test-best  --deterministic  --rank 1

# 检查上一个命令是否成功执行
if [ $? -ne 0 ]; then
    echo "1 执行失败" >> find.out
    exit 1
fi
echo "1 执行成功" >> find.out


 bash tools/dist_train.sh configs/stgcn++_hand/stfgcn_ipn_No_mediapipe_snip_j.py  1  --validate  --test-best  --deterministic  --rank 2

 # 检查上一个命令是否成功执行
if [ $? -ne 0 ]; then
    echo "2 执行失败" >> find.out
    exit 1
fi
echo "2 执行成功" >> find.out

 bash tools/dist_train.sh configs/stgcn++_hand/stfgcn_ipn_No_mediapipe_snip_j.py  1  --validate  --test-best  --deterministic  --rank 3
 
  # 检查上一个命令是否成功执行
if [ $? -ne 0 ]; then
    echo "3 执行失败" >> find.out
    exit 1
fi
echo "3 执行成功" >> find.out

 bash tools/dist_train.sh configs/stgcn++_hand/stfgcn_ipn_No_mediapipe_snip_j.py  1  --validate  --test-best  --deterministic  --rank 4
 
  # 检查上一个命令是否成功执行
if [ $? -ne 0 ]; then
    echo "4 执行失败" >> find.out
    exit 1
fi
echo "4 执行成功" >> find.out

 bash tools/dist_train.sh configs/stgcn++_hand/stfgcn_ipn_No_mediapipe_snip_j.py  1  --validate  --test-best  --deterministic  --rank 5
 
  # 检查上一个命令是否成功执行
if [ $? -ne 0 ]; then
    echo "5 执行失败" >> find.out
    exit 1
fi
echo "5 执行成功" >> find.out

 bash tools/dist_train.sh configs/stgcn++_hand/stfgcn_ipn_No_mediapipe_snip_j.py  1  --validate  --test-best  --deterministic  --rank 6
 
  # 检查上一个命令是否成功执行
if [ $? -ne 0 ]; then
    echo "6 执行失败" >> find.out
    exit 1
fi
echo "6 执行成功" >> find.out