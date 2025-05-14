#!/bin/bash


echo "$(date '+%Y-%m-%d %H:%M:%S')>>>找寻AA和CTR随机执行开始" >> find.out
 bash tools/dist_train.sh configs/aagcn_ipn_No_mediapipe_snip_j1.py  1  --validate  --test-best  --deterministic  --rank 1

 # 检查上一个命令是否成功执行
if [ $? -ne 0 ]; then
    echo "AA1 执行失败" >> find.out
    exit 1
fi
echo "AA1 执行成功" >> find.out


 bash tools/dist_train.sh configs/ctrgcn_ipn_No_mediapipe_snip_j1.py  1  --validate  --test-best  --deterministic  --rank 1
 
  # 检查上一个命令是否成功执行
if [ $? -ne 0 ]; then
    echo "CTR1 执行失败" >> find.out
    exit 1
fi
echo "CTR1 执行成功" >> find.out

 bash tools/dist_train.sh configs/aagcn_ipn_No_mediapipe_snip_j1.py  1  --validate  --test-best  --deterministic  --rank 2

 # 检查上一个命令是否成功执行
if [ $? -ne 0 ]; then
    echo "AA2 执行失败" >> find.out
    exit 1
fi
echo "AA2 执行成功" >> find.out


 bash tools/dist_train.sh configs/ctrgcn_ipn_No_mediapipe_snip_j1.py  1  --validate  --test-best  --deterministic  --rank 2
 
  # 检查上一个命令是否成功执行
if [ $? -ne 0 ]; then
    echo "CTR2 执行失败" >> find.out
    exit 1
fi
echo "CTR2 执行成功" >> find.out

 bash tools/dist_train.sh configs/aagcn_ipn_No_mediapipe_snip_j1.py  1  --validate  --test-best  --deterministic  --rank 3

 # 检查上一个命令是否成功执行
if [ $? -ne 0 ]; then
    echo "AA3 执行失败" >> find.out
    exit 1
fi
echo "AA3 执行成功" >> find.out


 bash tools/dist_train.sh configs/ctrgcn_ipn_No_mediapipe_snip_j1.py  1  --validate  --test-best  --deterministic  --rank 3
 
  # 检查上一个命令是否成功执行
if [ $? -ne 0 ]; then
    echo "CTR3 执行失败" >> find.out
    exit 1
fi
echo "CTR3 执行成功" >> find.out