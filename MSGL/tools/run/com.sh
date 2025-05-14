#!/bin/bash

echo "$(date '+%Y-%m-%d %H:%M:%S')>>>msstgcn执行开始" >> find.out
 bash tools/dist_train.sh configs/stgcn++_hand/stfgcn_ipn_No_mediapipe_snip_j.py  1  --validate  --test-best  --deterministic   --rank 1

# 检查上一个命令是否成功执行
if [ $? -ne 0 ]; then
    echo "attmf 执行失败" >> find.out
    exit 1
fi
echo "attmf 执行成功" >> find.out

 bash tools/dist_train.sh configs/stgcn++_hand/stfgcngt_ipn_No_mediapipe_snip_j.py  1  --validate  --test-best  --deterministic   --rank 1

# 检查上一个命令是否成功执行
if [ $? -ne 0 ]; then
    echo "gt 执行失败" >> find.out
    exit 1
fi
echo "gt 执行成功" >> find.out


 bash tools/dist_train.sh configs/stgcn++_hand/stfgcngf_ipn_No_mediapipe_snip_j.py  1  --validate  --test-best  --deterministic   --rank 1

 # 检查上一个命令是否成功执行
if [ $? -ne 0 ]; then
    echo "gf 执行失败" >> find.out
    exit 1
fi
echo "gf 执行成功" >> find.out


 bash tools/dist_train.sh configs/stgcn++_hand/stfgcnat_ipn_No_mediapipe_snip_j.py  1  --validate  --test-best  --deterministic   --rank 1
 
  # 检查上一个命令是否成功执行
if [ $? -ne 0 ]; then
    echo "at 执行失败" >> find.out
    exit 1
fi
echo "at 执行成功" >> find.out


