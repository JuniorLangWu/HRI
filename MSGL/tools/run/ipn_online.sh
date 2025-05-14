#!/bin/bash

D=45

S=30
Q=$((S / 2))
echo "$(date '+%Y-%m-%d %H:%M:%S')>>>ipn_p${S}_${D}_None执行开始" > ipn_p${S}_${D}_None.txt
bash tools/online_ipn_test_two.sh --clf_queue_size $Q --sample_duration_clf $S --set_work >>ipn_p${S}_${D}_None.txt
# 检查上一个命令是否成功执行
if [ $? -ne 0 ]; then
    echo "ipn_p${S}_${D}_None执行失败" >> ipn_p${S}_${D}_None.txt
    exit 1
fi
echo "$(date '+%Y-%m-%d %H:%M:%S')>>>ipn_p${S}_${D}_None执行成功" >> ipn_p${S}_${D}_None.txt

# S=84
# Q=$((S / 2))
# echo "$(date '+%Y-%m-%d %H:%M:%S')>>>ipn_p${S}_${D}_None执行开始" > ipn_p${S}_${D}_None.txt
# bash tools/online_ipn_test_two.sh --clf_queue_size $Q --sample_duration_clf $S --set_work >>ipn_p${S}_${D}_None.txt
# # 检查上一个命令是否成功执行
# if [ $? -ne 0 ]; then
#     echo "ipn_p${S}_${D}_None执行失败" >> ipn_p${S}_${D}_None.txt
#     exit 1
# fi
# echo "$(date '+%Y-%m-%d %H:%M:%S')>>>ipn_p${S}_${D}_None执行成功" >> ipn_p${S}_${D}_None.txt

# S=92
# Q=$((S / 2))
# echo "$(date '+%Y-%m-%d %H:%M:%S')>>>ipn_p${S}_${D}_None执行开始" > ipn_p${S}_${D}_None.txt
# bash tools/online_ipn_test_two.sh --clf_queue_size $Q --sample_duration_clf $S --set_work >>ipn_p${S}_${D}_None.txt
# # 检查上一个命令是否成功执行
# if [ $? -ne 0 ]; then
#     echo "ipn_p${S}_${D}_None执行失败" >> ipn_p${S}_${D}_None.txt
#     exit 1
# fi
# echo "$(date '+%Y-%m-%d %H:%M:%S')>>>ipn_p${S}_${D}_None执行成功" >> ipn_p${S}_${D}_None.txt



