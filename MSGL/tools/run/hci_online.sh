#!/bin/bash

D=10

S=4
Q=$((S / 2))
echo "$(date '+%Y-%m-%d %H:%M:%S')>>>hci_p${S}_${D}_None执行开始" > hci_p${S}_${D}_None.txt
bash tools/online_hci_test_two.sh --clf_queue_size $Q --sample_duration_clf $S --set_work >>hci_p${S}_${D}_None.txt
# 检查上一个命令是否成功执行
if [ $? -ne 0 ]; then
    echo "hci_p${S}_${D}_None执行失败" >> hci_p${S}_${D}_None.txt
    exit 1
fi
echo "$(date '+%Y-%m-%d %H:%M:%S')>>>hci_p${S}_${D}_None执行成功" >> hci_p${S}_${D}_None.txt

