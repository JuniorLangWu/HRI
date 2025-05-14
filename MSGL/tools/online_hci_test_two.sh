#!/usr/bin/env bash

set -x
CONFIG=configs/online/stfgcn_hci_online_two.py

MKL_SERVICE_FORCE_INTEL=1 PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# Arguments starting from the forth one are captured by ${@:1}
python \
    $(dirname "$0")/two_stage_online_test.py $CONFIG ${@:1}
