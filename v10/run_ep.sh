#!/bin/bash
cd /workspace/huawei/v10; export PYTHONUNBUFFERED=1
for cfg in "2 3e-3" "2 4e-3" "2 6e-3"; do set -- $cfg; o=feat_sq10_ep$1_$2; mkdir -p $o
  env SQ_EP=$1 SQ_LR=$2 SQ_BS=512 SQ_MAXSTEPS=100000 SQ_WORK=/workspace/huawei SQ_OUT=$o taskset -c 0-3 /workspace/venv/bin/python seq10.py val25 | grep total
  /workspace/venv/bin/python sqauc.py $o
done
echo EP_DONE
