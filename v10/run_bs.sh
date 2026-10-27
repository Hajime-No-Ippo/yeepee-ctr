#!/bin/bash
cd /workspace/huawei/v10; export PYTHONUNBUFFERED=1

for cfg in "512 2e-3" "512 3e-3" "1024 4e-3"; do set -- $cfg
  mkdir -p feat_sq10_bs$1_$2
  env SQ_BS=$1 SQ_LR=$2 SQ_MAXSTEPS=100000 SQ_WORK=/workspace/huawei SQ_OUT=feat_sq10_bs$1_$2 taskset -c 0-3 /workspace/venv/bin/python seq10.py val25
  /workspace/venv/bin/python sqauc.py feat_sq10_bs$1_$2
done
echo BS_DONE
