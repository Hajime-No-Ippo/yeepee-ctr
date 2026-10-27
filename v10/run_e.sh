#!/bin/bash
cd /workspace/huawei/v10; export PYTHONUNBUFFERED=1; mkdir -p feat_sq10e
SQ_EP=2 SQ_LR=4e-3 SQ_BS=512 SQ_MAXSTEPS=100000 SQ_WORK=/workspace/huawei SQ_OUT=feat_sq10e taskset -c 0-3 /workspace/venv/bin/python seq10.py val25 f5 f1 f2 f3 val23 f4 f6 val26 | grep -v " ep"
/workspace/venv/bin/python sqauc.py feat_sq10e
bash scr.sh v10E feat_lr2,feat_tt,feat_pp,feat_sq10e "25 23 26" --drop '^sq_(pdot|pcos|mdot|mcos)$'
/workspace/venv/bin/python cmp10.py v10E --w 0
echo E_DONE
