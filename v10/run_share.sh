#!/bin/bash
# v10 "shared-fit" scheme: each sequence fit scores two consecutive train folds (the later one with a 1-day-staler model), eval keeps a fresh fit.
cd /workspace/huawei/v10; export PYTHONUNBUFFERED=1
while pgrep -f "run_abl2.sh" >/dev/null; do sleep 10; done
SQ_WORK=/workspace/huawei SQ_OUT=feat_sq10p taskset -c 0-3 /workspace/venv/bin/python seq10.py f1 f2@f1 f3 f4@f3
SQ_WORK=/workspace/huawei SQ_OUT=feat_sq10q taskset -c 0-3 /workspace/venv/bin/python seq10.py f2 f3@f2 f4 f5@f4
D='^sq_(pdot|pcos|mdot|mcos)$'
bash scr.sh v10P feat_lr2,feat_tt,feat_pp,feat_sq10p "25 23" --drop "$D"
bash scr.sh v10P feat_lr2,feat_tt,feat_pp,feat_sq10q "26" --drop "$D"
echo SHARE_DONE
