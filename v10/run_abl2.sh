#!/bin/bash
cd /workspace/huawei/v10
while pgrep -f "seq10.py val25" >/dev/null; do sleep 10; done
bash scr.sh v10C feat_lr2,feat_tt,feat_pp,feat_sq10c "25 23 26" --drop '^sq_(pdot|pcos|mdot|mcos)$'
V9SEED=1 bash scr.sh v10Bs1 feat_lr2,feat_tt,feat_pp,feat_sq10 "25 23" --drop '^sq_(pdot|pcos|mdot|mcos)$'
V9SEED=1 bash scr.sh trims1 feat_lr2,feat_tt,feat_pp "25 23"
echo ABL2_DONE
