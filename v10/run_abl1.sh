#!/bin/bash
cd /workspace/huawei/v10
LR=feat_lr2,feat_tt,feat_pp,feat_sq10
bash scr.sh v10L  $LR "25 23 26" --drop '^sq_(?!logit$)'
bash scr.sh v10A  $LR "25 23 26"
bash scr.sh v10B  $LR "25 23 26" --drop '^sq_(pdot|pcos|mdot|mcos)$'
echo ABL1_DONE
