#!/bin/bash
cd /workspace/huawei/v13
LP='"max_bin":63,"force_col_wise":true'; SLOW="\"learning_rate\":0.03,\"feature_fraction\":0.55,\"num_leaves\":95,\"lambda_l2\":30,$LP"
BIG='{"learning_rate":0.015,"num_leaves":127,"feature_fraction":0.5,"lambda_l2":30,"min_data_in_leaf":200,"max_bin":255,"force_col_wise":true}'
L=feat_lr2,feat_tt,feat_pp,feat_sqC,feat_dw,feat_ev
for c in ${CUTS:-25 23}; do ./e.sh ev_s "$c" 500 400,500 0.5 "{$SLOW}" $L; done
echo B1_DONE
