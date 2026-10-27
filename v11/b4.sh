#!/bin/bash
cd /workspace/huawei/v11
LP='"max_bin":63,"force_col_wise":true'; SLOW="\"learning_rate\":0.03,\"feature_fraction\":0.55,\"num_leaves\":95,\"lambda_l2\":30,$LP"
BIG='{"learning_rate":0.015,"num_leaves":127,"feature_fraction":0.5,"lambda_l2":30,"min_data_in_leaf":200,"max_bin":255,"force_col_wise":true}'
L=feat_lr2,feat_tt,feat_pp,feat_sqC
./e.sh sC_s 27 500 400,500 0.5 "{$SLOW}" $L
./e.sh sCbig 27 1200 800,1000,1200 1.0 "$BIG" $L
echo B4_DONE
