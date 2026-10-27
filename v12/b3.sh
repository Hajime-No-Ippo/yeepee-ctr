#!/bin/bash
cd /workspace/huawei/v12; export SEEDS=1
LP='"max_bin":63,"force_col_wise":true'; SLOW="\"learning_rate\":0.03,\"feature_fraction\":0.55,\"num_leaves\":95,\"lambda_l2\":30,$LP"
BIG='{"learning_rate":0.015,"num_leaves":127,"feature_fraction":0.5,"lambda_l2":30,"min_data_in_leaf":200,"max_bin":255,"force_col_wise":true}'
LD=feat_lr2,feat_tt,feat_pp,feat_sqC,feat_dw; L0=feat_lr2,feat_tt,feat_pp,feat_sqC
for c in 25 23; do
./e.sh dw_s "$c" 500 400,500 0.5 "{$SLOW}" $LD
./e.sh dwbig "$c" 1200 800,1000,1200 1.0 "$BIG" $LD
./e.sh sC_s "$c" 500 400,500 0.5 "{$SLOW}" $L0
./e.sh sCbig "$c" 1200 800,1000,1200 1.0 "$BIG" $L0
done
echo B3_DONE
