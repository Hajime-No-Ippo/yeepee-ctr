#!/bin/bash
cd /workspace/huawei/v13
LP='"max_bin":63,"force_col_wise":true'; SLOW="\"learning_rate\":0.03,\"feature_fraction\":0.55,\"num_leaves\":95,\"lambda_l2\":30,$LP"
L=feat_lr2,feat_tt,feat_pp,feat_sqC,feat_dw
for c in 25 23; do ./e.sh aw_s "$c" 500 400,500 0.5 "{$SLOW}" $L --agew 0.7; done
echo B3_DONE
