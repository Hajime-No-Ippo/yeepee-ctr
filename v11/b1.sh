#!/bin/bash
cd /workspace/huawei/v11
LP='"max_bin":63,"force_col_wise":true'
# reproduce v10B b/s with 8 threads (sanity) then capacity variants
./e.sh nf1_b "25 23" 400 300,400 1.0 "{$LP}"
./e.sh nf1_s "25 23" 500 400,500 1.0 "{\"learning_rate\":0.03,\"feature_fraction\":0.55,\"num_leaves\":95,\"lambda_l2\":30,$LP}"
./e.sh big1 "25 23" 3000 1000,1500,2000,2500,3000 1.0 '{"learning_rate":0.015,"num_leaves":127,"feature_fraction":0.5,"lambda_l2":30,"min_data_in_leaf":200,"max_bin":255,"force_col_wise":true}'
echo B1_DONE
