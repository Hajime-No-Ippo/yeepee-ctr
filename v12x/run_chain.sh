#!/bin/bash
cd /workspace/huawei/v12x
./run_q4b.sh > logs_q4b.log 2>&1
./run_q3.sh > logs_q3.log 2>&1
echo CHAIN_DONE >> logs_q3.log
