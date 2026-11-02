#!/bin/bash
cd /workspace/huawei/v21gfeat
while [ ! -f out/emb_u1b_k1_s2.npz ]; do sleep 10; done
sleep 8; pkill -f sweep_train_emb.sh; pkill -f gcn_k.py; sleep 2
export GSEEDS=0,1,2 GTAG=_s3
taskset -c 4-7 /workspace/venv_v17/bin/python mkgf.py f1 f2 f3 f4 f5 f6 val23 val25 val26 > logs/mkgf_s3.log 2>&1
echo MKGF_DONE >> logs/mkgf_s3.log
