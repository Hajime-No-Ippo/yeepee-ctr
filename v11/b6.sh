#!/bin/bash
cd /workspace/huawei/v11
L=feat_lr2,feat_tt,feat_pp,feat_sqC
for s in 1 2; do for c in 25 23; do /workspace/venv/bin/python mlp11.py $c mlp2 --ep 4 --seed $s --lr $L; done; done
for c in 25 23; do /workspace/venv/bin/python mlp11.py $c mlp3 --ep 7 --seed 0 --lr $L; done
echo B6_DONE
