#!/bin/bash
cd /workspace/huawei/v11
for c in 25 23 26; do /workspace/venv/bin/python mlp11.py $c mlp2 --ep 4 --seed 0 --lr feat_lr2,feat_tt,feat_pp,feat_sqC; done
echo B5_DONE
