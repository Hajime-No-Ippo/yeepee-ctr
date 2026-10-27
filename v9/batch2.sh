#!/bin/bash
cd /workspace/huawei/v9; B=feat_lr2,feat_tt,feat_pp
run(){ T=$1; shift; s=$(date +%s); ./scr.sh $T "$@"; echo "$T elapsed $(( $(date +%s)-s ))s"; /workspace/venv/bin/python cmp9.py $T; }
V9SEED=1000 run trimr $B "25 23 26"
run tAs $B,feat_tmAs "25 23 26"
