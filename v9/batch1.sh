#!/bin/bash
cd /workspace/huawei/v9; B=feat_lr2,feat_tt,feat_pp
run(){ T=$1; shift; s=$(date +%s); ./scr.sh $T "$@"; echo "$T elapsed $(( $(date +%s)-s ))s"; /workspace/venv/bin/python cmp9.py $T; }
run tA $B,feat_tmA "25 26 23"
run tB $B,feat_tmB "25 23 26"
V9DROP='^tA_|^tB_' run noTT $B "25 23 26"
run agw8 $B "25 23 26" --agew 0.8
V9DROP='tmR_stale_h' run tB2 $B,feat_tmB,feat_tmB2 "25 23 26"
