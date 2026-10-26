#!/bin/bash
cd /workspace/huawei/v2work
P=/workspace/venv/bin/python
run(){ echo "=== $(date +%T) $*"; $P screen.py "$@" --threads 4 2>&1 | grep -v Warning | tail -1 | cut -c1-200; }
for c in 25 26; do
 run $c i_lr2all --lr feat_lr2
 run $c i_lr2_1012c --lr feat_lr2 --drop '^lr(4|6|8|16)_|_con1'
 run $c i_lr2_81012 --lr feat_lr2 --drop '^lr(4|6|16)_|_con'
done
for c in 25 26; do run $c gap_lr2all --lr feat_lr2 --maxfold 3; run $c gap_lrall --lr feat_lr --maxfold 3; done
echo DONE
