#!/bin/bash
cd /workspace/huawei/v2work
P=/workspace/venv/bin/python
run(){ echo "=== $(date +%T) $*"; $P screen.py "$@" --threads 4 2>&1 | grep -v Warning | tail -3; }
for c in 25 26; do
 run $c i_lr12 --lr feat_lr --drop '^lr1[06]_'
 run $c gap_lr12 --lr feat_lr --drop '^lr1[06]_' --maxfold 3
done
for c in 25 26; do run $c i_lrall --lr feat_lr; run $c i_lr10 --lr feat_lr --drop '^lr1[26]_'; done
echo DONE
