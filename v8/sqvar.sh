#!/bin/bash
# variant screen: name envs...  -> feat_sqv/NAME/val25,val26 + sqeval
cd /workspace/huawei/v8; export SQ_EP=${SQ_EP:-4}
run(){ n=$1; shift; env "$@" SQ_OUT=/workspace/huawei/v8/feat_sqv/$n taskset -c 0-3 /workspace/venv/bin/python seq.py val25 val26 2>&1 | grep "fit C\|val2"; for c in 25 26; do /workspace/venv/bin/python sqeval.py feat_sqv/$n $c | head -1; done; }
run np6 SQ_NP=6
run slate SQ_SLATE=1
run all SQ_NTR=700000 SQ_EP=3
echo SQVAR_DONE
