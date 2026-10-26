#!/bin/bash
cd /workspace/huawei/v8; export SQ_EP=${SQ_EP:-4}
run(){ n=$1; shift; env "$@" SQ_OUT=/workspace/huawei/v8/feat_sqv/$n taskset -c 0-3 /workspace/venv/bin/python seq.py val25 val26 2>&1 | grep "fit C\|val2"; for c in 25 26; do /workspace/venv/bin/python sqeval.py feat_sqv/$n $c | head -1; done; }
run combo SQ_SLATE=1 SQ_NP=6 SQ_NTR=700000 SQ_EP=3
echo SQVAR2_DONE
