#!/bin/bash
# usage: runq.sh jobsfile ; runs 2 parallel jobs (2 threads each) on cores 4-7
cd /workspace/huawei/v21graph2; source /workspace/venv_v17/bin/activate
run(){ set -- $@; k=$1; tag=$2; sd=$(echo "$@" | sed -n 's/.*--seed \([0-9]*\).*/\1/p'); taskset -c 4-7 python -u gcn2.py "$@" --thr 2 > logs/${tag}_k${k}_s${sd}.log 2>&1; }
export -f run
cat $1 | xargs -P2 -I{} bash -c 'run {}'
