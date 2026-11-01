#!/bin/bash
# rund2.sh CORES THREADS CUT TAG SEEDS ARGS...
c=$1; nt=$2; cut=$3; tag=$4; seeds=$5; shift 5
MLP_THREADS=$nt OMP_NUM_THREADS=$nt taskset -c $c /workspace/venv_v17/bin/python /workspace/huawei/v21fit/div21.py $cut $tag --var A --seeds $seeds "$@" > /workspace/huawei/v21fit/logs/${tag}_A_c${cut}.out 2>&1
