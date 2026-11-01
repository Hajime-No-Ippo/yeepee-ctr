#!/bin/bash
# rund.sh CORES CUT TAG VAR SEEDS TEACHER ARGS...
c=$1; cut=$2; tag=$3; var=$4; seeds=$5; teach=$6; shift 6
MLP_THREADS=${NT:-4} OMP_NUM_THREADS=${NT:-4} taskset -c $c /workspace/venv_v17/bin/python /workspace/huawei/v21fit/dist21.py $cut $tag --var $var --seeds $seeds --teacher $teach "$@" > /workspace/huawei/v21fit/logs/${tag}_${var}_c${cut}.out 2>&1
