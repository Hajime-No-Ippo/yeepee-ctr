#!/bin/bash
# usage: sweep.sh cores name args...
c=$1; n=$2; shift 2
MLP_THREADS=4 OMP_NUM_THREADS=4 taskset -c $c /workspace/venv_v17/bin/python seq21.py 23 sw_$n --var A --seeds 0 --noseq "$@" > logs/sw_$n.out 2>&1
