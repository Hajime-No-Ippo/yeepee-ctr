#!/bin/bash
# run.sh CORES THREADS CUT CFGNAME  (cfg/CFGNAME.json) -> logs/CFGNAME_c{CUT}.out
c=$1; nt=$2; cut=$3; cfg=$4
MLP_THREADS=$nt OMP_NUM_THREADS=$nt taskset -c $c nohup /workspace/venv_v17/bin/python -W ignore /workspace/huawei/v21dcn/dcn21.py $cut x --var A --cfg /workspace/huawei/v21dcn/cfg/$cfg.json > /workspace/huawei/v21dcn/logs/${cfg}_c${cut}.out 2>&1 &
