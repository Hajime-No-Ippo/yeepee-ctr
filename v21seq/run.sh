#!/bin/bash
# run.sh CORES CUT TAG VAR SEEDS ARGS...   (skips seeds whose output exists)
c=$1; cut=$2; tag=$3; var=$4; seeds=$5; shift 5
todo=""; for s in ${seeds//,/ }; do [ -f ens/${tag}_${var}_c${cut}_s${s}.npy ] || todo="$todo,$s"; done; todo=${todo#,}
[ -z "$todo" ] && exit 0
MLP_THREADS=4 OMP_NUM_THREADS=4 taskset -c $c /workspace/venv_v17/bin/python seq21.py $cut $tag --var $var --seeds $todo "$@" > logs/run_${tag}_${var}_c${cut}_$(echo $todo|tr ',' '_').out 2>&1
