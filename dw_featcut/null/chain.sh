#!/bin/bash
cd /workspace/huawei/v12; export POLARS_MAX_THREADS=4 OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
N=/workspace/huawei/dw_featcut/null
taskset -c 4-7 /workspace/venv_v17/bin/python $N/run_val.py 25 0,1 $N/sets.json
taskset -c 4-7 /workspace/venv_v17/bin/python $N/run_val.py 23 0,1 $N/sets.json
echo CHAIN_DONE
