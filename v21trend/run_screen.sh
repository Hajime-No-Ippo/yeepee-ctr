#!/bin/bash
cd /workspace/huawei/v21trend; source /workspace/venv_v17/bin/activate
for v in coA_trI coA_trU coA_trIU; do taskset -c 4-7 python rollval_t.py fit --variant $v --threads 4 >> logs/fit_stdout.log 2>&1; done
taskset -c 4-7 python rollval_t.py fit --variant coA --seed 1 --threads 4 >> logs/fit_stdout.log 2>&1
echo SCREEN_DONE >> logs/fit_stdout.log
