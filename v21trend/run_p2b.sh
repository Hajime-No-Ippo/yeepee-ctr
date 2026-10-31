#!/bin/bash
cd /workspace/huawei/v21trend; source /workspace/venv_v17/bin/activate
for v in coA_cl coA_co2 coA_dec coA_cs; do taskset -c 4-7 python rollval_t.py fit --variant $v --threads 4 >> logs/p2b_stdout.log 2>&1; done
echo P2B_FIT_DONE >> logs/p2b_stdout.log
