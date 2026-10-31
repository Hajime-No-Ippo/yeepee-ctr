#!/bin/bash
cd /workspace/huawei/v21trend; source /workspace/venv_v17/bin/activate
while ! grep -q SCREEN_DONE logs/fit_stdout.log; do sleep 10; done
export OMP_NUM_THREADS=4 POLARS_MAX_THREADS=4
taskset -c 4-7 python score3.py --names coA,coA_trI,coA_trU,coA_trIU,coA_s1 --ref coA --boot 200 --seg 0 --tag p1 > logs/score_p1.log 2>&1
taskset -c 4-7 python mkphase2.py 1 2 3 4 5 6 7 > logs/mk2.log 2>&1
taskset -c 4-7 python mkdecay.py > logs/mkdec.log 2>&1
echo FEATS_DONE >> logs/p2_stdout.log
for v in coA_dec coA_cs coA_co2 coA_cl; do taskset -c 4-7 python rollval_t.py fit --variant $v --threads 4 >> logs/p2_stdout.log 2>&1; done
echo P2_FIT_DONE >> logs/p2_stdout.log
