cd /workspace/huawei/v21diag
while [ ! -f logs/run4_done ]; do sleep 30; done
L=feat_lr2,feat_tt,feat_pp,feat_sqC,feat_dw,feat_Bp,feat_NBs
export MLP_THREADS=4 OMP_NUM_THREADS=4
P="taskset -c 4-7 /workspace/venv_v17/bin/python lr21.py"
$P 25 v21lrB --ep 3 --seed 0 --loss bce --wd 1e-2 --lr $L > logs/lr21_c25.out 2>&1
$P 23 v21lrB --ep 3 --seed 0 --loss bce --wd 1e-2 --lr $L > logs/lr21_c23.out 2>&1
touch logs/run5_done
