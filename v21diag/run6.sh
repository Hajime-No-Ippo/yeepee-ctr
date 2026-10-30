cd /workspace/huawei/v21diag
while [ ! -f logs/run5_done ]; do sleep 20; done
export MLP_THREADS=4 OMP_NUM_THREADS=4
P="taskset -c 4-7 /workspace/venv_v17/bin/python mlp21list.py"
LB=feat_lr2,feat_tt,feat_pp,feat_sqC,feat_dw,feat_Bp,feat_NBs
LA=feat_lr2,feat_tt,feat_pp,feat_sqC,feat_dw,feat_Ap,feat_NAs
$P 25 v21listA --ep 4 --seed 0 --loss list --lr $LA > logs/mlp21listA_c25.out 2>&1
$P 23 v21listA --ep 4 --seed 0 --loss list --lr $LA > logs/mlp21listA_c23.out 2>&1
for s in 0 1 2; do $P 27 v21listB --ep 4 --seed $s --loss list --lr $LB > logs/mlp21listB_c27_s$s.out 2>&1; done
for s in 0 1 2; do $P 27 v21listA --ep 4 --seed $s --loss list --lr $LA > logs/mlp21listA_c27_s$s.out 2>&1; done
touch logs/run6_done
