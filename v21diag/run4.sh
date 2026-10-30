cd /workspace/huawei/v21diag
while pgrep -f "queue[123].sh" >/dev/null || pgrep -f xlgb_cut >/dev/null; do sleep 20; done
L=feat_lr2,feat_tt,feat_pp,feat_sqC,feat_dw,feat_Bp,feat_NBs
export MLP_THREADS=4 OMP_NUM_THREADS=4
P="taskset -c 4-7 /workspace/venv_v17/bin/python mlp21list.py"
$P 25 v21listB --ep 4 --seed 1 --loss list --lr $L > logs/mlp21list_c25_s1.out 2>&1
$P 25 v21bceB --ep 4 --seed 0 --loss bce --lr $L > logs/mlp21bce_c25.out 2>&1
$P 23 v21listB --ep 4 --seed 1 --loss list --lr $L > logs/mlp21list_c23_s1.out 2>&1
$P 25 v21listB --ep 4 --seed 2 --loss list --lr $L > logs/mlp21list_c25_s2.out 2>&1
$P 23 v21listB --ep 4 --seed 2 --loss list --lr $L > logs/mlp21list_c23_s2.out 2>&1
touch logs/run4_done
