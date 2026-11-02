cd /workspace/huawei/v21graph; source /workspace/venv_v17/bin/activate
R="taskset -c 4-7 python -u gcn.py"
for s in 1 2; do
 $R 5 u1 --seed $s --steps 350 --lr 1e-2 --L 3 --d 64 --l2 3e-3 --save 350 --thr 2 > logs/u1_k5_s$s.log 2>&1
 $R 6 u1 --seed $s --steps 350 --lr 1e-2 --L 3 --d 64 --l2 3e-3 --save 350 --thr 2 > logs/u1_k6_s$s.log 2>&1
 $R 3 u1b --seed $s --steps 800 --lr 1e-2 --L 3 --d 64 --l2 3e-3 --save 800 --thr 2 > logs/u1b_k3_s$s.log 2>&1
done
for s in 0 1 2; do $R 7 u1 --seed $s --steps 350 --lr 1e-2 --L 3 --d 64 --l2 3e-3 --save 350 --thr 2 > logs/u1_k7_s$s.log 2>&1; done
