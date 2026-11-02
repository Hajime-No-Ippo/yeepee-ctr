cd /workspace/huawei/v21graph; source /workspace/venv_v17/bin/activate
while pgrep -f "sweep6.sh" >/dev/null; do sleep 20; done
R="taskset -c 4-7 python -u gcn.py"
for s in 3 4 5 6 7 8 9 10; do $R 7 u1 --seed $s --steps 350 --lr 1e-2 --L 3 --d 64 --l2 3e-3 --save 350 --thr 2 > logs/u1_k7_s$s.log 2>&1; done
