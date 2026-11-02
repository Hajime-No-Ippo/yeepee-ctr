cd /workspace/huawei/v21graph; source /workspace/venv_v17/bin/activate
while pgrep -f "gcn.py 6 u1" >/dev/null; do sleep 5; done
taskset -c 4-7 python -u gcn.py 3 u1b --mon val23 --steps 800 --every 50 --lr 1e-2 --L 3 --d 64 --l2 3e-3 --save 400,500,600,700,800 --thr 2 > logs/u1b_k3.log 2>&1
