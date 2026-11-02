cd /workspace/huawei/v21graph; source /workspace/venv_v17/bin/activate
P="taskset -c 4-7 python -u gcn.py 5"
$P u1 --mon val25 --steps 700 --every 50 --lr 1e-2 --L 3 --d 64 --l2 3e-3 --save 150,250,350,500,700 --thr 2 > logs/u1.log 2>&1
$P u2 --mon val25 --steps 700 --every 50 --lr 1e-2 --L 3 --d 64 --l2 3e-3 --slate 3 --bpr 0.5 --drop 0.2 --save 150,250,350,500,700 --thr 2 > logs/u2.log 2>&1
