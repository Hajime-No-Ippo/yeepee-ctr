cd /workspace/huawei/v21graph; source /workspace/venv_v17/bin/activate
P="taskset -c 4-7 python gcn.py 5"
$P s1 --mon val25 --steps 1500 --every 250 --lr 2e-2 --thr 2 2>&1 | grep -v Warn > logs/s1.log
$P s2 --mon val25 --steps 1500 --every 250 --lr 2e-2 --L 3 --d 64 --thr 2 2>&1 | grep -v Warn > logs/s2.log
$P s3 --mon val25 --steps 1500 --every 250 --lr 2e-2 --l2 1e-3 --thr 2 2>&1 | grep -v Warn > logs/s3.log
$P s4 --mon val25 --steps 1500 --every 250 --lr 2e-2 --slate 0 --thr 2 2>&1 | grep -v Warn > logs/s4.log
