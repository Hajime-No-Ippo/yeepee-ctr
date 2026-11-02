cd /workspace/huawei/v21graph; source /workspace/venv_v17/bin/activate
for k in 3 6; do mon=val$((k==3?23:26)); taskset -c 4-7 python -u gcn.py $k u1 --mon $mon --steps 400 --every 50 --lr 1e-2 --L 3 --d 64 --l2 3e-3 --save 250,300,350 --thr 2 > logs/u1_k$k.log 2>&1; done
