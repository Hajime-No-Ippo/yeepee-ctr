cd /workspace/huawei/v21graph; source /workspace/venv_v17/bin/activate
for k in 3 5; do taskset -c 4-7 python gcn.py $k t1 --mon $([ $k = 3 ] && echo val23 || echo val25) --steps 1500 --every 500 --save 600,1000,1500 --thr 2 2>&1 | grep -v Warn > logs/t1_k$k.log; done
