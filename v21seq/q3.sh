export MLP_THREADS=4 OMP_NUM_THREADS=4
taskset -c 0-3 /workspace/venv_v17/bin/python seq21.py 27 ctl6 --var A --seeds 0,1,2 --ep 6 --noseq --alpha 0.3 > logs/q3_27ctl.out 2>&1
