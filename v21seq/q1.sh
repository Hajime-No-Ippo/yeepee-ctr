export MLP_THREADS=4 OMP_NUM_THREADS=4
for c in 23 25; do taskset -c 0-3 /workspace/venv_v17/bin/python seq21.py $c ctl6 --var A --seeds 1,2 --ep 6 --noseq --alpha 0.3 > logs/q1_$c.out 2>&1; done
