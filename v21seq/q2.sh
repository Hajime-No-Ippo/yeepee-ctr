export MLP_THREADS=4 OMP_NUM_THREADS=4
for c in 23 25; do taskset -c 4-7 /workspace/venv_v17/bin/python seq21.py $c v21seq_big --var A --seeds 1,2 --ep 6 --elr 0 --enc trf --aug 0.3 --hd 128 --tok 256 --alpha 0.3 > logs/q2_$c.out 2>&1; done
