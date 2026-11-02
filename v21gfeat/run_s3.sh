#!/bin/bash
cd /workspace/huawei/v21gfeat; export GTAG=_s3
for c in 25 23 26; do for v in numg num; do
  [ -f ens/gf_${v}_s3_c${c}_s0.npy ] || taskset -c 4-7 /workspace/venv_v17/bin/python -u train_g.py $c $v --threads 2 > logs/t_${v}_s3_c${c}.log 2>&1
done; done; echo DONE > logs/run_s3.done
