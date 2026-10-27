#!/bin/bash
# usage: scr.sh TAG LRLIST "CUTS" [extra ens args]
cd /workspace/huawei/v10; P=/workspace/venv/bin/python; TAG=$1; LR=$2; CUTS=$3; shift 3
export POLARS_MAX_THREADS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
w5(){ while for p in $(pgrep -f python); do [ "$(readlink /proc/$p/cwd 2>/dev/null)" = /workspace/huawei/v5 ] && echo $p; done | grep -q .; do sleep 5; done; }
LP='"max_bin":63,"force_col_wise":true'; SLOW='"learning_rate":0.03,"feature_fraction":0.55,"num_leaves":95,"lambda_l2":30'
for c in $CUTS; do
  w5; taskset -c 0-3 $P ens.py $c ${TAG}_b --seeds 0 --threads 4 --lr $LR --negfrac 0.5 --nfolds 4 --params "{$LP}" "$@" 2>&1 | grep -v Warn | tail -1
  w5; taskset -c 0-3 $P ens.py $c ${TAG}_s --seeds 0 --rounds 500 --threads 4 --lr $LR --negfrac 0.5 --nfolds 4 --params "{$SLOW,$LP}" "$@" 2>&1 | grep -v Warn | tail -1
  $P blendv.py $TAG $c
done
