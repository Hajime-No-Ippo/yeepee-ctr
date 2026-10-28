#!/bin/bash
cd /workspace/huawei/dw_lb/lc
export POLARS_MAX_THREADS=6 OMP_NUM_THREADS=6 PYTHONUNBUFFERED=1
SLOW='{"learning_rate":0.03,"feature_fraction":0.55,"num_leaves":95,"lambda_l2":30,"max_bin":63,"force_col_wise":true}'
L=feat_lr2,feat_tt,feat_pp,feat_sqC,feat_dw
for nf in 1 2 3 4; do
 for c in 25; do
  /workspace/venv/bin/python ens.py $c lc_nf$nf --seeds 0 --threads 6 --lr $L --negfrac 0.5 --nfolds $nf --rounds 500 --save 400,500 --params "$SLOW" --drop '^sq_(pdot|pcos|mdot|mcos)$' 2>&1 | grep -v Warn | tail -2
 done
done
echo LC_DONE
