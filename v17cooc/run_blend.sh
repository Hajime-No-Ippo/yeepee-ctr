#!/bin/bash
# Blend components retrained with co-occurrence features. B = TRANSDUCTIVE (primary), A = train-only (comparison).
# usage: run_blend.sh ; order: B c27, B c25, A c25, A c27   (dwbig:1 + dw_s:1 + 2*dwmlp x3 seeds, v12 recipe)
cd /workspace/huawei/v17cooc; P=/workspace/venv/bin/python; export POLARS_MAX_THREADS=8 OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
BASE=feat_lr2,feat_tt,feat_pp,feat_sqC,feat_dw
SLOW='{"learning_rate":0.03,"feature_fraction":0.55,"num_leaves":95,"lambda_l2":30,"max_bin":63,"force_col_wise":true}'
wait_idle(){ while pgrep -f "ens.py|xlgb|run_groups|mkdw|xens|ens17|mlp17|mlp11|mlp12|run_lgb" | grep -v $$ >/dev/null; do sleep 20; done; }
run(){ v=$1; c=$2; L=$BASE,feat_${v}p
  if [ "$3" = big ]; then wait_idle; $P xlgb17.py $c co${v}p --lr $L --rounds 1000 --save 1000 2>&1 | grep -v Warn | tail -1; fi
  wait_idle; $P ens17.py $c co${v}_s --seeds 0 --threads 8 --lr $L --negfrac 0.5 --nfolds 4 --rounds 500 --save 400,500 --params "$SLOW" --drop '^sq_(pdot|pcos|mdot|mcos)$' 2>&1 | grep -v Warn | tail -1
  for s in 0 1 2; do wait_idle; $P mlp17.py $c co${v}mlp --ep 4 --seed $s --lr $L 2>&1 | grep -v Warn | tail -1; done; }
while ! grep -q LGB_DONE logs/run_lgb.out; do sleep 30; done
run B 27 big; run B 25; run A 25; run A 27 big
echo BLEND_DONE
