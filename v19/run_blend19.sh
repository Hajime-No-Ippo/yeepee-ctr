#!/bin/bash
# usage: run_blend19.sh CUT VARIANTS(e.g. "B A") [big]  -> dw_s + MLP x3 (+dwbig if 'big') with v17cooc feats + pruned NMF feats (feat_N{v}s)
cd /workspace/huawei/v19; P=/workspace/venv/bin/python; export POLARS_MAX_THREADS=8 OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
BASE=feat_lr2,feat_tt,feat_pp,feat_sqC,feat_dw
SLOW='{"learning_rate":0.03,"feature_fraction":0.55,"num_leaves":95,"lambda_l2":30,"max_bin":63,"force_col_wise":true}'
wait_idle(){ while pgrep -f "xlgb|xpair|ens17|mlp17|mlp_pair|mkmf|mkcooc" | grep -v $$ >/dev/null; do sleep 15; done; }
c=$1
for v in $2; do L=$BASE,feat_${v}p,feat_N${v}s
  if [ "$3" = big ]; then wait_idle; $P xlgb17.py $c nm${v}s --lr $L --rounds 1000 --save 1000 2>&1 | grep -v Warn | tail -1; fi
  wait_idle; $P ens17.py $c nm${v}s_s --seeds 0 --threads 8 --lr $L --negfrac 0.5 --nfolds 4 --rounds 500 --save 400,500 --params "$SLOW" --drop '^sq_(pdot|pcos|mdot|mcos)$' 2>&1 | grep -v Warn | tail -1
  for s in 0 1 2; do wait_idle; $P mlp17.py $c nm${v}smlp --ep 4 --seed $s --lr $L 2>&1 | grep -v Warn | tail -1; done
done
echo BLEND_DONE_$c
