#!/bin/bash
# dwbig-recipe LightGBM (xlgb17.py = v12x/xlgb.py) with co-occurrence features. A = train-only, B = TRANSDUCTIVE (eval histories in graph)
cd /workspace/huawei/v12; export POLARS_MAX_THREADS=8 OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
BASE=feat_lr2,feat_tt,feat_pp,feat_sqC,feat_dw
for c in 25 23 26; do for v in A B; do
 while pgrep -f "ens.py|xlgb|run_groups|mkdw|xens" | grep -v $$ >/dev/null; do sleep 20; done
 /workspace/venv/bin/python /workspace/huawei/v17cooc/xlgb17.py $c co${v}p --lr $BASE,feat_${v}p --rounds 1000 --save 1000 2>&1 | grep -v Warn | tail -2
done; done
echo LGB_DONE
