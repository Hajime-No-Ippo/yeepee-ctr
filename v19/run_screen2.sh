#!/bin/bash
export POLARS_MAX_THREADS=8 OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1; cd /workspace/huawei/v12
BASE=feat_lr2,feat_tt,feat_pp,feat_sqC,feat_dw
for c in 25 23; do for v in A B; do
 while pgrep -f "xlgb|xpair|ens17|mlp17|mkmf|mkcooc" | grep -v $$ >/dev/null; do sleep 20; done
 /workspace/venv/bin/python /workspace/huawei/v19/xlgb17.py $c nm${v}s --lr $BASE,feat_${v}p,feat_N${v}s --rounds 1000 --save 1000 2>&1 | grep -v Warn | tail -2
done; done
echo SCREEN2_DONE
