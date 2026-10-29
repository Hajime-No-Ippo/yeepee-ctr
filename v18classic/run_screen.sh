#!/bin/bash
# dwbig-recipe LightGBM (xlgb18.py) on v17cooc-A features + one v18 family (or all). baseline = v17cooc coAp_c{cut} (stored)
cd /workspace/huawei/v12; export POLARS_MAX_THREADS=8 OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
BASE=feat_lr2,feat_tt,feat_pp,feat_sqC,feat_dw,feat_Ap
for c in ${CUTS:-25 23 26}; do for fam in ${FAMS:-fr au uc ty all}; do
 if [ $fam = all ]; then X=feat_fr,feat_au,feat_uc,feat_ty; else X=feat_$fam; fi
 while pgrep -f "ens.py|xlgb|run_groups|mkdw|xens|ens17|ens18|mlp17|mlp18" | grep -v $$ >/dev/null; do sleep 20; done
 /workspace/venv/bin/python /workspace/huawei/v18classic/xlgb18.py $c v18$fam --lr $BASE,$X --rounds 1000 --save 1000 2>&1 | grep -v Warn | tail -2
done; done
echo SCREEN_DONE
