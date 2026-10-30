#!/bin/bash
# usage: run_cut.sh VAR(A|B) SET(base|S|SP|SPC) CUT SEED
export POLARS_MAX_THREADS=8 OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
V=$1; SET=$2; C=$3; SD=$4
cd /workspace/huawei/v12
while [ ! -e /workspace/huawei/v21diag/QUICK_DONE ] || pgrep -f "v17run.py|ens.py|xlgb|xpair|ens17|mlp17|mkmf|mkcooc|blend19|mkprobe|prefilter|v20/|\.venv/bin" >/dev/null; do sleep 20; done
BASE=feat_lr2,feat_tt,feat_pp,feat_sqC,feat_dw
D=""; [ "$SET" != base ] && D="--drop $(cat /workspace/huawei/v19cut/drop_${SET}_${V}.txt)"
/workspace/venv_v17/bin/python /workspace/huawei/v19cut/xlgb_cut.py $C ${V}_${SET} --seed $SD --lr $BASE,feat_${V}p,feat_N${V}s --rounds 1000 --save 1000 $D 2>&1 | grep -v Warn | tail -3
