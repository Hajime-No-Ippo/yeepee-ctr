#!/bin/bash
# usage: run_cut.sh VAR(A|B) SET(base|S|SP|SPC) CUT SEED   (4 threads pinned to cores 4-7)
export POLARS_MAX_THREADS=4 OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
V=$1; SET=$2; C=$3; SD=$4
cd /workspace/huawei/v12
BASE=feat_lr2,feat_tt,feat_pp,feat_sqC,feat_dw
D=""; [ "$SET" != base ] && D="--drop $(cat /workspace/huawei/v19cut/drop_${SET}_${V}.txt)"
taskset -c 4-7 /workspace/venv_v17/bin/python /workspace/huawei/v19cut/xlgb_cut.py $C ${V}_${SET} --seed $SD --threads 4 --lr $BASE,feat_${V}p,feat_N${V}s --rounds 1000 --save 1000 $D 2>&1 | grep -v Warn | tail -3
