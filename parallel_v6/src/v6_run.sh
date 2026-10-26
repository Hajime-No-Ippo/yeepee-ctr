#!/bin/bash
# v6 = v4_lite config (2 base seeds + slow) + feat_lr2 (exact-SVD history-centroid cosines, k 4..16, + con05/con1), fixed feat_x (train-only Ug)
cd /workspace/huawei/v2work
P=/workspace/venv/bin/python
export POLARS_MAX_THREADS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 XDIR=feat_xf
w5(){ while for p in $(pgrep -f python); do [ "$(readlink /proc/$p/cwd 2>/dev/null)" = /workspace/huawei/v5 ] && echo $p; done | grep -q .; do sleep 1; done; }
st(){ w5; echo "=== $(date +%T) $*"; s=$SECONDS; "$@" 2>&1 | grep -v Warning | tail -3; echo "--- $(( SECONDS - s ))s"; }
SLOW='{"learning_rate":0.03,"feature_fraction":0.55,"num_leaves":95,"lambda_l2":30}'
[ -f feat_xf/eval.parquet ] || st env X_OUT=/workspace/huawei/v2work/feat_xf $P extra.py f1 f2 f3 f4 f5 f6 val25 val26 eval
for c in 25 26 27; do
  st $P ens.py $c v6base --seeds 0,1 --threads 4 --lr feat_lr2
  st $P ens.py $c v6slow --seeds 0 --rounds 500 --threads 4 --lr feat_lr2 --params "$SLOW"
done
echo V6DONE
