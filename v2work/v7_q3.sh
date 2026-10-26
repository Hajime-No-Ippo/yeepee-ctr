#!/bin/bash
cd /workspace/huawei/v2work
P=/workspace/venv/bin/python
export POLARS_MAX_THREADS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 XDIR=feat_xf TT_THREADS=4
w5(){ while for p in $(pgrep -f python); do [ "$(readlink /proc/$p/cwd 2>/dev/null)" = /workspace/huawei/v5 ] && echo $p; done | grep -q .; do sleep 5; done; }
st(){ w5; echo "=== $(date +%T) $*"; s=$SECONDS; "$@" 2>&1 | grep -v Warning | tail -${TL:-3} | cut -c1-400; echo "--- $(( SECONDS - s ))s"; }
while pgrep -f v7_q2.sh >/dev/null; do sleep 10; done
mkdir -p /tmp/lrtime
TL=10 st env LR_OUT=/tmp/lrtime LR_KS=4,6,8,10,12,16 LR_CON=1 $P lrfeat2.py f1 f2 f3 f4 f5 f6 eval
L=feat_lr2,feat_tt
SLOW='{"learning_rate":0.03,"feature_fraction":0.55,"num_leaves":95,"lambda_l2":30,"max_bin":63,"force_col_wise":true}'
for c in 25 26; do
  st $P ens.py $c lt_cmb_base --seeds 0 --threads 4 --lr $L --negfrac 0.5 --nfolds 4 --params '{"max_bin":63,"force_col_wise":true}'
  st $P ens.py $c lt_cmb_slow --seeds 0 --rounds 500 --threads 4 --lr $L --negfrac 0.5 --nfolds 4 --params "$SLOW"
done
echo Q3_DONE
