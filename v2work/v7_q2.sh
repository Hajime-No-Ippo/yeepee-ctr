#!/bin/bash
cd /workspace/huawei/v2work
P=/workspace/venv/bin/python
export POLARS_MAX_THREADS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 XDIR=feat_xf TT_THREADS=4
w5(){ while for p in $(pgrep -f python); do [ "$(readlink /proc/$p/cwd 2>/dev/null)" = /workspace/huawei/v5 ] && echo $p; done | grep -q .; do sleep 5; done; }
st(){ w5; echo "=== $(date +%T) $*"; s=$SECONDS; "$@" 2>&1 | grep -v Warning | tail -${TL:-3} | cut -c1-400; echo "--- $(( SECONDS - s ))s"; }
while pgrep -f "screen.py 25 v7_B" >/dev/null; do sleep 5; done
L=feat_lr2,feat_tt
st $P screen.py 26 v7_AB --lr $L --threads 4
st $P screen.py 26 v7_A --lr $L --drop '^tB_' --threads 4
st $P screen.py 26 v7_B --lr $L --drop '^tA_' --threads 4
for c in 25 26; do
  st $P screen.py $c gap_v7_AB --lr $L --maxfold 3 --threads 4
  st $P screen.py $c gap_v7_A --lr $L --drop '^tB_' --maxfold 3 --threads 4
  st $P screen.py $c gap_v7_B --lr $L --drop '^tA_' --maxfold 3 --threads 4
done
echo SCREEN_DONE
$P passcheck.py; rc=$?; [ $rc -ne 0 ] && { echo NO_PASS; exit 0; }
SLOW='{"learning_rate":0.03,"feature_fraction":0.55,"num_leaves":95,"lambda_l2":30}'
for c in 25 26 27; do
  st $P ens.py $c v7base --seeds 0,1 --threads 4 --lr $L
  st $P ens.py $c v7slow --seeds 0 --rounds 500 --threads 4 --lr $L --params "$SLOW"
done
TL=6 st env V6PFX=v7 EXD=$L $P lite_build/blend_v7.py val
TL=6 st env V6PFX=v7 EXD=$L PRED_OUT=eval_pred_v7.parquet $P lite_build/blend_v7.py eval
echo V7DONE
for c in 25 26; do
  $P -c "import lightgbm as lgb,numpy as np;m=lgb.Booster(model_file='ens/v7base_c${c}_s0.txt');g=m.feature_importance('gain');n=m.feature_name();open('keep175_c${c}.txt','w').write('\n'.join(n[i] for i in np.argsort(-g)[:175]))"
  st $P ens.py $c lt_neg50 --seeds 0 --threads 4 --lr $L --negfrac 0.5
  st $P ens.py $c lt_f3 --seeds 0 --threads 4 --lr $L --nfolds 3
  st $P ens.py $c lt_top175 --seeds 0 --threads 4 --lr $L --keep keep175_c${c}.txt
  st $P ens.py $c lt_bin63 --seeds 0 --threads 4 --lr $L --params '{"max_bin":63,"force_col_wise":true}'
done
echo LITE_DONE
