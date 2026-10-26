#!/bin/bash
# v7 screen (ttfeat A/B on top of feat_lr2) + laptop-lite measurements. Waits for Code Buddy's v5 jobs.
cd /workspace/huawei/v2work
P=/workspace/venv/bin/python
export POLARS_MAX_THREADS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 XDIR=feat_xf TT_THREADS=4
w5(){ while for p in $(pgrep -f python); do [ "$(readlink /proc/$p/cwd 2>/dev/null)" = /workspace/huawei/v5 ] && echo $p; done | grep -q .; do sleep 5; done; }
st(){ w5; echo "=== $(date +%T) $*"; s=$SECONDS; "$@" 2>&1 | grep -v Warning | tail -${TL:-3} | cut -c1-400; echo "--- $(( SECONDS - s ))s"; }
TL=40 st $P ttfeat.py f1 f2 f3 f4 f5 f6 val25 val26 eval
[ -f feat_tt/val26.parquet ] || { echo TTFAIL; exit 1; }
for c in 25 26; do
  st $P screen.py $c v7_AB --lr feat_lr2,feat_tt --threads 4
  st $P screen.py $c v7_A --lr feat_lr2,feat_tt --drop '^tB_' --threads 4
  st $P screen.py $c v7_B --lr feat_lr2,feat_tt --drop '^tA_' --threads 4
done
for c in 25 26; do
  st $P screen.py $c gap_v7_AB --lr feat_lr2,feat_tt --maxfold 3 --threads 4
  st $P screen.py $c gap_v7_A --lr feat_lr2,feat_tt --drop '^tB_' --maxfold 3 --threads 4
  st $P screen.py $c gap_v7_B --lr feat_lr2,feat_tt --drop '^tA_' --maxfold 3 --threads 4
done
echo SCREEN_DONE
for c in 25 26; do
  st $P ens.py $c lite_nor --seeds 0 --rounds 400 --threads 4 --lr feat_lr2 --drop '^r_'
  st $P ens.py $c lite_nox2_f3 --seeds 0 --rounds 400 --threads 4 --lr feat_lr2 --drop '^(r_|s_)' --nfolds 3
done
echo Q_DONE
