#!/bin/bash
cd /workspace/huawei/v2work
P=/workspace/venv/bin/python
export POLARS_MAX_THREADS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 XDIR=feat_xf
w5(){ while for p in $(pgrep -f python); do [ "$(readlink /proc/$p/cwd 2>/dev/null)" = /workspace/huawei/v5 ] && echo $p; done | grep -q .; do sleep 5; done; }
st(){ w5; echo "=== $(date +%T) $*"; s=$SECONDS; "$@" 2>&1 | grep -v Warning | tail -1 | cut -c1-300; echo "--- $(( SECONDS - s ))s"; }
while pgrep -f "v7_q2.sh|v7_q3.sh" >/dev/null; do sleep 10; done
for c in 25 26; do
  st $P screen.py $c xAn_lr2 --lr feat_lr2,feat_xAn --threads 4
  st $P screen.py $c xAn_v7 --lr feat_lr2,feat_tt,feat_xAn --threads 4
done
for c in 25 26; do
  st $P screen.py $c gap_xAn_lr2 --lr feat_lr2,feat_xAn --maxfold 3 --threads 4
  st $P screen.py $c gap_xAn_v7 --lr feat_lr2,feat_tt,feat_xAn --maxfold 3 --threads 4
done
echo Q4_DONE
