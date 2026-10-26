#!/bin/bash
cd /workspace/huawei/v2work
P=/workspace/venv/bin/python
export POLARS_MAX_THREADS=4 OMP_NUM_THREADS=4
w5(){ while for p in $(pgrep -f python); do [ "$(readlink /proc/$p/cwd 2>/dev/null)" = /workspace/huawei/v5 ] && echo $p; done | grep -q .; do sleep 5; done; }
while pgrep -f v7_q4.sh >/dev/null; do sleep 10; done
for c in 25 26; do w5; echo "=== $(date +%T) $c"; $P ens.py $c lt_cmb_nox2 --seeds 0 --threads 4 --lr feat_lr2,feat_tt --negfrac 0.5 --nfolds 4 --params '{"max_bin":63,"force_col_wise":true}' --drop '^(r|s)_' 2>&1 | grep -v Warn | tail -1; done
echo Q5_DONE
