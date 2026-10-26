#!/bin/bash
cd /workspace/huawei/v8; P=/workspace/venv/bin/python
export POLARS_MAX_THREADS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 TT_THREADS=4 PYTHONUNBUFFERED=1
w5(){ while for p in $(pgrep -f python); do [ "$(readlink /proc/$p/cwd 2>/dev/null)" = /workspace/huawei/v5 ] && echo $p; done | grep -q .; do sleep 5; done; }
T=taskset\ -c\ 0-3
w5; echo "== $(date +%T) v3"; $T $P v3fold.py val23
w5; echo "== $(date +%T) v2"; $T $P build_folds.py c23
w5; echo "== $(date +%T) x"; $T $P extra.py val23
w5; echo "== $(date +%T) lr"; LR_KS=4,6,8,10,12,16 LR_CON=1 $T $P lrfeat2.py val23
w5; echo "== $(date +%T) tt"; $T $P ttfeat.py val23
echo "== $(date +%T) B23_DONE"
