#!/bin/bash
cd /workspace/huawei/v8b/work; P=/workspace/venv/bin/python
export POLARS_MAX_THREADS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
w5(){ while for p in $(pgrep -f python); do [ "$(readlink /proc/$p/cwd 2>/dev/null)" = /workspace/huawei/v5 ] && echo $p; done | grep -q .; do sleep 5; done; }
T="taskset -c 0-3"
w5; echo "== $(date +%T) v2"; $T $P build_folds.py valall 2>&1 | grep -v Warn | grep -i "built\|error\|Trace" ; ls feat
w5; echo "== $(date +%T) v3"; $T $P v3fold.py f1 f2 f3 f4 f5 val23 val25 val26 2>&1 | grep -v Warn | grep "wrote\|rror"
echo "== $(date +%T) BV_DONE"
