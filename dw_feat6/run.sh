#!/bin/bash
cd /workspace/huawei/dw_feat6
PY=/workspace/huawei/v20/.venv/bin/python
for sc in "25 0" "23 0" "25 1" "23 1"; do set -- $sc; [ -f out/f6_c$1_s$2.npy ] && continue
  nice -n 19 taskset -c 0-3 $PY -u fs6.py $1 $2 > log_c$1_s$2.txt 2>&1; done
echo done > done.flag
