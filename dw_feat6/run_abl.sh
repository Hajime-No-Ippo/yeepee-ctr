#!/bin/bash
cd /workspace/huawei/dw_feat6
while [ ! -f done.flag ]; do sleep 20; done
for a in "25 0 feat6n" "25 0 feat6g" "23 0 feat6n"; do
  set -- $a
  nice -n 19 taskset -c 0-3 /workspace/venv_v17/bin/python -u fs6a.py $1 $2 $3 > log_abl_$3_c$1_s$2.txt 2>&1
done
touch done_abl.flag
