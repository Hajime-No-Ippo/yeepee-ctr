#!/bin/bash
cd /workspace/huawei/v19cut
for job in "A S 25 0" "A S 25 1" "A base 25 1"; do
  echo "== $job $(date +%T)"; ./run_cut.sh $job; /workspace/venv_v17/bin/python mkreadme.py
done
echo QUEUE3_DONE
