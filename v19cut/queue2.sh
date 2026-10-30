#!/bin/bash
cd /workspace/huawei/v19cut
while pgrep -f queue1.sh >/dev/null; do sleep 15; done
for job in "B base 25 0" "B SPC 25 0" "B SPC 25 1" "B SPC 23 0" "B SPC 23 1" "B base 23 0"; do
  echo "== $job $(date +%T)"; ./run_cut.sh $job; /workspace/venv_v17/bin/python mkreadme.py
done
echo QUEUE2_DONE
