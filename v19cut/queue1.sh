#!/bin/bash
cd /workspace/huawei/v19cut
for job in "B base 25 1" "B base 23 1" "B S 25 0" "B S 23 0" "B S 25 1" "B S 23 1" "B SP 25 0" "B SP 23 0" "B SP 25 1" "B SP 23 1"; do
  echo "== $job $(date +%T)"; ./run_cut.sh $job
done
echo QUEUE1_DONE
