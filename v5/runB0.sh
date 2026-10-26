#!/bin/bash
cd /workspace/huawei/v5; source env.sh
until grep -q SCREENS_DONE out/runA3.out 2>/dev/null; do sleep 15; done
./guard.sh
grep -q PRECHECK_DONE out/slate_pre.log 2>/dev/null || { nice -n 19 $PY slate.py precheck > out/slate_pre.log 2>&1; echo PRECHECK_DONE >> out/slate_pre.log; }
