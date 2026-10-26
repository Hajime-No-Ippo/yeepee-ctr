#!/bin/bash
cd /workspace/huawei/v5; source env.sh
while pgrep -f "v5val.py base " >/dev/null; do sleep 15; done
./guard.sh
grep -q A0CHECK out/a0check.log 2>/dev/null || nice -n 19 $PY extra.py a0check > out/a0check.log 2>&1
./guard.sh
for n in f1 f2 f3 f4 f5 f6 val23 val25 val26 eval; do [ -f feat_x_fix/$n.parquet ] || FIX=1 OUT=/workspace/huawei/v5/feat_x_fix nice -n 19 $PY extra.py $n; done > out/build_fix.log 2>&1
./guard.sh
XDIRS=feat_x_fix,feat_x2 nice -n 19 $PY v5val.py base_fix A0 25 > out/base_fix.log 2>&1
echo DONE >> out/base_fix.log
