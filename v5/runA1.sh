#!/bin/bash
cd /workspace/huawei/v5; source env.sh
while pgrep -f "runA0.sh" >/dev/null; do sleep 15; done
./guard.sh
grep -q A0CHECK out/a0check.log 2>/dev/null || nice -n 19 $PY extra.py a0check > out/a0check.log 2>&1
./guard.sh
nice -n 19 $PY con_sweep.py pass1 > out/con_sweep_p1.log 2>&1
nice -n 19 $PY con_sweep.py pass2 > out/con_sweep_p2.log 2>&1
echo DONE >> out/con_sweep_p2.log
