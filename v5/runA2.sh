#!/bin/bash
cd /workspace/huawei/v5; source env.sh
until grep -q DONE out/con_sweep_p2.log 2>/dev/null; do sleep 15; done
$PY select_sel.py write > out/select_sel.log 2>&1 || exit 1
./guard.sh
for n in f1 f2 f3 f4 f5 f6 val23 val25 val26 eval; do [ -f feat_xA/$n.parquet ] || FIX=1 NEW=1 SEL=/workspace/huawei/v5/out/SEL.json OUT=/workspace/huawei/v5/feat_xA nice -n 19 $PY extra.py $n; done > out/build_xA.log 2>&1
$PY check_xA.py > out/check_xA.log 2>&1 || exit 1
echo BUILT >> out/check_xA.log
