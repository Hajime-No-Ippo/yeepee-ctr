#!/bin/bash
cd /workspace/huawei/v21diag; PY=/workspace/venv_v17/bin/python
for c in 23 25; do $PY step3.py build $c A 3 2>&1 | grep -v "Warn\|nanmean\|subtract\|nanvar"; $PY step3.py fit $c A 3 2>&1 | grep -v "Warn\|subtract" | head -1; done
for v in B A; do $PY step3.py build 27 $v 3 2>&1 | grep -v "Warn\|nanmean\|subtract\|nanvar"; $PY step3.py fit 27 $v 3 2>&1 | grep -v "Warn\|subtract"| head -1; done
$PY combine3.py A 2>&1 | grep -v Warn
echo DONE3B
