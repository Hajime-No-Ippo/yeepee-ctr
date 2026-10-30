#!/bin/bash
cd /workspace/huawei/v21diag; PY=/workspace/venv_v17/bin/python
idle(){ n=0; while [ $n -lt 3 ]; do if pgrep -f "v17run.py|seqrun|seq11|xlgb|ens17|mlp17|mlp_pair|mkcooc|mkmf|v19cut|screen_cut|prefilter" | grep -v $$ >/dev/null; then n=0; else n=$((n+1)); fi; sleep 10; done; }
for c in 23 25; do idle; echo "build $c $(date)"; $PY step3.py build $c B 4 2>&1 | grep -v Warn; done
for c in 23 25; do idle; echo "fit $c $(date)"; $PY step3.py fit $c B 4 2>&1 | grep -v Warn; done
$PY combine3.py B 2>&1 | grep -v Warn
echo STEP3_DONE $(date)
