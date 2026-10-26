#!/bin/bash
cd /workspace/huawei/v5; source env.sh
until grep -q PRECHECK_DONE out/slate_pre.log 2>/dev/null; do sleep 15; done
grep -q '"tag": "ctl_fix"' out/results.jsonl || { ./guard.sh; echo "=== $(date +%T) ctl_fix 25"; XDIRS=feat_x_fix,feat_x2 STEP=A nice -n 19 $PY screen.py 25 ctl_fix --frac 0.25 --seeds 2 --rounds 300 --threads 3 --folds f 2>&1 | grep -v Warning | tail -1; }
export XDIRS=feat_xA,feat_x2
P=$($PY pick.py passed); echo "passed=$P"
[ -z "$P" ] && { echo NO_GROUP_PASSED; echo A4_DONE; exit 0; }
if [[ "$P" == *,* ]]; then
  D=$($PY groups.py "$P")
  for c in 25 26; do
    grep -q "\"cut\": $c, \"tag\": \"scr_Astack\"" out/results.jsonl && continue
    ./guard.sh; echo "=== $(date +%T) scr_Astack $c keep=$P"
    STEP=A nice -n 19 $PY screen.py $c scr_Astack --frac 0.25 --seeds 2 --rounds 300 --threads 3 --folds f --drop "$D" 2>&1 | grep -v Warning | tail -1
  done
fi
F=$($PY pick.py final); echo "final=$F"; echo "$F" > out/final_groups.txt
D=$($PY groups.py "$F")
./guard.sh; echo "=== $(date +%T) confirm vA"
DROPF="$D" nice -n 19 $PY v5val.py vA A 25 26 23 > out/vA.log 2>&1
echo A4_DONE
