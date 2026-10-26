#!/bin/bash
cd /workspace/huawei/v5; source env.sh
until grep -q FINAL_DONE out/runFinal.out 2>/dev/null; do sleep 15; done
grep -q "DECISION_NOT_PASSED" out/runFinal.out && grep -q '"kind": "decision"' out/results.jsonl && ! grep -q '"pass_": true' out/results.jsonl && { echo "A not accepted -> C skipped"; echo C_DONE; exit 0; }
./guard.sh; echo "=== $(date +%T) ppmi check"
grep -q '"kind": "ppmi_check"' out/results.jsonl || nice -n 19 $PY ppmi.py check > out/ppmi_check.log 2>&1
grep -q '"skip": true' out/ppmi_check.log && { echo "C1 skipped (corr>0.9)"; echo C_DONE; exit 0; }
./guard.sh; echo "=== $(date +%T) ppmi build"
for n in f1 f2 f3 f4 f5 f6 val23 val25 val26 eval; do [ -f feat_pp/$n.parquet ] || nice -n 19 $PY ppmi.py $n; done > out/build_pp.log 2>&1
export XDIRS=feat_xA,feat_x2,feat_pp
for c in 25 26; do
  grep -q "\"cut\": $c, \"tag\": \"scr_C1\"" out/results.jsonl && continue
  ./guard.sh; echo "=== $(date +%T) scr_C1 $c"
  STEP=C1 nice -n 19 $PY screen.py $c scr_C1 --frac 0.25 --seeds 2 --rounds 300 --threads 3 --folds f 2>&1 | grep -v Warning | tail -1 | cut -c1-300
done
echo C_DONE
