#!/bin/bash
# screens: base_v5 + each group, cuts 25,26
cd /workspace/huawei/v5; source env.sh
until grep -q BUILT out/check_xA.log 2>/dev/null; do sleep 15; done
export XDIRS=feat_xA,feat_x2
for g in base A1 A2 A3; do
  if [ $g = base ]; then keep=""; tag=base_v5; else keep=$g; tag=scr_$g; fi
  D=$($PY groups.py "$keep")
  for c in 25 26; do
    grep -q "\"tag\": \"$tag\".*\"cut\": $c\|\"cut\": $c.*\"tag\": \"$tag\"" out/results.jsonl && continue
    ./guard.sh
    echo "=== $(date +%T) $tag $c drop=$D"
    STEP=A nice -n 19 $PY screen.py $c $tag --frac 0.25 --seeds 2 --rounds 300 --threads 3 --folds f --drop "$D" 2>&1 | grep -v Warning | tail -2
  done
done
echo SCREENS_DONE
