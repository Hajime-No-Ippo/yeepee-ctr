#!/bin/bash
cd /workspace/huawei/v2work
P=/workspace/venv/bin/python
run(){ echo "=== $(date +%T) $*"; $P screen.py "$@" --threads 4 2>&1 | grep -v Warning | tail -4; }
for c in 25 26; do run $c gap_base --maxfold 3; done
for m in usfix notr share hs; do
  for c in 25 26; do run $c i_$m --mods $m; run $c gap_$m --mods $m --maxfold 3; done
done
echo "=== done $(date +%T)"
