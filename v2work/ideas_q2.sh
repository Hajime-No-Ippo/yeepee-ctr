#!/bin/bash
cd /workspace/huawei/v2work
P=/workspace/venv/bin/python
run(){ echo "=== $(date +%T) $*"; $P screen.py "$@" --threads 4 2>&1 | grep -v Warning | tail -4; }
for c in 25 26; do run $c c_usfix_notr --mods usfix,notr; run $c gap_c_usfix_notr --mods usfix,notr --maxfold 3; done
for c in 25 26; do run $c base_seed1 --seedoff 1; run $c gap_base_seed1 --maxfold 3 --seedoff 1; done
echo "=== done $(date +%T)"
