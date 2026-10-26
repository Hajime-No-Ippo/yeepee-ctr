#!/bin/bash
# Wait while any non-v5 heavy job runs: the pattern list, OR any python process whose cwd/args are under /workspace/huawei/v2work (Grok Bot request, 17:07).
while true; do
  busy=""
  for p in $(pgrep -f 'ens.py|combo.py|screen.py|extra|v3fold|build_folds|rcov|seq.py|python'); do
    cwd=$(readlink /proc/$p/cwd 2>/dev/null); cmd=$(tr '\0' ' ' </proc/$p/cmdline 2>/dev/null)
    [[ "$cwd" == /workspace/huawei/v5* || "$cmd" == *huawei/v5* || "$cmd" == *pgrep* || "$cmd" == *guard.sh* ]] && continue
    if [[ "$cmd" =~ ens.py|combo.py|screen.py|extra|v3fold|build_folds|rcov|seq.py ]] || { [[ "$cmd" == *python* ]] && [[ "$cwd" == /workspace/huawei/v2work* || "$cmd" == */workspace/huawei/v2work* ]]; }; then
      busy="$busy $p:$cmd"
    fi
  done
  [ -z "$busy" ] && break
  echo "$(date +%T) waiting on:$busy"; sleep 60
done
