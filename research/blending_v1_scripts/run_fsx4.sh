#!/bin/bash
# section-9 queue v4: strictly one of my screens at a time (2 thr, nice 19); waits for any running fsx.py.
cd /workspace/huawei/research/blending_v1_scripts
PY=/workspace/huawei/v20/.venv/bin/python
for BCS in gf:23:1 gf:25:1 ttsc:23:1 coAx:25:0 coAx:23:0 prtm:25:0 prtm:23:0 ttbag+tth+ttr:25:0 ttbag+tth+ttr:23:0; do
  IFS=: read B C S <<< "$BCS"
  [ -f out/fsx_${B}_c${C}_s${S}.npy ] && continue
  while pgrep -f "fsx.py " >/dev/null; do sleep 20; done
  nice -n 19 $PY -u fsx.py $B $C $S > out/log_fsx_${B}_c${C}_s${S}.txt 2>&1
done
echo done > out/fsx_queue_done.flag
