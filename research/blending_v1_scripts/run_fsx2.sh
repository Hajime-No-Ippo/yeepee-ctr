#!/bin/bash
# section-9 queue v2 (re-ordered after parent priority hint): one LightGBM screen at a time (2 thr, nice 19). Waits for any running fsx.py first.
cd /workspace/huawei/research/blending_v1_scripts
while pgrep -f "fsx.py " >/dev/null; do sleep 20; done
while [ ! -f feat9/tt2_done.flag ]; do sleep 20; done
PY=/workspace/huawei/v20/.venv/bin/python
for BC in ttbag:25 ttsc:25 ttbag:23 ttsc:23 gf:25 gf:23 ttbag+tth+ttr:25 ttbag+tth+ttr:23 coAx:25 coAx:23 prtm:25 prtm:23; do
  B=${BC%%:*}; C=${BC##*:}
  [ -f out/fsx_${B}_c${C}_s0.npy ] && continue
  nice -n 19 $PY -u fsx.py $B $C 0 > out/log_fsx_${B}_c${C}.txt 2>&1
done
echo done > out/fsx_queue_done.flag
