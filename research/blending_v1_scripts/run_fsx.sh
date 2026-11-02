#!/bin/bash
# section-9 queue: one LightGBM screen at a time (2 threads, nice 19), after tower-block build finishes. C25 first then C23 per block.
cd /workspace/huawei/research/blending_v1_scripts
while [ ! -f feat9/tt_done.flag ]; do sleep 20; done
PY=/workspace/huawei/v20/.venv/bin/python
for B in ttbag gf ttbag+tth+ttr coAx prtm; do for C in 25 23; do
  [ -f out/fsx_${B}_c${C}_s0.npy ] && continue
  nice -n 19 $PY -u fsx.py $B $C 0 > out/log_fsx_${B}_c${C}.txt 2>&1
done; done
echo done > out/fsx_queue_done.flag
