#!/bin/bash
cd /workspace/huawei/research/blending_v1_scripts
for job in "tmx 23" "sq4 23" "tmx 25" "sq4 25"; do
  nice -n 19 ../../v20/.venv/bin/python fs490.py $job 0 >> fs_run.log 2>&1
done
echo ALLDONE >> fs_run.log
