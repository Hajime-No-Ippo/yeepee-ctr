#!/bin/bash
cd /workspace/huawei/v12; export POLARS_MAX_THREADS=8 OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
for sc in "25 0" "23 0" "25 1" "23 1"; do set -- $sc
  t0=$(date +%s); /workspace/venv/bin/python /workspace/huawei/v12x/xens.py $1 xgb --seed $2 --rounds 600 --save 300,450,600 2>&1 | grep -v "^\s*\[bt\]" | tail -2
  echo "{\"tag\":\"xgb\",\"cut\":$1,\"seed\":$2,\"wall_s\":$(( $(date +%s)-t0 ))}" >> /workspace/huawei/v12x/ens/timings.jsonl
done; echo XGB_DONE
