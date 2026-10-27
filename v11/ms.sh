#!/bin/bash
# multi-seed full seq runs: ms.sh PREFIX THREADS SEED... (all concurrently), extra env passed through
cd /workspace/huawei/v11; export PYTHONUNBUFFERED=1 SQ_WORK=/workspace/huawei SQ_NTR=500000
P=$1; T=$2; shift 2
for s in "$@"; do sleep 20;
  mkdir -p $P/s$s
  ( t0=$(date +%s); SQ_SEED=$s SQ_THREADS=$T SQ_OUT=$P/s$s /workspace/venv/bin/python seq11.py f1 f2 val23 f3 f4 val25 f5 f6 val26 eval 2>&1 | grep -v " ep[0-9]" > logs/${P}_s$s.log; echo "wall $(( $(date +%s)-t0 ))s" >> logs/${P}_s$s.log ) &
done
wait; echo MS_DONE $P
