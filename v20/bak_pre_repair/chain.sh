#!/bin/bash
# v20 guarded chain. Every stage waits (lib20.wait_idle) for other agents' heavy python jobs; taskset -c ${V20_CORES:-0-3}, 4 threads, one heavy job at a time. Resumable (results.jsonl).
cd /workspace/huawei/v20; P=/workspace/huawei/v20/.venv/bin/python; export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 POLARS_MAX_THREADS=4 OPENBLAS_NUM_THREADS=4
run(){ echo "$(date +%T) START $1" | tee -a logs/chain.log; taskset -c ${V20_CORES:-0-3} $P $1.py >> logs/$1.log 2>&1; rc=$?; echo "$(date +%T) END $1 rc=$rc" | tee -a logs/chain.log; $P report.py; return $rc; }
run s3_av
run s0_ref || exit 1
run s2_importance || exit 1
run s4_curve || exit 1
run s5_grid || exit 1
run s6_tiers || exit 1
taskset -c ${V20_CORES:-0-3} $P decide.py >> logs/decide.log 2>&1; rc=$?; echo "$(date +%T) decide rc=$rc" | tee -a logs/chain.log
[ $rc -eq 0 ] && run s8_submit
echo "$(date +%T) CHAIN_DONE" | tee -a logs/chain.log
