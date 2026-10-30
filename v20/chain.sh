#!/bin/bash
# v20 guarded chain (repaired 2026-10-30 after box restart). Every fitting stage waits (lib20.wait_idle) for other agents' heavy python jobs;
# taskset -c ${V20_CORES:-0-3}, 4 threads, one heavy job at a time. Resumable (results.jsonl + out/). A missing prerequisite or failed stage STOPS the chain loudly (exit != 0).
# s3_av is intentionally NOT run: av_flagged.json (0431, authoritative; backup out/av_flagged_0431.json) is preserved as-is.
cd /workspace/huawei/v20 || exit 9; P=/workspace/huawei/v20/.venv/bin/python; export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 POLARS_MAX_THREADS=4 OPENBLAS_NUM_THREADS=4
mkdir -p out logs
die(){ echo "$(date +%T) FATAL $*" | tee -a logs/chain.log; exit ${2:-1}; }
need(){ for f in "$@"; do [ -s "$f" ] || die "missing prerequisite $f" 3; done; }
run(){ echo "$(date +%T) START $1" | tee -a logs/chain.log; taskset -c ${V20_CORES:-0-3} $P $1.py >> logs/$1.log 2>&1; rc=$?; echo "$(date +%T) END $1 rc=$rc" | tee -a logs/chain.log; $P report.py >> logs/report.log 2>&1 || echo "$(date +%T) WARN report.py failed" | tee -a logs/chain.log; return $rc; }
stage(){ run "$1" || die "stage $1 failed (see logs/$1.log)" 1; }
[ -x $P ] || die "python $P missing" 3
need av_flagged.json out/av_flagged_0431.json gain_rank.json feats_ref484.json
[ -s out/redundancy.json ] || stage prefilter
need out/redundancy.json
stage preflight
stage s0_ref;        need out/ref.json
stage s2_importance; need out/importance.csv out/stable_feats.json
stage s4_curve;      need out/ranked_feats.json out/tiers_chosen.json
stage s5_grid;       need out/best_params.json
stage s6_tiers;      need out/tiers.csv
taskset -c ${V20_CORES:-0-3} $P decide.py >> logs/decide.log 2>&1; rc=$?; echo "$(date +%T) decide rc=$rc" | tee -a logs/chain.log
[ $rc -eq 0 ] || [ $rc -eq 2 ] || die "decide.py crashed rc=$rc" 1
need out/decision.json
stage s7_avdrop;     need out/avdrop.json     # A/B ONLY - s8 below still uses the undropped decided tier
if [ $rc -eq 0 ]; then stage s8_submit; need out/submission_info.json; else echo "$(date +%T) no tier qualified (decide rc=2): no submission" | tee -a logs/chain.log; fi
$P report.py >> logs/report.log 2>&1
echo "$(date +%T) CHAIN_DONE" | tee -a logs/chain.log
