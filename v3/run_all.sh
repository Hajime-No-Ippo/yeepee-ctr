#!/usr/bin/env bash
# One-command rerun from the CSVs: parse -> features -> (optional validation) -> final fit -> submission -> validate.
# Usage: ./run_all.sh            (final submission only, uses out/best_config.json)
#        ./run_all.sh --validate (also baselines, ablation, val runs at cutoffs Jan25/Jan26)
set -euo pipefail
cd "$(dirname "$0")"
PY=/workspace/venv/bin/python
export POLARS_MAX_THREADS=${POLARS_MAX_THREADS:-6}
mkdir -p data feat logs out
$PY prep.py /workspace/huawei/data | tee logs/prep.log
$PY build_feats.py all --force | tee logs/build_feats.log
if [[ "${1:-}" == "--validate" ]]; then
  $PY baselines.py | tee logs/baselines.log
  $PY ablation.py | tee logs/ablation.log
  $PY train.py --cutoff 25 --save full25 | tee logs/val25.log
  $PY train.py --cutoff 26 --save full26 | tee logs/val26.log
  $PY train.py --cutoff 25 --obj lambdarank | tee logs/lambdarank25.log
  $PY segments.py full25 25 | tee logs/seg25.log; $PY segments.py full26 26 | tee logs/seg26.log
  $PY leakcheck.py notgt lag1d | tee logs/leakcheck.log
fi
$PY predict.py --out /workspace/huawei/submission_v2.csv | tee logs/predict.log
