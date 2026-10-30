#!/usr/bin/env bash
# run_v17cooc.sh - rebuild v17cooc (Yeepee TechArena CTR) from raw train.csv + evaluation.csv. Resumable (stage markers in $WORK_DIR/c<cut>/.done).
#   DATA_DIR=/path/to/dir_with_csvs WORK_DIR=/path/to/scratch ./run_v17cooc.sh [--variant A|B|both] [--cut 27|25|both] [--threads 8] [--from STAGE] [--to STAGE] [--p33 P33_PATH] [--list] [--force]
# Defaults: DATA_DIR=./data  WORK_DIR=./work (relative to this directory), --variant both, --cut 27.
# Variant B is TRANSDUCTIVE (evaluation-history entries enter the co-occurrence graph; no labels). Nothing is ever uploaded; the CSVs under $WORK_DIR/c27/out are for manual submission.
# Steps (in order): setup prep_v3 prep_v2 feat_v3 feat_v2 feat_x feat_lr feat_tt feat_pp feat_sq feat_dw cooc_A cooc_B cooc_sel
#                   then per variant V: V_big V_s V_mlp0 V_mlp1 V_mlp2 V_blend [cut 27: V_write V_validate (V_probe if --p33/P33_PATH)]
set -euo pipefail
cd "$(dirname "$0")"
export DATA_DIR="${DATA_DIR:-$PWD/data}" WORK_DIR="${WORK_DIR:-$PWD/work}"
PY="${PYTHON:-python3}"
"$PY" - <<'PYC' || { echo "missing/old python packages: pip install -r requirements.txt (torch CPU: pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu)"; exit 1; }
import polars, numpy, scipy, sklearn, lightgbm, torch
PYC
exec "$PY" v17run.py --data "$DATA_DIR" --work "$WORK_DIR" "$@"
