#!/bin/bash
# usage: e.sh TAG "CUTS" ROUNDS SAVE NEGFRAC PARAMSJSON [LRDIRS] [extra ens args]   (8 threads, v10B drop)
cd /workspace/huawei/v11; P=/workspace/venv/bin/python; TAG=$1; CUTS=$2; R=$3; SV=$4; NF=$5; PJ=$6; LR=${7:-feat_lr2,feat_tt,feat_pp,feat_sq10}; shift $(( $# < 7 ? $# : 7 ))
export POLARS_MAX_THREADS=8 OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
for c in $CUTS; do
  t0=$(date +%s)
  $P ens.py $c $TAG --seeds ${SEEDS:-0} --threads 8 --lr $LR --negfrac $NF --nfolds ${NFOLDS:-4} --rounds $R --save $SV --params "$PJ" --drop '^sq_(pdot|pcos|mdot|mcos)$' "$@" 2>&1 | grep -v Warn | tail -3
  echo "{\"tag\":\"$TAG\",\"cut\":$c,\"wall_s\":$(( $(date +%s)-t0 ))}" >> timings.jsonl
done
