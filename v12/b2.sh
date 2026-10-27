#!/bin/bash
cd /workspace/huawei/v12; P=/workspace/venv/bin/python
export POLARS_MAX_THREADS=8 OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
for c in 25 23 26 27; do for s in 0 1 2; do
 [ $c = 26 ] && [ $s != 0 ] && continue
 t0=$(date +%s)
 $P mlp11.py $c dwmlp --ep 4 --seed $s --lr feat_lr2,feat_tt,feat_pp,feat_sqC,feat_dw 2>&1 | grep -v Warn | tail -3
 echo "{\"tag\":\"dwmlp\",\"cut\":$c,\"seed\":$s,\"wall_s\":$(( $(date +%s)-t0 ))}" >> timings.jsonl
done; done
echo B2_DONE
