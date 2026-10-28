#!/bin/bash
# feature retrains (dwbig recipe) after q2
while pgrep -f "run_q4.sh|run_xgb.sh" >/dev/null; do sleep 20; done
cd /workspace/huawei/v12; export POLARS_MAX_THREADS=8 OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
B=feat_lr2,feat_tt,feat_pp,feat_sqC,feat_dw
run(){ t0=$(date +%s); /workspace/venv/bin/python /workspace/huawei/v12x/xlgb.py "$@" 2>&1 | grep -v Warn | tail -2
  echo "{\"args\":\"$*\",\"wall_s\":$(( $(date +%s)-t0 ))}" >> /workspace/huawei/v12x/ens/timings.jsonl; }
for c in 25 23; do
 run $c f_prtm --lr $B,feat_prop,feat_tm --rounds 1000 --save 1000
 run $c f_tm --lr $B,feat_tm --rounds 1000 --save 1000
 run $c f_pr --lr $B,feat_prop --rounds 1000 --save 1000
 run $c f_tmx --lr $B,feat_tm,feat_tmx --rounds 1000 --save 1000
 run $c f_all --lr $B,feat_prop,feat_tm,feat_tmx --rounds 1000 --save 1000
done
echo Q3_DONE
